from __future__ import annotations

import argparse
import os
from html import escape
from pathlib import Path

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "gradio is not installed in the current environment. "
        "Install it first, for example with: pip install gradio"
    ) from exc

from fewshot_app_backend import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DB_ROOT,
    DEFAULT_EN_KO_MODEL,
    DEFAULT_GPU_MEM_UTIL,
    DEFAULT_JSON_OUTPUT_ROOT,
    DEFAULT_KO_EN_MODEL,
    DEFAULT_PDF_OUTPUT_ROOT,
    DEFAULT_TEXT_OUTPUT_ROOT,
    DirectionConfig,
    FewshotAppBackend,
    METHOD_LABELS,
    build_default_direction_configs,
    extract_text_blocks_from_pdf,
    extract_text_entries_from_json,
)


def parse_direction_keys(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return ["ko_en", "en_ko"]
    direction_keys = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = [item for item in direction_keys if item not in {"ko_en", "en_ko"}]
    if invalid:
        raise ValueError(f"Unsupported direction key(s): {', '.join(invalid)}")
    if not direction_keys:
        raise ValueError("At least one direction must be selected.")
    return direction_keys


def parse_method_keys(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return ["fewshot_baseline", "segment_mt"]
    method_keys = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = [item for item in method_keys if item not in METHOD_LABELS]
    if invalid:
        raise ValueError(f"Unsupported method key(s): {', '.join(invalid)}")
    if not method_keys:
        raise ValueError("At least one method must be selected.")
    return method_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio app for MFDS translation methods")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--ko-en-model", type=str, default=DEFAULT_KO_EN_MODEL)
    parser.add_argument("--en-ko-model", type=str, default=DEFAULT_EN_KO_MODEL)
    parser.add_argument("--tokenizer-model", type=str, default=None)
    parser.add_argument("--ko-en-gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES value for the Korean->English worker")
    parser.add_argument("--en-ko-gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES value for the English->Korean worker")
    parser.add_argument("--db-root", type=str, default=DEFAULT_DB_ROOT)
    parser.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-fewshot", type=int, default=30)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument(
        "--directions",
        type=str,
        default="ko_en,en_ko",
        help="Comma-separated direction keys to enable: ko_en, en_ko",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="fewshot_baseline,segment_mt",
        help="Comma-separated method keys to enable: fewshot_baseline, segment_mt",
    )
    return parser.parse_args()


QUEUE_HEADERS = [
    "Job ID",
    "Method",
    "Direction",
    "Type",
    "State",
    "Queue Pos",
    "Segment Progress",
    "Segments",
    "Stage",
    "Submitted",
]

CLEARED_JOB_STATE = "__manual_clear__"
BROWSER_STATE_STORAGE_KEY = "mfds-fewshot-browser-state-v1"
BROWSER_STATE_SECRET = os.environ.get(
    "MFDS_GRADIO_BROWSER_STATE_SECRET",
    "mfds-fewshot-browser-state-secret-v1",
)

APP_CSS = """
.mfds-job-card,
.mfds-activity-banner {
    border: 1px solid #d1d5db;
    border-radius: 14px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    padding: 14px 16px;
    color: #0f172a;
}
.mfds-job-card.is-empty,
.mfds-activity-banner.idle {
    background: #f8fafc;
}
.mfds-job-title-row,
.mfds-activity-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 8px;
}
.mfds-job-lines,
.mfds-activity-lines {
    display: grid;
    gap: 4px;
    color: #334155;
    font-size: 0.95rem;
}
.mfds-job-card b,
.mfds-job-card strong,
.mfds-activity-banner b,
.mfds-activity-banner strong {
    color: #0f172a;
}
.mfds-progress-track {
    margin-top: 10px;
    width: 100%;
    height: 12px;
    border-radius: 999px;
    overflow: hidden;
    background: #e2e8f0;
}
.mfds-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
}
.mfds-progress-bar.is-active {
    position: relative;
    overflow: hidden;
}
.mfds-progress-bar.is-active::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.28) 25%,
        transparent 25%,
        transparent 50%,
        rgba(255, 255, 255, 0.28) 50%,
        rgba(255, 255, 255, 0.28) 75%,
        transparent 75%,
        transparent
    );
    background-size: 18px 18px;
    animation: mfds-stripes 1s linear infinite;
}
.mfds-state-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
    color: inherit;
}
.mfds-state-badge.running {
    background: #dbeafe;
    color: #1d4ed8;
}
.mfds-state-badge.queued {
    background: #fef3c7;
    color: #b45309;
}
.mfds-state-badge.cancelling {
    background: #fee2e2;
    color: #dc2626;
}
.mfds-state-badge.completed {
    background: #dcfce7;
    color: #166534;
}
.mfds-state-badge.failed {
    background: #fee2e2;
    color: #b91c1c;
}
.mfds-state-badge.cancelled,
.mfds-state-badge.idle {
    background: #e2e8f0;
    color: #475569;
}
.mfds-live-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: currentColor;
    animation: mfds-pulse 1.2s ease-in-out infinite;
}
.mfds-activity-meta {
    color: #475569;
    font-size: 0.9rem;
}
@keyframes mfds-pulse {
    0%, 100% { opacity: 0.35; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1); }
}
@keyframes mfds-stripes {
    from { background-position: 0 0; }
    to { background-position: 18px 0; }
}
"""


def get_state_badge(state: str) -> tuple[str, str, bool]:
    normalized = (state or "").strip().lower()
    mapping = {
        "queued": ("Queued", "queued", True),
        "running": ("Translating", "running", True),
        "cancelling": ("Cancelling", "cancelling", True),
        "completed": ("Completed", "completed", False),
        "failed": ("Failed", "failed", False),
        "cancelled": ("Cancelled", "cancelled", False),
    }
    return mapping.get(normalized, ("Idle", "idle", False))


def render_job_snapshot(snapshot: dict[str, object] | None) -> str:
    if not snapshot:
        return (
            "<div class=\"mfds-job-card is-empty\">"
            "<div class=\"mfds-job-title-row\">"
            "<b>Current Job</b>"
            "<span class=\"mfds-state-badge idle\">Idle</span>"
            "</div>"
            "<div class=\"mfds-job-lines\">No tracked job.</div>"
            "</div>"
        )

    raw_state = str(snapshot.get("state", "") or "")
    state = escape(raw_state)
    badge_label, badge_class, is_active = get_state_badge(raw_state)
    job_id = escape(str(snapshot.get("job_id", "")))
    method_label = escape(str(snapshot.get("method_label", "")))
    direction = escape(str(snapshot.get("direction_label", "")))
    progress_percent = float(snapshot.get("progress_percent", 0.0) or 0.0)
    segment_progress_percent = float(snapshot.get("segment_progress_percent", 0.0) or 0.0)
    completed_segments = int(snapshot.get("completed_segments", 0) or 0)
    total_segments = int(snapshot.get("total_segments", 0) or 0)
    stage = escape(str(snapshot.get("stage", "")))
    input_kind = escape(str(snapshot.get("input_kind", "")).upper())
    method_key = str(snapshot.get("method_key", "") or "")
    fewshot_count = int(snapshot.get("fewshot_count", 0) or 0)
    segment_window_size = int(snapshot.get("segment_window_size", 1) or 1)

    extra_line = ""
    if input_kind == "PDF":
        extra_line = (
            f"<div>Pages: {escape(str(snapshot.get('page_count', '')))}"
            f" | Blocks: {escape(str(snapshot.get('block_count', '')))}</div>"
        )

    method_detail_line = ""
    if method_key == "fewshot_baseline":
        method_detail_line = f"<div>Few-shot: {fewshot_count}</div>"
    elif method_key == "segment_mt":
        method_detail_line = f"<div>Segment window: {segment_window_size}</div>"

    download_line = ""
    if snapshot.get("translated_file_path"):
        download_line = "<div>Download: Ready</div>"

    badge_inner = escape(badge_label)
    if is_active:
        badge_inner = f"<span class=\"mfds-live-dot\"></span>{badge_inner}"

    progress_classes = "mfds-progress-bar"
    if is_active:
        progress_classes += " is-active"

    return (
        "<div class=\"mfds-job-card\">"
        "<div class=\"mfds-job-title-row\">"
        "<b>Current Job</b>"
        f"<span class=\"mfds-state-badge {badge_class}\">{badge_inner}</span>"
        "</div>"
        "<div class=\"mfds-job-lines\">"
        f"<div>ID: {job_id}</div>"
        f"<div>State: {state}</div>"
        f"<div>Method: {method_label} | Direction: {direction} | Type: {input_kind}</div>"
        f"{method_detail_line}"
        f"<div>Overall: {progress_percent:.1f}% | Segments: {segment_progress_percent:.1f}% "
        f"({completed_segments}/{total_segments})</div>"
        f"{extra_line}"
        f"{download_line}"
        "</div>"
        "<div class=\"mfds-progress-track\">"
        f"<div class=\"{progress_classes}\" style=\"width:{progress_percent:.1f}%;\"></div>"
        "</div>"
        f"<div class=\"mfds-activity-meta\" style=\"margin-top:8px;\">{stage}</div>"
        "</div>"
    )


def render_activity_banner(snapshot: dict[str, object] | None) -> str:
    if not snapshot:
        return (
            "<div class=\"mfds-activity-banner idle\">"
            "<div class=\"mfds-activity-head\">"
            "<strong>Translation Status</strong>"
            "<span class=\"mfds-state-badge idle\">Idle</span>"
            "</div>"
            "<div class=\"mfds-activity-lines\">"
            "<div>Ready for a new translation.</div>"
            "</div>"
            "</div>"
        )

    raw_state = str(snapshot.get("state", "") or "")
    badge_label, badge_class, is_active = get_state_badge(raw_state)
    stage = escape(str(snapshot.get("stage", "") or ""))
    progress_percent = float(snapshot.get("progress_percent", 0.0) or 0.0)
    completed_segments = int(snapshot.get("completed_segments", 0) or 0)
    total_segments = int(snapshot.get("total_segments", 0) or 0)
    download_ready = bool(snapshot.get("translated_file_path"))
    input_kind = str(snapshot.get("input_kind", "") or "").strip().lower()

    headline = "Translation Status"
    summary = (
        f"Overall {progress_percent:.1f}% | Segments {completed_segments}/{total_segments}"
    )
    extra_line = ""
    if raw_state == "completed" and download_ready:
        extra_line = "<div>Download file is ready.</div>"
    elif raw_state == "failed":
        extra_line = "<div>Check the status box for the latest error message.</div>"
    elif raw_state == "cancelled":
        extra_line = "<div>The translation was cancelled.</div>"
    elif raw_state == "queued":
        extra_line = "<div>The translation is preparing to start.</div>"
    elif raw_state == "cancelling":
        extra_line = "<div>The current batch will finish before cancellation is applied.</div>"
    elif raw_state == "running":
        if input_kind in {"pdf", "json"}:
            extra_line = "<div>Document translation updates live in the panes below.</div>"
        else:
            extra_line = "<div>Live translation preview will keep updating below.</div>"

    badge_inner = escape(badge_label)
    if is_active:
        badge_inner = f"<span class=\"mfds-live-dot\"></span>{badge_inner}"

    return (
        f"<div class=\"mfds-activity-banner {badge_class}\">"
        "<div class=\"mfds-activity-head\">"
        f"<strong>{headline}</strong>"
        f"<span class=\"mfds-state-badge {badge_class}\">{badge_inner}</span>"
        "</div>"
        "<div class=\"mfds-activity-lines\">"
        f"<div>{summary}</div>"
        f"<div>{stage}</div>"
        f"{extra_line}"
        "</div>"
        "</div>"
    )


def summarize_status(snapshot: dict[str, object] | None, fallback: str = "") -> str:
    if not snapshot:
        return fallback
    state = str(snapshot.get("state", ""))
    if state == "failed":
        error_text = str(snapshot.get("error", "")).strip()
        if error_text:
            lines = [line.strip() for line in error_text.splitlines() if line.strip()]
            if lines:
                return lines[-1]
        return "Job failed."
    if state == "completed":
        return f"Job {snapshot.get('job_id', '')} completed."
    if state == "cancelled":
        return f"Job {snapshot.get('job_id', '')} cancelled."
    return str(snapshot.get("stage", "") or fallback)


def build_cancellable_job_update(
    queue_rows: list[list[str]],
    preferred_job_id: str | None = None,
):
    choices: list[tuple[str, str]] = []
    valid_ids: list[str] = []
    for row in queue_rows:
        if len(row) < 6:
            continue
        job_id = str(row[0])
        state = str(row[4])
        if state not in {"queued", "running", "cancelling"}:
            continue
        label = f"{job_id} | {row[1]} | {row[2]} | {row[3]} | {state} | {row[6]}"
        choices.append((label, job_id))
        valid_ids.append(job_id)

    selected_value = None
    if preferred_job_id and preferred_job_id in valid_ids:
        selected_value = preferred_job_id
    elif valid_ids:
        selected_value = valid_ids[0]

    return gr.update(choices=choices, value=selected_value)


def build_demo(
    app_backend: FewshotAppBackend,
    direction_configs: dict[str, DirectionConfig],
    *,
    method_keys: list[str],
    max_fewshot: int,
) -> gr.Blocks:
    method_choices = [(METHOD_LABELS[key], key) for key in method_keys]
    default_method_key = method_keys[0]
    direction_order = ["ko_en", "en_ko"]
    active_configs = [direction_configs[key] for key in direction_order if key in direction_configs]
    direction_summary = " / ".join(config.label for config in active_configs)
    method_summary = " / ".join(METHOD_LABELS[key] for key in method_keys)
    direction_choices = []
    for config in active_configs:
        if config.key == "ko_en":
            direction_choices.append(("한 -> 영", config.key))
        else:
            direction_choices.append(("영 -> 한", config.key))
    default_direction_key = active_configs[0].key
    if len(active_configs) == 1:
        if active_configs[0].key == "ko_en":
            guidance_line = "Select the translation direction explicitly. This deployment currently serves only Korean -> English."
            manual_placeholder = "Paste Korean source text here."
        else:
            guidance_line = "Select the translation direction explicitly. This deployment currently serves only English -> Korean."
            manual_placeholder = "Paste English source text here."
    else:
        guidance_line = "Select the translation direction manually instead of using automatic language detection."
        manual_placeholder = "Paste source text here."

    browser_session_defaults = {
        "current_job_id": "",
        "manual_text": "",
        "direction_key": default_direction_key,
        "method_key": default_method_key,
        "fewshot_count": 3,
        "segment_window_size": 1,
    }
    preview_state_defaults = {
        "job_id": "",
        "completed_segments": -1,
        "state": "",
        "translated_file_path": "",
    }
    UNCHANGED = object()

    def update_method_controls(method_key: str, *, busy: bool = False) -> tuple[object, object]:
        normalized = (method_key or "").strip()
        is_fewshot = normalized == "fewshot_baseline"
        return (
            gr.update(visible=is_fewshot, interactive=not busy, value=None),
            gr.update(visible=not is_fewshot, interactive=not busy, value=None),
        )

    def normalize_browser_session(raw_session: object) -> dict[str, object]:
        normalized = dict(browser_session_defaults)
        if isinstance(raw_session, dict):
            normalized.update(raw_session)

        direction_key = str(normalized.get("direction_key", "") or "").strip()
        if direction_key not in direction_configs:
            direction_key = default_direction_key

        method_key = str(normalized.get("method_key", "") or "").strip()
        if method_key not in method_keys:
            method_key = default_method_key

        try:
            fewshot_count = int(normalized.get("fewshot_count", 3) or 3)
        except (TypeError, ValueError):
            fewshot_count = 3
        fewshot_count = max(0, min(max_fewshot, fewshot_count))

        try:
            segment_window_size = int(normalized.get("segment_window_size", 1) or 1)
        except (TypeError, ValueError):
            segment_window_size = 1
        segment_window_size = max(1, min(12, segment_window_size))

        current_job_id = str(normalized.get("current_job_id", "") or "")
        if current_job_id != CLEARED_JOB_STATE:
            current_job_id = current_job_id.strip()

        manual_text = str(normalized.get("manual_text", "") or "")

        normalized.update(
            {
                "current_job_id": current_job_id,
                "manual_text": manual_text,
                "direction_key": direction_key,
                "method_key": method_key,
                "fewshot_count": fewshot_count,
                "segment_window_size": segment_window_size,
            }
        )
        return normalized

    def build_browser_session(
        current_job_id: str,
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
    ) -> dict[str, object]:
        normalized = normalize_browser_session(
            {
                "current_job_id": current_job_id,
                "manual_text": manual_text,
                "direction_key": direction_key,
                "method_key": method_key,
                "fewshot_count": fewshot_count,
                "segment_window_size": segment_window_size,
            }
        )
        normalized["current_job_id"] = CLEARED_JOB_STATE if current_job_id == CLEARED_JOB_STATE else str(current_job_id or "").strip()
        normalized["manual_text"] = str(manual_text or "")
        return normalized

    def normalize_preview_state(raw_state: object) -> dict[str, object]:
        normalized = dict(preview_state_defaults)
        if isinstance(raw_state, dict):
            normalized.update(raw_state)

        try:
            completed_segments = int(normalized.get("completed_segments", -1) or -1)
        except (TypeError, ValueError):
            completed_segments = -1

        normalized.update(
            {
                "job_id": str(normalized.get("job_id", "") or "").strip(),
                "completed_segments": completed_segments,
                "state": str(normalized.get("state", "") or "").strip(),
                "translated_file_path": str(normalized.get("translated_file_path", "") or "").strip(),
            }
        )
        return normalized

    def build_preview_state(
        job_id: str,
        snapshot: dict[str, object] | None,
    ) -> dict[str, object]:
        if snapshot is None:
            return dict(preview_state_defaults)
        return normalize_preview_state(
            {
                "job_id": job_id or str(snapshot.get("job_id", "") or ""),
                "completed_segments": int(snapshot.get("completed_segments", 0) or 0),
                "state": str(snapshot.get("state", "") or ""),
                "translated_file_path": str(snapshot.get("translated_file_path", "") or ""),
            }
        )

    def restore_browser_session(
        browser_session: dict[str, object] | None,
    ) -> tuple[dict[str, object], str, str, str, str, int, int, str]:
        normalized = normalize_browser_session(browser_session)
        tracked_job_id = str(normalized.get("current_job_id", "") or "")
        manual_text = str(normalized.get("manual_text", "") or "")
        if tracked_job_id and tracked_job_id != CLEARED_JOB_STATE and not manual_text:
            resolved_job_id = app_backend.resolve_current_job_id(tracked_job_id)
            snapshot = app_backend.get_job_snapshot(resolved_job_id)
            if snapshot is not None and str(snapshot.get("input_kind", "") or "") == "text":
                manual_text = str(snapshot.get("extracted_text", "") or "")
                normalized["manual_text"] = manual_text
                normalized["current_job_id"] = resolved_job_id or tracked_job_id
        return (
            normalized,
            str(normalized.get("current_job_id", "") or ""),
            manual_text,
            str(normalized.get("direction_key", default_direction_key)),
            str(normalized.get("method_key", default_method_key)),
            int(normalized.get("fewshot_count", 3) or 3),
            int(normalized.get("segment_window_size", 1) or 1),
            manual_text,
        )

    def is_busy_snapshot(snapshot: dict[str, object] | None) -> bool:
        if snapshot is None:
            return False
        return str(snapshot.get("state", "") or "") in {"queued", "running", "cancelling"}

    def get_active_snapshot() -> tuple[str, dict[str, object] | None]:
        active_job_id = app_backend.resolve_current_job_id()
        snapshot = app_backend.get_job_snapshot(active_job_id)
        if is_busy_snapshot(snapshot):
            return active_job_id, snapshot
        return "", None

    def resolve_tracked_job(current_job_id: str | None) -> tuple[str, dict[str, object] | None]:
        if current_job_id == CLEARED_JOB_STATE:
            return "", None
        requested_job_id = str(current_job_id or "").strip()
        if requested_job_id:
            resolved_job_id = app_backend.resolve_current_job_id(requested_job_id)
            snapshot = app_backend.get_job_snapshot(resolved_job_id)
            if snapshot is not None:
                return resolved_job_id, snapshot
        return get_active_snapshot()

    def read_document_preview(input_file_path: str | None) -> tuple[str, str]:
        resolved_path = str(input_file_path or "").strip()
        if not resolved_path:
            return "", ""

        file_path = Path(resolved_path)
        file_suffix = file_path.suffix.lower()
        if file_suffix == ".pdf":
            _, page_count, blocks, extracted_text = extract_text_blocks_from_pdf(resolved_path)
            status_text = (
                f"Loaded {len(blocks)} PDF segments from {file_path.name} "
                f"across {page_count} page(s)."
            )
            return extracted_text, status_text
        if file_suffix == ".json":
            _, _, json_entries, extracted_text = extract_text_entries_from_json(resolved_path)
            status_text = f"Loaded {len(json_entries)} JSON text segments from {file_path.name}."
            return extracted_text, status_text
        raise ValueError("Only PDF and JSON files are supported.")

    def compose_ui_state(
        *,
        tracked_job_id: str,
        snapshot: dict[str, object] | None,
        manual_text: str,
        source_preview: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        status_text: str,
        translation_value: object = UNCHANGED,
        translated_file_value: object = UNCHANGED,
        input_file_value: object = UNCHANGED,
        manual_text_value: object = UNCHANGED,
    ) -> tuple[object, ...]:
        busy = is_busy_snapshot(snapshot)
        tracked_state_value = tracked_job_id
        if tracked_state_value != CLEARED_JOB_STATE:
            tracked_state_value = str(tracked_state_value or "").strip()
        resolved_job_id = tracked_state_value if tracked_state_value not in {"", CLEARED_JOB_STATE} else ""

        extracted_source = ""
        if snapshot is not None:
            extracted_source = str(snapshot.get("extracted_text", "") or "")
        source_value = extracted_source or str(source_preview or "") or str(manual_text or "")

        if translation_value is UNCHANGED:
            translation_output = gr.skip()
        else:
            translation_output = translation_value

        if translated_file_value is UNCHANGED:
            translated_file_output = gr.skip()
        else:
            translated_file_output = translated_file_value

        if input_file_value is UNCHANGED:
            input_file_update = gr.update(interactive=not busy)
        else:
            input_file_update = gr.update(value=input_file_value, interactive=not busy)

        if manual_text_value is UNCHANGED:
            manual_text_update = gr.update(interactive=not busy)
        else:
            manual_text_update = gr.update(value=manual_text_value, interactive=not busy)

        is_fewshot = method_key == "fewshot_baseline"
        direction_update = gr.update(interactive=not busy and len(direction_choices) > 1)
        method_update = gr.update(interactive=not busy and len(method_choices) > 1)
        fewshot_update = gr.update(
            value=fewshot_count,
            visible=is_fewshot,
            interactive=not busy,
        )
        segment_update = gr.update(
            value=segment_window_size,
            visible=not is_fewshot,
            interactive=not busy,
        )

        return (
            tracked_state_value,
            resolved_job_id,
            render_job_snapshot(snapshot),
            render_activity_banner(snapshot),
            status_text,
            source_value,
            translation_output,
            translated_file_output,
            build_preview_state(resolved_job_id, snapshot),
            build_browser_session(
                tracked_state_value,
                manual_text,
                direction_key,
                method_key,
                fewshot_count,
                segment_window_size,
            ),
            source_value,
            direction_update,
            method_update,
            fewshot_update,
            segment_update,
            input_file_update,
            manual_text_update,
            gr.update(interactive=not busy),
            gr.update(interactive=not busy),
            gr.update(interactive=busy),
            gr.update(interactive=not busy),
        )

    def refresh_ui(
        current_job_id: str,
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        preview_state: dict[str, object] | None,
        source_preview_state: str,
    ) -> tuple[object, ...]:
        current_preview_state = normalize_preview_state(preview_state)
        if current_job_id == CLEARED_JOB_STATE:
            return compose_ui_state(
                tracked_job_id=CLEARED_JOB_STATE,
                snapshot=None,
                manual_text=manual_text,
                source_preview=source_preview_state,
                direction_key=direction_key,
                method_key=method_key,
                fewshot_count=fewshot_count,
                segment_window_size=segment_window_size,
                status_text="",
                translation_value="",
                translated_file_value=None,
            )

        resolved_job_id, snapshot = resolve_tracked_job(current_job_id)
        if snapshot is None:
            return compose_ui_state(
                tracked_job_id="",
                snapshot=None,
                manual_text=manual_text,
                source_preview=source_preview_state,
                direction_key=direction_key,
                method_key=method_key,
                fewshot_count=fewshot_count,
                segment_window_size=segment_window_size,
                status_text="",
            )

        status_text = summarize_status(snapshot)
        next_preview_state = build_preview_state(resolved_job_id or "", snapshot)
        preview_changed = (
            str(current_preview_state.get("job_id", "") or "") != str(next_preview_state.get("job_id", "") or "")
            or int(current_preview_state.get("completed_segments", -1) or -1)
            != int(next_preview_state.get("completed_segments", -1) or -1)
            or str(current_preview_state.get("state", "") or "") != str(next_preview_state.get("state", "") or "")
            or str(current_preview_state.get("translated_file_path", "") or "")
            != str(next_preview_state.get("translated_file_path", "") or "")
        )

        snapshot_translation = str(snapshot.get("translation", "") or "")
        translation_value: object = UNCHANGED
        if preview_changed:
            translation_value = snapshot_translation
        elif snapshot_translation and str(snapshot.get("state", "") or "") in {"completed", "failed", "cancelled"}:
            translation_value = snapshot_translation

        snapshot_file_path = snapshot.get("translated_file_path") or None
        translated_file_value: object = UNCHANGED
        if preview_changed:
            translated_file_value = snapshot_file_path
        elif snapshot_file_path and str(snapshot.get("state", "") or "") == "completed":
            translated_file_value = snapshot_file_path

        return compose_ui_state(
            tracked_job_id=resolved_job_id or "",
            snapshot=snapshot,
            manual_text=manual_text,
            source_preview=source_preview_state,
            direction_key=direction_key,
            method_key=method_key,
            fewshot_count=fewshot_count,
            segment_window_size=segment_window_size,
            status_text=status_text,
            translation_value=translation_value,
            translated_file_value=translated_file_value,
        )

    def preview_manual_input(
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        preview_state: dict[str, object] | None,
    ) -> tuple[object, ...]:
        return refresh_ui(
            CLEARED_JOB_STATE,
            manual_text,
            direction_key,
            method_key,
            fewshot_count,
            segment_window_size,
            preview_state,
            manual_text,
        )

    def preview_document_input(
        input_file_path: str | None,
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        preview_state: dict[str, object] | None,
    ) -> tuple[object, ...]:
        try:
            source_preview, status_text = read_document_preview(input_file_path)
        except Exception as exc:
            source_preview = ""
            status_text = str(exc)

        result = list(
            refresh_ui(
                CLEARED_JOB_STATE,
                manual_text,
                direction_key,
                method_key,
                fewshot_count,
                segment_window_size,
                preview_state,
                source_preview,
            )
        )
        result[4] = status_text
        return tuple(result)

    def submit_text_job(
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        preview_state: dict[str, object] | None,
    ) -> tuple[object, ...]:
        active_job_id, active_snapshot = get_active_snapshot()
        if active_snapshot is not None:
            result = list(
                refresh_ui(
                    active_job_id,
                    manual_text,
                    direction_key,
                    method_key,
                    fewshot_count,
                    segment_window_size,
                    preview_state,
                    manual_text,
                )
            )
            result[4] = (
                f"Job {active_job_id} is already running. "
                "Wait for it to finish or cancel it first."
            )
            return tuple(result)

        try:
            job_id = app_backend.submit_text_job(
                manual_text,
                fewshot_count,
                direction_key=direction_key,
                method_key=method_key,
                segment_window_size=segment_window_size,
            )
        except Exception as exc:
            result = list(
                refresh_ui(
                    CLEARED_JOB_STATE,
                    manual_text,
                    direction_key,
                    method_key,
                    fewshot_count,
                    segment_window_size,
                    preview_state,
                    manual_text,
                )
            )
            result[4] = str(exc)
            return tuple(result)

        result = list(
            refresh_ui(
                job_id,
                manual_text,
                direction_key,
                method_key,
                fewshot_count,
                segment_window_size,
                preview_state,
                manual_text,
            )
        )
        result[4] = f"Job {job_id} submitted."
        result[7] = None
        result[15] = gr.update(value=None, interactive=False)
        return tuple(result)

    def submit_file_job(
        input_file_path: str | None,
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        preview_state: dict[str, object] | None,
        source_preview_state: str,
    ) -> tuple[object, ...]:
        active_job_id, active_snapshot = get_active_snapshot()
        if active_snapshot is not None:
            result = list(
                refresh_ui(
                    active_job_id,
                    manual_text,
                    direction_key,
                    method_key,
                    fewshot_count,
                    segment_window_size,
                    preview_state,
                    source_preview_state,
                )
            )
            result[4] = (
                f"Job {active_job_id} is already running. "
                "Wait for it to finish or cancel it first."
            )
            return tuple(result)

        source_preview = source_preview_state
        if not source_preview:
            try:
                source_preview, _ = read_document_preview(input_file_path)
            except Exception:
                source_preview = ""

        try:
            resolved_path = str(input_file_path or "").strip()
            if not resolved_path:
                raise ValueError("Upload a PDF or JSON file first.")
            file_suffix = Path(resolved_path).suffix.lower()
            if file_suffix == ".pdf":
                job_id = app_backend.submit_pdf_job(
                    resolved_path,
                    fewshot_count,
                    direction_key=direction_key,
                    method_key=method_key,
                    segment_window_size=segment_window_size,
                )
            elif file_suffix == ".json":
                job_id = app_backend.submit_json_job(
                    resolved_path,
                    fewshot_count,
                    direction_key=direction_key,
                    method_key=method_key,
                    segment_window_size=segment_window_size,
                )
            else:
                raise ValueError("Only PDF and JSON files are supported.")
        except Exception as exc:
            result = list(
                refresh_ui(
                    CLEARED_JOB_STATE,
                    manual_text,
                    direction_key,
                    method_key,
                    fewshot_count,
                    segment_window_size,
                    preview_state,
                    source_preview,
                )
            )
            result[4] = str(exc)
            return tuple(result)

        result = list(
            refresh_ui(
                job_id,
                manual_text,
                direction_key,
                method_key,
                fewshot_count,
                segment_window_size,
                preview_state,
                source_preview,
            )
        )
        result[4] = f"Job {job_id} submitted."
        result[15] = gr.update(value=None, interactive=False)
        return tuple(result)

    def cancel_current_job(
        current_job_id: str,
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        preview_state: dict[str, object] | None,
        source_preview_state: str,
    ) -> tuple[object, ...]:
        resolved_job_id, snapshot = resolve_tracked_job(current_job_id)
        if snapshot is None:
            result = list(
                refresh_ui(
                    CLEARED_JOB_STATE,
                    manual_text,
                    direction_key,
                    method_key,
                    fewshot_count,
                    segment_window_size,
                    preview_state,
                    source_preview_state,
                )
            )
            result[4] = "No running translation to cancel."
            return tuple(result)

        try:
            status_message = app_backend.cancel_job(resolved_job_id)
        except Exception as exc:
            result = list(
                refresh_ui(
                    resolved_job_id,
                    manual_text,
                    direction_key,
                    method_key,
                    fewshot_count,
                    segment_window_size,
                    preview_state,
                    source_preview_state,
                )
            )
            result[4] = str(exc)
            return tuple(result)

        result = list(
            refresh_ui(
                resolved_job_id,
                manual_text,
                direction_key,
                method_key,
                fewshot_count,
                segment_window_size,
                preview_state,
                source_preview_state,
            )
        )
        result[4] = status_message
        return tuple(result)

    def clear_form(
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
    ) -> tuple[object, ...]:
        return compose_ui_state(
            tracked_job_id=CLEARED_JOB_STATE,
            snapshot=None,
            manual_text="",
            source_preview="",
            direction_key=direction_key,
            method_key=method_key,
            fewshot_count=fewshot_count,
            segment_window_size=segment_window_size,
            status_text="",
            translation_value="",
            translated_file_value=None,
            input_file_value=None,
            manual_text_value="",
        )

    with gr.Blocks(title="MFDS Translation App") as demo:
        gr.Markdown(
            "\n".join(
                [
                    "# MFDS Translation App",
                    f"Enabled direction(s): {direction_summary}",
                    f"Enabled method(s): {method_summary}",
                    guidance_line,
                    "Only one translation runs at a time. While a job is active, new input is locked until it completes or is cancelled.",
                    "Document uploads show extracted segments in Source, and the translated segments appear live in Translation.",
                    "PDF uploads no longer render a translated PDF. They stay as segment-based text output in the Translation pane.",
                    "JSON uploads still keep their translated JSON export file.",
                    "Scanned PDFs require OCR and are not supported here.",
                ]
            )
        )

        current_job_state = gr.State("")
        preview_state = gr.State(dict(preview_state_defaults))
        source_preview_state = gr.State("")
        browser_state = gr.BrowserState(
            default_value=browser_session_defaults,
            storage_key=BROWSER_STATE_STORAGE_KEY,
            secret=BROWSER_STATE_SECRET,
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=180):
                direction_radio = gr.Radio(
                    choices=direction_choices,
                    value=default_direction_key,
                    label="Direction",
                    interactive=len(direction_choices) > 1,
                )
            with gr.Column(scale=1, min_width=180):
                method_radio = gr.Radio(
                    choices=method_choices,
                    value=default_method_key,
                    label="Method",
                    interactive=len(method_choices) > 1,
                )

        with gr.Row():
            fewshot_slider = gr.Slider(
                minimum=0,
                maximum=max_fewshot,
                value=3,
                step=1,
                label="Few-shot examples per segment",
                visible=default_method_key == "fewshot_baseline",
            )
            segment_window_slider = gr.Slider(
                minimum=1,
                maximum=12,
                value=1,
                step=1,
                label="Segment Window Size",
                visible=default_method_key == "segment_mt",
            )

        with gr.Tabs():
            with gr.Tab("Document Upload"):
                input_file = gr.File(
                    label="Document",
                    file_types=[".pdf", ".json"],
                    type="filepath",
                )
                translate_file_button = gr.Button("Translate Document", variant="primary")
            with gr.Tab("Direct Input"):
                manual_text_box = gr.Textbox(
                    label="Direct Input",
                    lines=8,
                    placeholder=manual_placeholder,
                )
                translate_text_button = gr.Button("Translate Text", variant="primary")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                source_box = gr.Textbox(
                    label="Source",
                    lines=22,
                    interactive=False,
                )
            with gr.Column(scale=1, min_width=320):
                target_box = gr.Textbox(
                    label="Translation",
                    lines=22,
                    interactive=False,
                )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                activity_box = gr.HTML(value=render_activity_banner(None))
            with gr.Column(scale=1, min_width=320):
                current_job_box = gr.HTML(
                    value=render_job_snapshot(None),
                    label="Current Job",
                )

        with gr.Row():
            cancel_button = gr.Button("Cancel Current Job")
            clear_button = gr.Button("Clear Form")

        with gr.Row():
            with gr.Column(scale=2):
                status_box = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                current_job_id_box = gr.Textbox(label="Current Job ID", interactive=False)
            with gr.Column(scale=1):
                translated_output_file = gr.File(
                    label="Download Result File",
                    interactive=False,
                )

        refresh_timer = gr.Timer(value=1.0, active=True)

        standard_outputs = [
            current_job_state,
            current_job_id_box,
            current_job_box,
            activity_box,
            status_box,
            source_box,
            target_box,
            translated_output_file,
            preview_state,
            browser_state,
            source_preview_state,
            direction_radio,
            method_radio,
            fewshot_slider,
            segment_window_slider,
            input_file,
            manual_text_box,
            translate_file_button,
            translate_text_button,
            cancel_button,
            clear_button,
        ]

        load_event = demo.load(
            fn=restore_browser_session,
            inputs=[browser_state],
            outputs=[
                browser_state,
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                source_preview_state,
            ],
        )
        load_event.then(
            fn=refresh_ui,
            inputs=[
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
        )

        direction_radio.change(
            fn=refresh_ui,
            inputs=[
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        method_radio.change(
            fn=refresh_ui,
            inputs=[
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        fewshot_slider.change(
            fn=refresh_ui,
            inputs=[
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        segment_window_slider.change(
            fn=refresh_ui,
            inputs=[
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        manual_text_box.change(
            fn=preview_manual_input,
            inputs=[
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        input_file.change(
            fn=preview_document_input,
            inputs=[
                input_file,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        translate_text_button.click(
            fn=submit_text_job,
            inputs=[
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        manual_text_box.submit(
            fn=submit_text_job,
            inputs=[
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        translate_file_button.click(
            fn=submit_file_job,
            inputs=[
                input_file,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        cancel_button.click(
            fn=cancel_current_job,
            inputs=[
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        clear_button.click(
            fn=clear_form,
            inputs=[direction_radio, method_radio, fewshot_slider, segment_window_slider],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        refresh_timer.tick(
            fn=refresh_ui,
            inputs=[
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )

    return demo


def main() -> None:
    args = parse_args()
    selected_direction_keys = parse_direction_keys(args.directions)
    selected_method_keys = parse_method_keys(args.methods)
    all_direction_configs = build_default_direction_configs(
        ko_en_gpu=args.ko_en_gpu,
        en_ko_gpu=args.en_ko_gpu,
        ko_en_model=args.ko_en_model,
        en_ko_model=args.en_ko_model,
        tokenizer_model=args.tokenizer_model,
        db_root=args.db_root,
        gpu_mem_util=args.gpu_mem_util,
        batch_size=args.batch_size,
    )
    direction_configs = {
        key: all_direction_configs[key]
        for key in selected_direction_keys
    }
    app_backend = FewshotAppBackend(
        direction_configs,
        method_keys=selected_method_keys,
        startup_timeout_s=args.startup_timeout,
        request_timeout_s=args.request_timeout,
    )
    demo = build_demo(app_backend, direction_configs, method_keys=selected_method_keys, max_fewshot=args.max_fewshot)
    allowed_paths = [
        str(DEFAULT_TEXT_OUTPUT_ROOT.resolve()),
        str(DEFAULT_PDF_OUTPUT_ROOT.resolve()),
        str(DEFAULT_JSON_OUTPUT_ROOT.resolve()),
    ]
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
