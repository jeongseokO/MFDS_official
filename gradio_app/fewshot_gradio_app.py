from __future__ import annotations

import argparse
import hashlib
import os
import queue
import threading
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
    DEFAULT_RETRIEVAL_BACKEND,
    DEFAULT_STREAMING_TRANSLATION,
    DEFAULT_TEXT_OUTPUT_ROOT,
    DirectionConfig,
    FewshotAppBackend,
    METHOD_LABELS,
    RETRIEVAL_BACKEND_LABELS,
    build_default_direction_configs,
    extract_text_blocks_from_pdf,
    extract_text_entries_from_json,
    segment_input_text,
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
FEWSHOT_PREVIEW_SEGMENT_LIMIT = 3
FEWSHOT_EDITOR_HEADERS = [
    "Segment #",
    "Example #",
    "Input segment",
    "Few-shot source",
    "Few-shot target",
]
REFRESH_INTERVAL_SECONDS = float(os.environ.get("MFDS_GRADIO_REFRESH_INTERVAL", "0.25"))

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
.mfds-fewshot-panel {
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: #ffffff;
    padding: 12px 14px;
    color: #0f172a;
}
.mfds-fewshot-panel summary {
    cursor: pointer;
    font-weight: 700;
}
.mfds-fewshot-panel ul {
    margin: 8px 0 0;
    padding-left: 20px;
}
.mfds-fewshot-panel li {
    margin: 6px 0;
}
.mfds-fewshot-query {
    margin-top: 6px;
    padding: 8px 10px;
    border-radius: 6px;
    background: #f8fafc;
    color: #111827;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.mfds-fewshot-table {
    width: 100%;
    margin-top: 8px;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 0.92rem;
}
.mfds-fewshot-table th,
.mfds-fewshot-table td {
    border: 1px solid #e2e8f0;
    padding: 7px 8px;
    vertical-align: top;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}
.mfds-fewshot-table th {
    background: #f1f5f9;
    color: #334155;
    font-weight: 700;
}
.mfds-fewshot-table .rank {
    width: 44px;
    text-align: center;
}
.mfds-fewshot-meta {
    color: #475569;
    font-size: 0.9rem;
}
.mfds-fewshot-src {
    color: #334155;
}
.mfds-fewshot-mt {
    color: #111827;
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
    retrieval_backend = str(snapshot.get("retrieval_backend", "") or "")
    retrieval_label = RETRIEVAL_BACKEND_LABELS.get(retrieval_backend, retrieval_backend)
    streaming_label = "On" if bool(snapshot.get("streaming_enabled", DEFAULT_STREAMING_TRANSLATION)) else "Off"

    extra_line = ""
    if input_kind == "PDF":
        extra_line = (
            f"<div>Pages: {escape(str(snapshot.get('page_count', '')))}"
            f" | Blocks: {escape(str(snapshot.get('block_count', '')))}</div>"
        )

    method_detail_line = ""
    if method_key == "fewshot_baseline":
        method_detail_line = (
            f"<div>Few-shot: {fewshot_count} | Retriever: {escape(retrieval_label)}"
            f" | Streaming: {streaming_label}</div>"
        )
    elif method_key == "segment_mt":
        method_detail_line = f"<div>Segment window: {segment_window_size} | Streaming: {streaming_label}</div>"

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
    retrieval_choices = [(label, key) for key, label in RETRIEVAL_BACKEND_LABELS.items()]
    configured_retrieval_backend = "faiss" if DEFAULT_RETRIEVAL_BACKEND == "bge" else DEFAULT_RETRIEVAL_BACKEND
    default_retrieval_backend = (
        configured_retrieval_backend
        if configured_retrieval_backend in RETRIEVAL_BACKEND_LABELS
        else retrieval_choices[0][1]
    )
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
        "retrieval_backend": default_retrieval_backend,
        "show_fewshot_examples": True,
    }
    preview_state_defaults = {
        "job_id": "",
        "completed_segments": -1,
        "translation_revision": -1,
        "state": "",
        "translated_file_path": "",
    }
    UNCHANGED = object()
    fewshot_preview_cache: dict[tuple[str, int, str, str], tuple[str, list[list[object]]]] = {}

    def compact_preview_text(value: object) -> str:
        return str(value or "").strip()

    def render_fewshot_preview_html(
        preview_rows: list[dict[str, object]],
        *,
        total_segments: int,
        fewshot_count: int,
    ) -> str:
        if fewshot_count <= 0:
            items = ["<li>Few-shot is set to 0.</li>"]
        elif not preview_rows:
            items = ["<li>No source segment is available.</li>"]
        else:
            items = []
            for segment_index, row in enumerate(preview_rows, start=1):
                segment = compact_preview_text(row.get("segment", ""))
                examples = row.get("examples", [])
                if not isinstance(examples, list) or not examples:
                    items.append(
                        "<li>"
                        f"<b>Input segment {segment_index}</b>"
                        f"<div class=\"mfds-fewshot-query\">{escape(segment)}</div>"
                        "<div class=\"mfds-fewshot-meta\">No retrieved examples.</div>"
                        "</li>"
                    )
                    continue
                table_rows = []
                for example_index, example in enumerate(examples, start=1):
                    if not isinstance(example, dict):
                        continue
                    src = escape(compact_preview_text(example.get("src", "")))
                    mt = escape(compact_preview_text(example.get("mt", "")))
                    table_rows.append(
                        "<tr>"
                        f"<td class=\"rank\">{example_index}</td>"
                        f"<td class=\"mfds-fewshot-src\">{src}</td>"
                        f"<td class=\"mfds-fewshot-mt\">{mt}</td>"
                        "</tr>"
                    )
                items.append(
                    "<li>"
                    f"<b>Input segment {segment_index}</b>"
                    f"<div class=\"mfds-fewshot-query\">{escape(segment)}</div>"
                    "<table class=\"mfds-fewshot-table\">"
                    "<thead><tr><th class=\"rank\">#</th><th>Few-shot source</th><th>Few-shot target</th></tr></thead>"
                    "<tbody>"
                    + "".join(table_rows)
                    + "</tbody></table>"
                    "</li>"
                )

        if total_segments > FEWSHOT_PREVIEW_SEGMENT_LIMIT:
            meta = (
                f"Showing {FEWSHOT_PREVIEW_SEGMENT_LIMIT} of {total_segments} source segments."
            )
        else:
            meta = f"Source segments: {total_segments}"

        return (
            "<details class=\"mfds-fewshot-panel\" open>"
            "<summary>Retrieved Few-Shot Examples</summary>"
            f"<div class=\"mfds-fewshot-meta\">{escape(meta)}</div>"
            "<ul>"
            + "".join(items)
            + "</ul>"
            "</details>"
        )

    def build_fewshot_editor_rows(preview_rows: list[dict[str, object]]) -> list[list[object]]:
        rows: list[list[object]] = []
        for segment_index, row in enumerate(preview_rows, start=1):
            segment = compact_preview_text(row.get("segment", ""))
            examples = row.get("examples", [])
            if not isinstance(examples, list):
                continue
            for example_index, example in enumerate(examples, start=1):
                if not isinstance(example, dict):
                    continue
                rows.append(
                    [
                        segment_index,
                        example_index,
                        segment,
                        compact_preview_text(example.get("src", "")),
                        compact_preview_text(example.get("mt", "")),
                    ]
                )
        return rows

    def build_fewshot_preview_update(
        *,
        source_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        retrieval_backend: str,
        show_examples: bool,
        busy: bool,
    ) -> tuple[object, object]:
        if method_key != "fewshot_baseline":
            return gr.update(visible=False, value=""), gr.update(visible=False, value=[])
        if not show_examples:
            return gr.update(visible=False), gr.update(visible=False)
        if busy:
            return gr.update(visible=True), gr.update(visible=True, interactive=False)

        normalized_source = str(source_text or "").strip()
        if normalized_source.startswith("Preparing source preview:"):
            return gr.update(
                visible=True,
                value=render_fewshot_preview_html([], total_segments=0, fewshot_count=fewshot_count),
            ), gr.update(visible=False, value=[])

        try:
            normalized_count = max(0, int(fewshot_count))
        except (TypeError, ValueError):
            normalized_count = 0

        if not normalized_source or normalized_count <= 0:
            return gr.update(
                visible=True,
                value=render_fewshot_preview_html([], total_segments=0, fewshot_count=normalized_count),
            ), gr.update(visible=False, value=[])

        source_hash = hashlib.sha1(normalized_source.encode("utf-8")).hexdigest()
        cache_key = (direction_key, normalized_count, str(retrieval_backend or ""), source_hash)
        cached_preview = fewshot_preview_cache.get(cache_key)
        if cached_preview is not None:
            cached_html, cached_rows = cached_preview
            return (
                gr.update(visible=True, value=cached_html),
                gr.update(visible=True, value=cached_rows, interactive=not busy),
            )

        try:
            preview_rows = app_backend.preview_fewshot_examples(
                normalized_source,
                direction_key,
                normalized_count,
                retrieval_backend=retrieval_backend,
                max_segments=FEWSHOT_PREVIEW_SEGMENT_LIMIT,
            )
            total_segments = len(segment_input_text(normalized_source)[0])
            html = render_fewshot_preview_html(
                preview_rows,
                total_segments=total_segments,
                fewshot_count=normalized_count,
            )
            editor_rows = build_fewshot_editor_rows(preview_rows)
        except Exception as exc:
            html = (
                "<details class=\"mfds-fewshot-panel\" open>"
                "<summary>Retrieved Few-Shot Examples</summary>"
                "<ul>"
                f"<li>{escape(str(exc))}</li>"
                "</ul>"
                "</details>"
            )
            editor_rows = []
        fewshot_preview_cache[cache_key] = (html, editor_rows)
        if len(fewshot_preview_cache) > 32:
            fewshot_preview_cache.pop(next(iter(fewshot_preview_cache)))
        return (
            gr.update(visible=True, value=html),
            gr.update(visible=bool(editor_rows), value=editor_rows, interactive=not busy),
        )

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

        retrieval_backend = str(normalized.get("retrieval_backend", "") or "").strip()
        if retrieval_backend not in RETRIEVAL_BACKEND_LABELS:
            retrieval_backend = default_retrieval_backend

        raw_show_fewshot_examples = normalized.get("show_fewshot_examples", True)
        if isinstance(raw_show_fewshot_examples, str):
            show_fewshot_examples = raw_show_fewshot_examples.strip().lower() not in {"0", "false", "no", "off"}
        else:
            show_fewshot_examples = bool(raw_show_fewshot_examples)

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
                "retrieval_backend": retrieval_backend,
                "show_fewshot_examples": show_fewshot_examples,
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
        retrieval_backend: str,
        show_fewshot_examples: bool,
    ) -> dict[str, object]:
        normalized = normalize_browser_session(
            {
                "current_job_id": current_job_id,
                "manual_text": manual_text,
                "direction_key": direction_key,
                "method_key": method_key,
                "fewshot_count": fewshot_count,
                "segment_window_size": segment_window_size,
                "retrieval_backend": retrieval_backend,
                "show_fewshot_examples": show_fewshot_examples,
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
        try:
            translation_revision = int(normalized.get("translation_revision", -1) or -1)
        except (TypeError, ValueError):
            translation_revision = -1

        normalized.update(
            {
                "job_id": str(normalized.get("job_id", "") or "").strip(),
                "completed_segments": completed_segments,
                "translation_revision": translation_revision,
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
                "translation_revision": int(snapshot.get("translation_revision", 0) or 0),
                "state": str(snapshot.get("state", "") or ""),
                "translated_file_path": str(snapshot.get("translated_file_path", "") or ""),
            }
        )

    def restore_browser_session(
        browser_session: dict[str, object] | None,
    ) -> tuple[dict[str, object], str, str, str, str, int, int, str, bool, str]:
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
            str(normalized.get("retrieval_backend", default_retrieval_backend)),
            bool(normalized.get("show_fewshot_examples", True)),
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

    def resolve_uploaded_file_path(input_file_value: object) -> str:
        if input_file_value is None:
            return ""
        if isinstance(input_file_value, (str, Path)):
            return str(input_file_value).strip()
        if isinstance(input_file_value, dict):
            for key in ("path", "name"):
                value = input_file_value.get(key)
                if value:
                    return str(value).strip()
        name = getattr(input_file_value, "name", None)
        if name:
            return str(name).strip()
        path = getattr(input_file_value, "path", None)
        if path:
            return str(path).strip()
        return str(input_file_value or "").strip()

    def format_document_loading_source(file_name: str, percent: float, description: str) -> str:
        normalized_percent = max(0, min(100, int(round(percent * 100))))
        return "\n".join(
            [
                f"Preparing source preview: {file_name}",
                "",
                f"{normalized_percent}% - {description}",
                "",
                "Extracted text will appear here automatically when processing completes.",
            ]
        )

    def read_document_preview(input_file_path: object, progress_callback=None) -> tuple[str, str]:
        resolved_path = resolve_uploaded_file_path(input_file_path)
        if not resolved_path:
            return "", ""

        file_path = Path(resolved_path)
        file_suffix = file_path.suffix.lower()
        if file_suffix == ".pdf":
            _, page_count, blocks, extracted_text = extract_text_blocks_from_pdf(
                resolved_path,
                progress_callback=progress_callback,
            )
            status_text = (
                f"Loaded {len(blocks)} PDF segments from {file_path.name} "
                f"across {page_count} page(s)."
            )
            return extracted_text, status_text
        if file_suffix == ".json":
            if progress_callback is not None:
                progress_callback(0.50, "Reading JSON text values")
            _, _, json_entries, extracted_text = extract_text_entries_from_json(resolved_path)
            if progress_callback is not None:
                progress_callback(1.0, "JSON text extraction complete")
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
        retrieval_backend: str,
        show_fewshot_examples: bool,
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
        fewshot_preview_update, fewshot_editor_update = build_fewshot_preview_update(
            source_text=source_value,
            direction_key=direction_key,
            method_key=method_key,
            fewshot_count=fewshot_count,
            retrieval_backend=retrieval_backend,
            show_examples=show_fewshot_examples,
            busy=busy,
        )
        retrieval_update = gr.update(
            value=retrieval_backend,
            visible=is_fewshot,
            interactive=not busy,
        )
        show_fewshot_examples_update = gr.update(
            value=show_fewshot_examples,
            visible=is_fewshot,
            interactive=True,
        )
        streaming_enabled_update = gr.update(interactive=not busy)

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
                retrieval_backend,
                show_fewshot_examples,
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
            retrieval_update,
            fewshot_preview_update,
            fewshot_editor_update,
            show_fewshot_examples_update,
            streaming_enabled_update,
        )

    def refresh_ui(
        current_job_id: str,
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        retrieval_backend: str,
        show_fewshot_examples: bool,
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
                retrieval_backend=retrieval_backend,
                show_fewshot_examples=show_fewshot_examples,
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
                retrieval_backend=retrieval_backend,
                show_fewshot_examples=show_fewshot_examples,
                status_text="",
            )

        status_text = summarize_status(snapshot)
        next_preview_state = build_preview_state(resolved_job_id or "", snapshot)
        preview_changed = (
            str(current_preview_state.get("job_id", "") or "") != str(next_preview_state.get("job_id", "") or "")
            or int(current_preview_state.get("completed_segments", -1) or -1)
            != int(next_preview_state.get("completed_segments", -1) or -1)
            or int(current_preview_state.get("translation_revision", -1) or -1)
            != int(next_preview_state.get("translation_revision", -1) or -1)
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
            retrieval_backend=str(snapshot.get("retrieval_backend", retrieval_backend) or retrieval_backend),
            show_fewshot_examples=show_fewshot_examples,
            status_text=status_text,
            translation_value=translation_value,
            translated_file_value=translated_file_value,
        )

    def refresh_timer_ui(
        current_job_id: str,
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        retrieval_backend: str,
        show_fewshot_examples: bool,
        preview_state: dict[str, object] | None,
        source_preview_state: str,
    ) -> tuple[object, ...]:
        result = list(
            refresh_ui(
                current_job_id,
                manual_text,
                direction_key,
                method_key,
                fewshot_count,
                segment_window_size,
                retrieval_backend,
                show_fewshot_examples,
                preview_state,
                source_preview_state,
            )
        )
        resolved_job_id, snapshot = resolve_tracked_job(current_job_id)
        result[22] = gr.skip()
        result[23] = gr.skip()
        if snapshot is None and not resolved_job_id:
            result[4] = gr.skip()
            result[5] = gr.skip()
            result[6] = gr.skip()
            result[7] = gr.skip()
            result[10] = gr.skip()
            result[24] = gr.skip()
            result[25] = gr.skip()
        return tuple(result)

    def build_document_preview_update(
        *,
        source_preview: str,
        status_text: str,
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        retrieval_backend: str,
        show_fewshot_examples: bool,
        preview_state: dict[str, object] | None,
        processing: bool,
    ) -> tuple[object, ...]:
        result = list(
            refresh_ui(
                CLEARED_JOB_STATE,
                manual_text,
                direction_key,
                method_key,
                fewshot_count,
                segment_window_size,
                retrieval_backend,
                show_fewshot_examples,
                preview_state,
                source_preview,
            )
        )
        result[4] = status_text
        result[5] = source_preview
        result[10] = source_preview
        if processing:
            result[15] = gr.update(interactive=False)
            result[16] = gr.update(interactive=False)
            result[17] = gr.update(interactive=False)
            result[18] = gr.update(interactive=False)
            result[20] = gr.update(interactive=False)
            result[21] = gr.update(interactive=False)
        return tuple(result)

    def preview_manual_input(
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        retrieval_backend: str,
        show_fewshot_examples: bool,
        preview_state: dict[str, object] | None,
    ) -> tuple[object, ...]:
        return refresh_ui(
            CLEARED_JOB_STATE,
            manual_text,
            direction_key,
            method_key,
            fewshot_count,
            segment_window_size,
            retrieval_backend,
            show_fewshot_examples,
            preview_state,
            manual_text,
        )

    def preview_document_input(
        input_file_path: object,
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        retrieval_backend: str,
        show_fewshot_examples: bool,
        preview_state: dict[str, object] | None,
    ):
        resolved_path = resolve_uploaded_file_path(input_file_path)
        if not resolved_path:
            yield build_document_preview_update(
                source_preview="",
                status_text="",
                manual_text=manual_text,
                direction_key=direction_key,
                method_key=method_key,
                fewshot_count=fewshot_count,
                segment_window_size=segment_window_size,
                retrieval_backend=retrieval_backend,
                show_fewshot_examples=show_fewshot_examples,
                preview_state=preview_state,
                processing=False,
            )
            return

        file_path = Path(resolved_path)
        file_suffix = file_path.suffix.lower()
        if file_suffix not in {".pdf", ".json"}:
            yield build_document_preview_update(
                source_preview="",
                status_text="Only PDF and JSON files are supported.",
                manual_text=manual_text,
                direction_key=direction_key,
                method_key=method_key,
                fewshot_count=fewshot_count,
                segment_window_size=segment_window_size,
                retrieval_backend=retrieval_backend,
                show_fewshot_examples=show_fewshot_examples,
                preview_state=preview_state,
                processing=False,
            )
            return

        progress_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        def progress_callback(progress_value: float, description: str) -> None:
            progress_queue.put(("progress", (progress_value, description)))

        def read_preview_worker() -> None:
            try:
                progress_queue.put(("result", read_document_preview(resolved_path, progress_callback=progress_callback)))
            except Exception as exc:
                progress_queue.put(("error", str(exc)))

        initial_description = "Starting local PDF OCR" if file_suffix == ".pdf" else "Reading JSON text values"
        yield build_document_preview_update(
            source_preview=format_document_loading_source(file_path.name, 0.0, initial_description),
            status_text=f"{initial_description}: {file_path.name}",
            manual_text=manual_text,
            direction_key=direction_key,
            method_key=method_key,
            fewshot_count=fewshot_count,
            segment_window_size=segment_window_size,
            retrieval_backend=retrieval_backend,
            show_fewshot_examples=show_fewshot_examples,
            preview_state=preview_state,
            processing=True,
        )

        worker = threading.Thread(target=read_preview_worker, daemon=True)
        worker.start()
        while True:
            try:
                event_kind, payload = progress_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event_kind == "progress":
                progress_value, description = payload  # type: ignore[misc]
                progress_text = format_document_loading_source(
                    file_path.name,
                    float(progress_value),
                    str(description),
                )
                yield build_document_preview_update(
                    source_preview=progress_text,
                    status_text=f"{int(round(float(progress_value) * 100))}% - {description}",
                    manual_text=manual_text,
                    direction_key=direction_key,
                    method_key=method_key,
                    fewshot_count=fewshot_count,
                    segment_window_size=segment_window_size,
                    retrieval_backend=retrieval_backend,
                    show_fewshot_examples=show_fewshot_examples,
                    preview_state=preview_state,
                    processing=True,
                )
                continue

            if event_kind == "result":
                source_preview, status_text = payload  # type: ignore[misc]
                yield build_document_preview_update(
                    source_preview=str(source_preview),
                    status_text=str(status_text),
                    manual_text=manual_text,
                    direction_key=direction_key,
                    method_key=method_key,
                    fewshot_count=fewshot_count,
                    segment_window_size=segment_window_size,
                    retrieval_backend=retrieval_backend,
                    show_fewshot_examples=show_fewshot_examples,
                    preview_state=preview_state,
                    processing=False,
                )
                return

            if event_kind == "error":
                yield build_document_preview_update(
                    source_preview="",
                    status_text=str(payload),
                    manual_text=manual_text,
                    direction_key=direction_key,
                    method_key=method_key,
                    fewshot_count=fewshot_count,
                    segment_window_size=segment_window_size,
                    retrieval_backend=retrieval_backend,
                    show_fewshot_examples=show_fewshot_examples,
                    preview_state=preview_state,
                    processing=False,
                )
                return

    def submit_text_job(
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        retrieval_backend: str,
        streaming_enabled: bool,
        show_fewshot_examples: bool,
        fewshot_editor_rows: object,
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
                    retrieval_backend,
                    show_fewshot_examples,
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
                retrieval_backend=retrieval_backend,
                streaming_enabled=bool(streaming_enabled),
                manual_fewshot_rows=fewshot_editor_rows,
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
                    retrieval_backend,
                    show_fewshot_examples,
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
                retrieval_backend,
                show_fewshot_examples,
                preview_state,
                manual_text,
            )
        )
        result[4] = f"Job {job_id} submitted."
        result[7] = None
        result[15] = gr.update(value=None, interactive=False)
        return tuple(result)

    def submit_file_job(
        input_file_path: object,
        manual_text: str,
        direction_key: str,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
        retrieval_backend: str,
        streaming_enabled: bool,
        show_fewshot_examples: bool,
        fewshot_editor_rows: object,
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
                    retrieval_backend,
                    show_fewshot_examples,
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
            resolved_path = resolve_uploaded_file_path(input_file_path)
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
                    retrieval_backend=retrieval_backend,
                    streaming_enabled=bool(streaming_enabled),
                    manual_fewshot_rows=fewshot_editor_rows,
                )
            elif file_suffix == ".json":
                job_id = app_backend.submit_json_job(
                    resolved_path,
                    fewshot_count,
                    direction_key=direction_key,
                    method_key=method_key,
                    segment_window_size=segment_window_size,
                    retrieval_backend=retrieval_backend,
                    streaming_enabled=bool(streaming_enabled),
                    manual_fewshot_rows=fewshot_editor_rows,
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
                    retrieval_backend,
                    show_fewshot_examples,
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
                retrieval_backend,
                show_fewshot_examples,
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
        retrieval_backend: str,
        show_fewshot_examples: bool,
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
                    retrieval_backend,
                    show_fewshot_examples,
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
                    retrieval_backend,
                    show_fewshot_examples,
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
                retrieval_backend,
                show_fewshot_examples,
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
        retrieval_backend: str,
        show_fewshot_examples: bool,
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
            retrieval_backend=retrieval_backend,
            show_fewshot_examples=show_fewshot_examples,
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
                    "PDF uploads run local OCR first, then show extracted segments in Source and translated segments live in Translation.",
                    "PDF uploads no longer render a translated PDF. They stay as segment-based text output in the Translation pane.",
                    "JSON uploads still keep their translated JSON export file.",
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

        retrieval_backend_radio = gr.Radio(
            choices=retrieval_choices,
            value=default_retrieval_backend,
            label="Few-shot retriever",
            visible=default_method_key == "fewshot_baseline",
        )
        streaming_enabled_checkbox = gr.Checkbox(
            value=DEFAULT_STREAMING_TRANSLATION,
            label="Stream output while translating",
        )
        show_fewshot_examples_checkbox = gr.Checkbox(
            value=True,
            label="Show retrieved few-shot examples",
            visible=default_method_key == "fewshot_baseline",
        )

        fewshot_examples_box = gr.HTML(
            value=render_fewshot_preview_html([], total_segments=0, fewshot_count=3),
            visible=default_method_key == "fewshot_baseline",
        )
        fewshot_examples_editor = gr.Dataframe(
            headers=FEWSHOT_EDITOR_HEADERS,
            value=[],
            row_count=(1, "dynamic"),
            col_count=(5, "fixed"),
            label="Edit retrieved few-shot examples for this job",
            interactive=True,
            visible=default_method_key == "fewshot_baseline",
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

        refresh_timer = gr.Timer(value=REFRESH_INTERVAL_SECONDS, active=True)

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
            retrieval_backend_radio,
            fewshot_examples_box,
            fewshot_examples_editor,
            show_fewshot_examples_checkbox,
            streaming_enabled_checkbox,
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
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
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
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
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
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
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
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
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
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
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
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        retrieval_backend_radio.change(
            fn=refresh_ui,
            inputs=[
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        show_fewshot_examples_checkbox.change(
            fn=refresh_ui,
            inputs=[
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
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
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
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
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
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
                retrieval_backend_radio,
                streaming_enabled_checkbox,
                show_fewshot_examples_checkbox,
                fewshot_examples_editor,
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
                retrieval_backend_radio,
                streaming_enabled_checkbox,
                show_fewshot_examples_checkbox,
                fewshot_examples_editor,
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
                retrieval_backend_radio,
                streaming_enabled_checkbox,
                show_fewshot_examples_checkbox,
                fewshot_examples_editor,
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
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
                preview_state,
                source_preview_state,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        clear_button.click(
            fn=clear_form,
            inputs=[
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
            ],
            outputs=standard_outputs,
            concurrency_limit=1,
        )
        refresh_timer.tick(
            fn=refresh_timer_ui,
            inputs=[
                current_job_state,
                manual_text_box,
                direction_radio,
                method_radio,
                fewshot_slider,
                segment_window_slider,
                retrieval_backend_radio,
                show_fewshot_examples_checkbox,
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
