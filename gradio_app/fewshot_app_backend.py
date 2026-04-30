from __future__ import annotations

import atexit
import copy
import functools
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRANSLATION_DIR = REPO_ROOT / "translation"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRANSLATION_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSLATION_DIR))


DEFAULT_KO_EN_MODEL = os.environ.get(
    "FEWSHOT_BASELINE_MODEL_KO_EN",
    "SKIML/mfds-vaivgem-ko-en-fewshot-lora",
)
DEFAULT_EN_KO_MODEL = os.environ.get(
    "FEWSHOT_BASELINE_MODEL_EN_KO",
    "SKIML/mfds-vaivgem-en-ko-fewshot-lora",
)
DEFAULT_DB_ROOT = os.environ.get(
    "MFDS_FAISS_DB_ROOT",
    str(REPO_ROOT / "resources" / "faiss" / "dev_with_doc_id"),
)
DEFAULT_HF_HOME = os.environ.get("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
DEFAULT_HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", f"{DEFAULT_HF_HOME}/hub")
DEFAULT_VLLM_ATTENTION_BACKEND = os.environ.get("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
DEFAULT_VLLM_ENFORCE_EAGER = os.environ.get("VLLM_ENFORCE_EAGER", "1")
DEFAULT_VLLM_USE_FLASHINFER_SAMPLER = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "0")
DEFAULT_MFDS_DISABLE_VLLM_FLASHINFER = os.environ.get("MFDS_DISABLE_VLLM_FLASHINFER", "0")
DEFAULT_GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.6"))
DEFAULT_BATCH_SIZE = int(os.environ.get("FEWSHOT_APP_BATCH_SIZE", "64"))
DEFAULT_PROGRESS_CHUNK_SIZE = int(os.environ.get("FEWSHOT_APP_PROGRESS_CHUNK_SIZE", "32"))
DEFAULT_PDF_OUTPUT_ROOT = Path(
    os.environ.get("MFDS_PDF_OUTPUT_ROOT", str(REPO_ROOT / ".cache" / "translated_pdfs"))
)
DEFAULT_JSON_OUTPUT_ROOT = Path(
    os.environ.get("MFDS_JSON_OUTPUT_ROOT", str(REPO_ROOT / ".cache" / "translated_jsons"))
)
DEFAULT_TEXT_OUTPUT_ROOT = Path(
    os.environ.get("MFDS_TEXT_OUTPUT_ROOT", str(REPO_ROOT / ".cache" / "translated_texts"))
)
DEFAULT_JOB_STATE_PATH = Path(
    os.environ.get(
        "MFDS_GRADIO_JOB_STATE_PATH",
        str(REPO_ROOT / ".cache" / "gradio_state" / "jobs.json"),
    )
)
DEFAULT_AUTO_BATCH_TOKEN_BUDGET = int(
    os.environ.get("FEWSHOT_APP_AUTO_BATCH_TOKEN_BUDGET", "12288")
)

_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_SENTENCE_RE = re.compile(
    r"""
    .+?
    (?:
        [.!?]+\s*
      | [\u3002\uff01\uff1f]\s*
      | $
    )
    """,
    re.VERBOSE | re.DOTALL,
)

METHOD_LABELS = {
    "fewshot_baseline": "Retrieval Few-shot",
    "segment_mt": "Segment",
}
ACTIVE_JOB_STATES = {"queued", "running", "cancelling"}
ACTIVE_JOB_STATE_PRIORITY = {
    "running": 0,
    "cancelling": 1,
    "queued": 2,
}


@dataclass(frozen=True)
class DirectionConfig:
    key: str
    source_lang: str
    target_lang: str
    model_path: str
    tokenizer_model: str | None
    visible_devices: str
    db_root: str = DEFAULT_DB_ROOT
    gpu_mem_util: float = DEFAULT_GPU_MEM_UTIL
    batch_size: int = DEFAULT_BATCH_SIZE
    train_hf2: bool = True

    @property
    def db_name(self) -> str:
        if self.key == "ko_en":
            return f"{self.db_root}_ko_to_en"
        return f"{self.db_root}_en_to_ko"

    @property
    def label(self) -> str:
        return f"{self.source_lang} -> {self.target_lang}"


@dataclass(frozen=True)
class TranslationResult:
    direction_key: str
    direction_label: str
    segment_count: int
    translation: str
    status: str


@dataclass(frozen=True)
class PdfTextBlock:
    page_index: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float
    color: tuple[float, float, float]
    align: int
    is_bold: bool


@dataclass(frozen=True)
class PdfTranslationResult:
    direction_key: str
    direction_label: str
    page_count: int
    block_count: int
    extracted_text: str
    translation: str
    translated_pdf_path: str
    status: str


@dataclass(frozen=True)
class JsonTextEntry:
    path: tuple[object, ...]
    original_text: str


@dataclass(frozen=True)
class PreparedTextUnit:
    original_text: str
    layout: tuple[tuple[str, int], ...]
    segment_count: int


@dataclass
class TranslationJob:
    job_id: str
    direction_key: str
    direction_label: str
    method_key: str
    method_label: str
    input_kind: str
    fewshot_count: int
    segment_window_size: int
    created_at: float
    units: list[PreparedTextUnit]
    segments: list[str]
    extracted_text: str = ""
    pdf_source_path: str | None = None
    pdf_blocks: list[PdfTextBlock] = field(default_factory=list)
    json_source_path: str | None = None
    json_payload: Any = None
    json_entries: list[JsonTextEntry] = field(default_factory=list)
    page_count: int = 0
    block_count: int = 0
    state: str = "queued"
    stage: str = "Queued"
    cancel_requested: bool = False
    total_segments: int = 0
    completed_segments: int = 0
    progress_percent: float = 0.0
    translation: str = ""
    translated_file_path: str | None = None
    translated_pdf_path: str | None = None
    error: str = ""
    started_at: float | None = None
    finished_at: float | None = None


class _JobCancelledError(RuntimeError):
    pass


def _serialize_prepared_unit(unit: PreparedTextUnit) -> dict[str, Any]:
    return {
        "original_text": unit.original_text,
        "layout": [list(item) for item in unit.layout],
        "segment_count": unit.segment_count,
    }


def _deserialize_prepared_unit(data: dict[str, Any]) -> PreparedTextUnit:
    raw_layout = data.get("layout", []) or []
    layout = []
    for item in raw_layout:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            layout.append((str(item[0]), int(item[1])))
    return PreparedTextUnit(
        original_text=str(data.get("original_text", "") or ""),
        layout=tuple(layout),
        segment_count=int(data.get("segment_count", 0) or 0),
    )


def _serialize_pdf_block(block: PdfTextBlock) -> dict[str, Any]:
    return {
        "page_index": block.page_index,
        "bbox": list(block.bbox),
        "text": block.text,
        "font_size": block.font_size,
        "color": list(block.color),
        "align": block.align,
        "is_bold": block.is_bold,
    }


def _deserialize_pdf_block(data: dict[str, Any]) -> PdfTextBlock:
    bbox = tuple(float(value) for value in (data.get("bbox") or [0, 0, 0, 0])[:4])
    color = tuple(float(value) for value in (data.get("color") or [0, 0, 0])[:3])
    return PdfTextBlock(
        page_index=int(data.get("page_index", 0) or 0),
        bbox=bbox,  # type: ignore[arg-type]
        text=str(data.get("text", "") or ""),
        font_size=float(data.get("font_size", 11.0) or 11.0),
        color=color,  # type: ignore[arg-type]
        align=int(data.get("align", 0) or 0),
        is_bold=bool(data.get("is_bold", False)),
    )


def _serialize_json_entry(entry: JsonTextEntry) -> dict[str, Any]:
    return {
        "path": list(entry.path),
        "original_text": entry.original_text,
    }


def _deserialize_json_entry(data: dict[str, Any]) -> JsonTextEntry:
    raw_path = data.get("path", []) or []
    return JsonTextEntry(
        path=tuple(raw_path),
        original_text=str(data.get("original_text", "") or ""),
    )


def _serialize_job(job: TranslationJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "direction_key": job.direction_key,
        "direction_label": job.direction_label,
        "method_key": job.method_key,
        "method_label": job.method_label,
        "input_kind": job.input_kind,
        "fewshot_count": job.fewshot_count,
        "segment_window_size": job.segment_window_size,
        "created_at": job.created_at,
        "units": [_serialize_prepared_unit(unit) for unit in job.units],
        "segments": list(job.segments),
        "extracted_text": job.extracted_text,
        "pdf_source_path": job.pdf_source_path,
        "pdf_blocks": [_serialize_pdf_block(block) for block in job.pdf_blocks],
        "json_source_path": job.json_source_path,
        "json_payload": job.json_payload,
        "json_entries": [_serialize_json_entry(entry) for entry in job.json_entries],
        "page_count": job.page_count,
        "block_count": job.block_count,
        "state": job.state,
        "stage": job.stage,
        "cancel_requested": job.cancel_requested,
        "total_segments": job.total_segments,
        "completed_segments": job.completed_segments,
        "progress_percent": job.progress_percent,
        "translation": job.translation,
        "translated_file_path": job.translated_file_path,
        "translated_pdf_path": job.translated_pdf_path,
        "error": job.error,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def _deserialize_job(data: dict[str, Any]) -> TranslationJob:
    return TranslationJob(
        job_id=str(data.get("job_id", "") or ""),
        direction_key=str(data.get("direction_key", "") or ""),
        direction_label=str(data.get("direction_label", "") or ""),
        method_key=str(data.get("method_key", "fewshot_baseline") or "fewshot_baseline"),
        method_label=str(data.get("method_label", "") or METHOD_LABELS.get(str(data.get("method_key", "")), "")),
        input_kind=str(data.get("input_kind", "") or ""),
        fewshot_count=int(data.get("fewshot_count", 0) or 0),
        segment_window_size=int(data.get("segment_window_size", 1) or 1),
        created_at=float(data.get("created_at", 0.0) or 0.0),
        units=[_deserialize_prepared_unit(item) for item in (data.get("units", []) or []) if isinstance(item, dict)],
        segments=[str(item) for item in (data.get("segments", []) or [])],
        extracted_text=str(data.get("extracted_text", "") or ""),
        pdf_source_path=data.get("pdf_source_path"),
        pdf_blocks=[_deserialize_pdf_block(item) for item in (data.get("pdf_blocks", []) or []) if isinstance(item, dict)],
        json_source_path=data.get("json_source_path"),
        json_payload=data.get("json_payload"),
        json_entries=[_deserialize_json_entry(item) for item in (data.get("json_entries", []) or []) if isinstance(item, dict)],
        page_count=int(data.get("page_count", 0) or 0),
        block_count=int(data.get("block_count", 0) or 0),
        state=str(data.get("state", "queued") or "queued"),
        stage=str(data.get("stage", "") or ""),
        cancel_requested=bool(data.get("cancel_requested", False)),
        total_segments=int(data.get("total_segments", 0) or 0),
        completed_segments=int(data.get("completed_segments", 0) or 0),
        progress_percent=float(data.get("progress_percent", 0.0) or 0.0),
        translation=str(data.get("translation", "") or ""),
        translated_file_path=data.get("translated_file_path"),
        translated_pdf_path=data.get("translated_pdf_path"),
        error=str(data.get("error", "") or ""),
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
    )


def _default_gpu_mapping() -> tuple[str, str]:
    raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw_visible_devices:
        visible_tokens = [token.strip() for token in raw_visible_devices.split(",") if token.strip()]
        if len(visible_tokens) >= 2:
            return visible_tokens[0], visible_tokens[1]
        if len(visible_tokens) == 1:
            return visible_tokens[0], visible_tokens[0]

    try:
        import torch

        gpu_count = int(torch.cuda.device_count())
    except Exception:
        gpu_count = 0

    if gpu_count >= 2:
        return ("0", "1")
    if gpu_count == 1:
        return ("0", "0")
    return ("", "")


def build_default_direction_configs(
    *,
    ko_en_gpu: str | None = None,
    en_ko_gpu: str | None = None,
    ko_en_model: str = DEFAULT_KO_EN_MODEL,
    en_ko_model: str = DEFAULT_EN_KO_MODEL,
    tokenizer_model: str | None = None,
    db_root: str = DEFAULT_DB_ROOT,
    gpu_mem_util: float = DEFAULT_GPU_MEM_UTIL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, DirectionConfig]:
    default_ko_gpu, default_en_gpu = _default_gpu_mapping()
    shared_tokenizer = tokenizer_model or None
    return {
        "ko_en": DirectionConfig(
            key="ko_en",
            source_lang="Korean",
            target_lang="English",
            model_path=ko_en_model,
            tokenizer_model=shared_tokenizer or ko_en_model,
            visible_devices=ko_en_gpu if ko_en_gpu is not None else default_ko_gpu,
            db_root=db_root,
            gpu_mem_util=gpu_mem_util,
            batch_size=batch_size,
        ),
        "en_ko": DirectionConfig(
            key="en_ko",
            source_lang="English",
            target_lang="Korean",
            model_path=en_ko_model,
            tokenizer_model=shared_tokenizer or en_ko_model,
            visible_devices=en_ko_gpu if en_ko_gpu is not None else default_en_gpu,
            db_root=db_root,
            gpu_mem_util=gpu_mem_util,
            batch_size=batch_size,
        ),
    }


def detect_direction_key(text: str) -> str:
    hangul_count = len(_HANGUL_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    if hangul_count == 0 and latin_count == 0:
        raise ValueError("Input must contain Korean or English text.")
    if hangul_count >= latin_count:
        return "ko_en"
    return "en_ko"


def extract_text_from_pdf(pdf_path: str) -> str:
    pdf_file = Path(pdf_path).expanduser().resolve()
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    try:
        completed = subprocess.run(
            [
                "pdftotext",
                "-q",
                "-enc",
                "UTF-8",
                "-nopgbrk",
                str(pdf_file),
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("pdftotext is not installed on this system.") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Failed to extract text from PDF: {message or exc}") from exc

    extracted_text = completed.stdout.replace("\x0c", "\n").strip()
    if not extracted_text:
        raise ValueError(
            "No extractable text was found in the PDF. "
            "Scanned PDFs require OCR, which this app does not perform."
        )
    return extracted_text


def _median(values: Sequence[float], default: float) -> float:
    if not values:
        return default
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _pdf_color_to_rgb(color_value: object) -> tuple[float, float, float]:
    if isinstance(color_value, int):
        red = ((color_value >> 16) & 0xFF) / 255.0
        green = ((color_value >> 8) & 0xFF) / 255.0
        blue = (color_value & 0xFF) / 255.0
        return (red, green, blue)
    if isinstance(color_value, (tuple, list)) and len(color_value) >= 3:
        return tuple(float(channel) for channel in color_value[:3])  # type: ignore[return-value]
    return (0.0, 0.0, 0.0)


def _guess_pdf_alignment(left_gaps: Sequence[float], right_gaps: Sequence[float], width: float) -> int:
    if not left_gaps or not right_gaps:
        return 0
    avg_left = sum(left_gaps) / len(left_gaps)
    avg_right = sum(right_gaps) / len(right_gaps)
    tolerance = max(4.0, width * 0.03)
    if abs(avg_left - avg_right) <= tolerance and avg_left > tolerance * 0.5 and avg_right > tolerance * 0.5:
        return 1
    if avg_left > avg_right + tolerance:
        return 2
    return 0


def _scale_progress(
    callback: Callable[[float, str], None] | None,
    start: float,
    end: float,
) -> Callable[[float, str], None] | None:
    if callback is None:
        return None
    span = end - start

    def _wrapped(value: float, desc: str) -> None:
        callback(start + (value * span), desc)

    return _wrapped


def _normalize_pdf_line_text(raw_text: str) -> str:
    return raw_text.replace("\u00a0", " ").replace("\r", "").strip()


def _build_pdf_text_block(page_index: int, block: dict[str, object]) -> PdfTextBlock | None:
    bbox = block.get("bbox")
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return None

    x0, y0, x1, y1 = (float(value) for value in bbox)
    width = x1 - x0
    height = y1 - y0
    if width <= 1 or height <= 1:
        return None

    lines = block.get("lines")
    if not isinstance(lines, list):
        return None

    text_lines: List[str] = []
    font_sizes: List[float] = []
    colors: List[object] = []
    left_gaps: List[float] = []
    right_gaps: List[float] = []
    bold_votes = 0
    span_votes = 0

    for line in lines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue

        line_parts: List[str] = []
        active_spans: List[dict[str, object]] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            span_text = str(span.get("text") or "")
            if not span_text:
                continue
            line_parts.append(span_text)
            if span_text.strip():
                active_spans.append(span)
                size = span.get("size")
                if isinstance(size, (int, float)) and float(size) > 0:
                    font_sizes.append(float(size))
                if "color" in span:
                    color_value = span.get("color")
                    if isinstance(color_value, list):
                        colors.append(tuple(color_value))
                    else:
                        colors.append(color_value)
                font_name = str(span.get("font") or "").lower()
                if any(keyword in font_name for keyword in ("bold", "black", "heavy", "semibold", "semi bold")):
                    bold_votes += 1
                span_votes += 1

        line_text = _normalize_pdf_line_text("".join(line_parts))
        if line_text:
            text_lines.append(line_text)

        if active_spans:
            first_bbox = active_spans[0].get("bbox")
            last_bbox = active_spans[-1].get("bbox")
            if isinstance(first_bbox, (tuple, list)) and len(first_bbox) == 4:
                left_gaps.append(max(0.0, float(first_bbox[0]) - x0))
            if isinstance(last_bbox, (tuple, list)) and len(last_bbox) == 4:
                right_gaps.append(max(0.0, x1 - float(last_bbox[2])))

    block_text = "\n".join(text_lines).strip()
    if not block_text:
        return None

    dominant_color = Counter(colors).most_common(1)[0][0] if colors else 0
    return PdfTextBlock(
        page_index=page_index,
        bbox=(x0, y0, x1, y1),
        text=block_text,
        font_size=max(4.0, _median(font_sizes, 11.0)),
        color=_pdf_color_to_rgb(dominant_color),
        align=_guess_pdf_alignment(left_gaps, right_gaps, width),
        is_bold=span_votes > 0 and bold_votes >= max(1, span_votes // 2),
    )


def extract_text_blocks_from_pdf(pdf_path: str) -> tuple[Path, int, List[PdfTextBlock], str]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyMuPDF is not installed in the current environment.") from exc

    pdf_file = Path(pdf_path).expanduser().resolve()
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    try:
        document = fitz.open(str(pdf_file))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {exc}") from exc

    blocks: List[PdfTextBlock] = []
    page_texts: List[str] = []
    try:
        page_count = document.page_count
        for page_index, page in enumerate(document):
            page_dict = page.get_text("dict", sort=True)
            page_blocks: List[str] = []
            for block in page_dict.get("blocks", []):
                if not isinstance(block, dict) or int(block.get("type", 0)) != 0:
                    continue
                pdf_block = _build_pdf_text_block(page_index, block)
                if pdf_block is None:
                    continue
                blocks.append(pdf_block)
                page_blocks.append(pdf_block.text)
            page_text = "\n\n".join(item for item in page_blocks if item.strip()).strip()
            if page_text:
                page_texts.append(page_text)
    finally:
        document.close()

    extracted_text = "\n\n".join(page_texts).strip()
    if not extracted_text or not blocks:
        raise ValueError(
            "No extractable text blocks were found in the PDF. "
            "Scanned PDFs require OCR, which this app does not perform."
        )
    return pdf_file, page_count, blocks, extracted_text


def rebuild_text_from_pdf_blocks(blocks: Sequence[PdfTextBlock], texts: Sequence[str]) -> str:
    page_to_texts: dict[int, List[str]] = {}
    for block, text in zip(blocks, texts):
        cleaned = str(text or "").strip()
        if not cleaned:
            continue
        page_to_texts.setdefault(block.page_index, []).append(cleaned)

    pages: List[str] = []
    for page_index in sorted(page_to_texts):
        page_text = "\n\n".join(page_to_texts[page_index]).strip()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages).strip()


def _collect_json_text_entries(
    node: Any,
    *,
    path: tuple[object, ...] = (),
    entries: list[JsonTextEntry],
) -> None:
    if isinstance(node, str):
        entries.append(JsonTextEntry(path=path, original_text=node))
        return
    if isinstance(node, list):
        for index, item in enumerate(node):
            _collect_json_text_entries(item, path=path + (index,), entries=entries)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _collect_json_text_entries(value, path=path + (str(key),), entries=entries)


def extract_text_entries_from_json(json_path: str) -> tuple[Path, Any, list[JsonTextEntry], str]:
    json_file = Path(json_path).expanduser().resolve()
    if not json_file.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    try:
        with json_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {json_file}") from exc

    entries: list[JsonTextEntry] = []
    _collect_json_text_entries(payload, entries=entries)
    non_empty_texts = [entry.original_text.strip() for entry in entries if entry.original_text.strip()]
    if not non_empty_texts:
        raise ValueError("No string values were found in the JSON file.")
    extracted_text = "\n\n".join(non_empty_texts).strip()
    return json_file, payload, entries, extracted_text


def _set_json_path_value(node: Any, path: Sequence[object], value: str) -> None:
    if not path:
        raise ValueError("JSON string path cannot be empty.")
    target = node
    for token in path[:-1]:
        target = target[token]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]


def build_translated_json_payload(
    payload: Any,
    entries: Sequence[JsonTextEntry],
    translated_texts: Sequence[str],
) -> Any:
    if len(translated_texts) > len(entries):
        raise ValueError("Too many translated JSON strings were provided.")

    translated_payload = copy.deepcopy(payload)
    for entry, translated_text in zip(entries, translated_texts):
        _set_json_path_value(translated_payload, entry.path, str(translated_text))
    return translated_payload


def rebuild_text_from_json_entries(texts: Sequence[str]) -> str:
    return "\n\n".join(item.strip() for item in texts if str(item).strip()).strip()


def _resolve_fc_match_font(query: str) -> str | None:
    if not shutil.which("fc-match"):
        return None
    try:
        completed = subprocess.run(
            ["fc-match", "-f", "%{file}\n", query],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    candidate = completed.stdout.strip().splitlines()
    if not candidate:
        return None
    font_path = Path(candidate[0].strip()).expanduser()
    if font_path.is_file():
        return str(font_path)
    return None


@functools.lru_cache(maxsize=1)
def _resolve_pdf_font_paths() -> tuple[str, str]:
    regular_candidates = [
        os.environ.get("MFDS_PDF_FONT_REGULAR", "").strip(),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    bold_candidates = [
        os.environ.get("MFDS_PDF_FONT_BOLD", "").strip(),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for query in ("Noto Sans CJK KR:style=Regular", "Noto Sans CJK KR", "DejaVu Sans"):
        resolved = _resolve_fc_match_font(query)
        if resolved:
            regular_candidates.insert(0, resolved)
            break
    for query in ("Noto Sans CJK KR:style=Bold", "Noto Sans CJK KR:style=Black", "DejaVu Sans:style=Bold"):
        resolved = _resolve_fc_match_font(query)
        if resolved:
            bold_candidates.insert(0, resolved)
            break

    regular_font = next((candidate for candidate in regular_candidates if candidate and Path(candidate).is_file()), None)
    bold_font = next((candidate for candidate in bold_candidates if candidate and Path(candidate).is_file()), None)
    if regular_font is None:
        raise RuntimeError("Could not find a usable font for translated PDF rendering.")
    return regular_font, bold_font or regular_font


def _get_pdf_output_root() -> Path:
    DEFAULT_PDF_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return DEFAULT_PDF_OUTPUT_ROOT


def _get_json_output_root() -> Path:
    DEFAULT_JSON_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return DEFAULT_JSON_OUTPUT_ROOT


def _get_text_output_root() -> Path:
    DEFAULT_TEXT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return DEFAULT_TEXT_OUTPUT_ROOT


def _cleanup_output_file(path_value: str | None) -> None:
    if not path_value:
        return
    try:
        path = Path(path_value)
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _build_pdf_output_path(
    *,
    original_pdf_path: Path,
    direction_key: str,
    job_id: str | None = None,
    completed_segments: int | None = None,
    total_segments: int | None = None,
    final: bool = False,
) -> Path:
    output_root = _get_pdf_output_root()
    if final:
        suffix = f"{direction_key}.translated.{job_id or uuid.uuid4().hex[:8]}.pdf"
    else:
        suffix = (
            f"{direction_key}.partial.{job_id or uuid.uuid4().hex[:8]}."
            f"{int(completed_segments or 0):06d}of{int(total_segments or 0):06d}.pdf"
        )
    return output_root / f"{original_pdf_path.stem}.{suffix}"


def _build_json_output_path(
    *,
    original_json_path: Path,
    direction_key: str,
    job_id: str | None = None,
    completed_segments: int | None = None,
    total_segments: int | None = None,
    final: bool = False,
) -> Path:
    output_root = _get_json_output_root()
    if final:
        suffix = f"{direction_key}.translated.{job_id or uuid.uuid4().hex[:8]}.json"
    else:
        suffix = (
            f"{direction_key}.partial.{job_id or uuid.uuid4().hex[:8]}."
            f"{int(completed_segments or 0):06d}of{int(total_segments or 0):06d}.json"
        )
    return output_root / f"{original_json_path.stem}.{suffix}"


def build_pdf_render_blocks(
    blocks: Sequence[PdfTextBlock],
    translated_blocks: Sequence[str],
) -> list[str | None]:
    if len(blocks) != len(translated_blocks):
        raise ValueError("Translated PDF rendering requires one translated text per block.")

    render_blocks: list[str | None] = []
    for block, translated_text in zip(blocks, translated_blocks):
        normalized_text = str(translated_text or "").strip()
        if not normalized_text:
            normalized_text = block.text
        render_blocks.append(None if normalized_text == block.text else normalized_text)
    return render_blocks


def _fit_and_insert_textbox(
    page: object,
    rect: object,
    text: str,
    *,
    fontfile: str,
    fontname: str,
    fontsize: float,
    color: tuple[float, float, float],
    align: int,
) -> float:
    normalized_text = str(text or "").replace("\x0c", "\n").replace("\u00a0", " ").strip()
    if not normalized_text:
        return 0.0

    rect_height = max(1.0, float(getattr(rect, "height", 0.0)))
    start_size = max(4.0, min(float(fontsize), rect_height * 0.9))
    min_size = max(4.0, min(start_size, 8.0) * 0.75)
    current_size = start_size

    while current_size >= min_size - 1e-6:
        remaining = page.insert_textbox(
            rect,
            normalized_text,
            fontname=fontname,
            fontfile=fontfile,
            fontsize=current_size,
            color=color,
            align=align,
            overlay=True,
        )
        if remaining >= 0:
            return current_size
        current_size -= 0.5

    if " " in normalized_text:
        parts = normalized_text.split()
        while len(parts) > 1:
            candidate = " ".join(parts[:-1]).strip() + " ..."
            remaining = page.insert_textbox(
                rect,
                candidate,
                fontname=fontname,
                fontfile=fontfile,
                fontsize=min_size,
                color=color,
                align=align,
                overlay=True,
            )
            if remaining >= 0:
                return min_size
            parts = parts[:-1]
    else:
        chars = list(normalized_text)
        while len(chars) > 1:
            candidate = "".join(chars[:-1]).strip() + "..."
            remaining = page.insert_textbox(
                rect,
                candidate,
                fontname=fontname,
                fontfile=fontfile,
                fontsize=min_size,
                color=color,
                align=align,
                overlay=True,
            )
            if remaining >= 0:
                return min_size
            chars = chars[:-1]

    return 0.0


def render_translated_pdf(
    *,
    original_pdf_path: Path,
    blocks: Sequence[PdfTextBlock],
    translated_blocks: Sequence[str | None],
    direction_key: str,
    output_path: Path | None = None,
) -> str:
    if len(blocks) != len(translated_blocks):
        raise ValueError("Translated PDF rendering requires one translated text per block.")

    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyMuPDF is not installed in the current environment.") from exc

    regular_font, bold_font = _resolve_pdf_font_paths()
    resolved_output_path = output_path or _build_pdf_output_path(
        original_pdf_path=original_pdf_path,
        direction_key=direction_key,
        final=True,
    )

    document = fitz.open(str(original_pdf_path))
    try:
        blocks_by_page: dict[int, List[tuple[PdfTextBlock, str]]] = {}
        for block, translated_text in zip(blocks, translated_blocks):
            if translated_text is None:
                continue
            text_to_render = str(translated_text or "").strip() or block.text
            if text_to_render == block.text:
                continue
            blocks_by_page.setdefault(block.page_index, []).append((block, text_to_render))

        for page_index, page_blocks in blocks_by_page.items():
            page = document[page_index]
            for block, _ in page_blocks:
                redact_rect = fitz.Rect(block.bbox)
                redact_rect.x0 -= 0.5
                redact_rect.y0 -= 0.5
                redact_rect.x1 += 0.5
                redact_rect.y1 += 0.5
                page.add_redact_annot(redact_rect, fill=None)

            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                text=fitz.PDF_REDACT_TEXT_REMOVE,
            )

            for block, translated_text in page_blocks:
                insert_rect = fitz.Rect(block.bbox)
                text_to_render = str(translated_text or "").strip() or block.text
                font_path = bold_font if block.is_bold else regular_font
                font_alias = "mfds_pdf_bold" if block.is_bold else "mfds_pdf_regular"
                _fit_and_insert_textbox(
                    page,
                    insert_rect,
                    text_to_render,
                    fontfile=font_path,
                    fontname=font_alias,
                    fontsize=block.font_size,
                    color=block.color,
                    align=block.align,
                )

        document.save(str(resolved_output_path), garbage=4, deflate=True)
    finally:
        document.close()

    return str(resolved_output_path)


def render_translated_json(
    *,
    original_json_path: Path,
    payload: Any,
    entries: Sequence[JsonTextEntry],
    translated_texts: Sequence[str],
    direction_key: str,
    output_path: Path | None = None,
) -> str:
    if len(entries) != len(translated_texts):
        raise ValueError("Translated JSON rendering requires one translated text per string entry.")

    resolved_output_path = output_path or _build_json_output_path(
        original_json_path=original_json_path,
        direction_key=direction_key,
        final=True,
    )
    translated_payload = build_translated_json_payload(payload, entries, translated_texts)
    with resolved_output_path.open("w", encoding="utf-8") as handle:
        json.dump(translated_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return str(resolved_output_path)


def render_translated_text(
    *,
    translation: str,
    direction_key: str,
    job_id: str,
) -> str:
    output_root = _get_text_output_root()
    output_path = output_root / f"mfds_translation.{direction_key}.{job_id}.txt"
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(str(translation or "").rstrip())
        handle.write("\n")
    return str(output_path)


def _split_line_into_sentences(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped:
        return []
    matches = [match.group(0).strip() for match in _SENTENCE_RE.finditer(stripped)]
    sentences = [item for item in matches if item]
    return sentences or [stripped]


def segment_input_text(text: str) -> tuple[List[str], List[tuple[str, int]]]:
    if not text or not text.strip():
        return [], []

    segments: List[str] = []
    layout: List[tuple[str, int]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            layout.append(("blank", 0))
            continue
        line_segments = _split_line_into_sentences(raw_line)
        if not line_segments:
            layout.append(("blank", 0))
            continue
        segments.extend(line_segments)
        layout.append(("line", len(line_segments)))

    if not segments and text.strip():
        segments = [text.strip()]
        layout = [("line", 1)]
    return segments, layout


def rebuild_text_from_segments(translated_segments: Sequence[str], layout: Sequence[tuple[str, int]]) -> str:
    lines: List[str] = []
    offset = 0
    for kind, count in layout:
        if kind == "blank":
            lines.append("")
            continue
        merged = " ".join(item.strip() for item in translated_segments[offset : offset + count] if item.strip())
        lines.append(merged.strip())
        offset += count
    return "\n".join(lines).strip()


def prepare_text_units(texts: Sequence[str]) -> tuple[list[PreparedTextUnit], list[str]]:
    units: list[PreparedTextUnit] = []
    all_segments: list[str] = []
    for text in texts:
        normalized_text = str(text or "").strip()
        segments, layout = segment_input_text(normalized_text)
        units.append(
            PreparedTextUnit(
                original_text=normalized_text,
                layout=tuple(layout),
                segment_count=len(segments),
            )
        )
        all_segments.extend(segments)
    return units, all_segments


def prepare_segment_mt_units(
    texts: Sequence[str],
    *,
    window_size: int,
) -> tuple[list[PreparedTextUnit], list[str]]:
    normalized_window_size = max(1, int(window_size))
    units: list[PreparedTextUnit] = []
    aggregated_segments: list[str] = []

    for text in texts:
        normalized_text = str(text or "").strip()
        segments, layout = segment_input_text(normalized_text)
        if not segments:
            units.append(
                PreparedTextUnit(
                    original_text=normalized_text,
                    layout=tuple(layout),
                    segment_count=0,
                )
            )
            continue

        offset = 0
        aggregated_layout: list[tuple[str, int]] = []
        unit_segments: list[str] = []
        for kind, count in layout:
            if kind == "blank":
                aggregated_layout.append(("blank", 0))
                continue

            line_segments = segments[offset : offset + count]
            offset += count
            line_chunk_count = 0
            for start in range(0, len(line_segments), normalized_window_size):
                chunk = [item.strip() for item in line_segments[start : start + normalized_window_size] if item.strip()]
                if not chunk:
                    continue
                unit_segments.append("\n".join(chunk).strip())
                line_chunk_count += 1
            aggregated_layout.append(("line", line_chunk_count))

        units.append(
            PreparedTextUnit(
                original_text=normalized_text,
                layout=tuple(aggregated_layout),
                segment_count=len(unit_segments),
            )
        )
        aggregated_segments.extend(unit_segments)

    return units, aggregated_segments


def rebuild_prepared_units(units: Sequence[PreparedTextUnit], translated_segments: Sequence[str]) -> list[str]:
    rebuilt_texts: list[str] = []
    offset = 0
    for unit in units:
        if unit.segment_count == 0:
            rebuilt_texts.append(unit.original_text)
            continue
        segment_slice = translated_segments[offset : offset + unit.segment_count]
        rebuilt_texts.append(rebuild_text_from_segments(segment_slice, unit.layout))
        offset += unit.segment_count
    return rebuilt_texts


def format_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def compute_segment_progress_percent(completed_segments: int, total_segments: int) -> float:
    if total_segments <= 0:
        return 0.0
    return min(100.0, max(0.0, (completed_segments / total_segments) * 100.0))


def _configure_worker_env(config: DirectionConfig) -> None:
    if config.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.visible_devices
    os.environ["HF_HOME"] = DEFAULT_HF_HOME
    os.environ["HF_HUB_CACHE"] = DEFAULT_HF_HUB_CACHE
    os.environ["HUGGINGFACE_HUB_CACHE"] = DEFAULT_HF_HUB_CACHE
    os.environ["MFDS_FAISS_DB_ROOT"] = config.db_root
    os.environ["VLLM_ATTENTION_BACKEND"] = DEFAULT_VLLM_ATTENTION_BACKEND
    os.environ["VLLM_ENFORCE_EAGER"] = DEFAULT_VLLM_ENFORCE_EAGER
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = DEFAULT_VLLM_USE_FLASHINFER_SAMPLER
    os.environ["MFDS_DISABLE_VLLM_FLASHINFER"] = DEFAULT_MFDS_DISABLE_VLLM_FLASHINFER


def _extract_primary_text(mt_output: object) -> str:
    if isinstance(mt_output, str):
        return mt_output.strip()
    if isinstance(mt_output, dict):
        mt_paths = mt_output.get("mt_paths") or []
        if mt_paths:
            return str(mt_paths[0]).strip()
        mt_value = mt_output.get("mt")
        if isinstance(mt_value, str):
            return mt_value.strip()
    return ""


def _normalize_translated_segment_text(text: object) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    # vLLM outputs sometimes contain newline spam or excessive spacing.
    return re.sub(r"\s+", " ", normalized).strip()


class _DirectionWorkerBackend:
    def __init__(
        self,
        config: DirectionConfig,
        *,
        translator: object | None = None,
        model_config: object | None = None,
        lora_request: object | None = None,
        owns_translator: bool = True,
    ) -> None:
        from translation_models import GenerationConfig, ModelConfig, vllm_translator
        from utils.retriever import MTRetriever

        self.config = config
        self._lora_request = lora_request
        self._owns_translator = owns_translator
        if model_config is None:
            model_config = ModelConfig(
                config.model_path,
                tokenizer_id=config.tokenizer_model,
                train_hf2=config.train_hf2,
            )
        self.model_config = model_config
        if translator is None:
            self._generation_config = GenerationConfig(
                num_gpus=1,
                gpu_mem_util=config.gpu_mem_util,
                sampling_params="greedy",
            )
            self._generation_config.repetition_penalty = None
            # Both fewshot_baseline and segment_mt use the same baseline model.
            # We preload it once at worker startup so method switching does not
            # trigger a model reload on the first request.
            translator = vllm_translator(model_config, self._generation_config)
        self.translator = translator
        self._tokenizer = self.translator.tokenizer
        self._max_model_len = int(getattr(self.translator.llm.llm_engine.model_config, "max_model_len", 4096))
        self._prompt_token_budget = max(self._max_model_len, int(DEFAULT_AUTO_BATCH_TOKEN_BUDGET))
        self.retriever = MTRetriever(
            db_name=config.db_name,
            encoder="BAAI/bge-m3",
            source="ko" if config.key == "ko_en" else "en",
            target="en" if config.key == "ko_en" else "ko",
        )

    def _count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(self._tokenizer(prompt, add_special_tokens=False).input_ids)

    def _build_fewshot_messages(
        self,
        src_sent: str,
        fewshot: Sequence[dict[str, str]],
    ) -> list[dict[str, str]]:
        system_prompt = "You are a professional translator."
        user_prompt = (
            f"I will give you one or more examples of text fragments, where the first one is in "
            f"{self.config.source_lang} and the second one is the translation of the first fragment "
            f"into {self.config.target_lang}. These sentences will be displayed below."
        )
        for i, demo in enumerate(fewshot):
            user_prompt += (
                f"\n{i+1}. {self.config.source_lang} text: {demo['src']}\n"
                f"{self.config.target_lang} translation: {demo['mt']}"
            )
        user_prompt += (
            f"\nAfter the example pairs, I will provide a/an {self.config.source_lang} sentence and I would like "
            f"you to translate it into {self.config.target_lang}. Please provide only the translation result "
            f"without any additional comments, formatting, or chat content. Translate the text from "
            f"{self.config.source_lang} to {self.config.target_lang}."
        )
        user_prompt += f"\nTranslate the following sentence: {src_sent}"
        if self.model_config.requires_system_prompt:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return [{"role": "user", "content": system_prompt + user_prompt}]

    def _estimate_prompt_tokens_for_simple(self, segment: str) -> int:
        messages = self.translator.create_simple_translation_messages_list(
            [segment],
            source_lang=self.config.source_lang,
            target_lang=self.config.target_lang,
        )[0]
        return self._count_prompt_tokens(messages)

    def _estimate_prompt_tokens_for_fewshot(
        self,
        segment: str,
        fewshot: Sequence[dict[str, str]],
    ) -> int:
        return self._count_prompt_tokens(self._build_fewshot_messages(segment, fewshot))

    def _pack_dynamic_batch_spans(self, token_counts: Sequence[int]) -> list[tuple[int, int]]:
        if not token_counts:
            return []

        spans: list[tuple[int, int]] = []
        max_items = max(1, int(self.config.batch_size))
        start = 0
        item_count = 0
        token_total = 0
        for index, raw_tokens in enumerate(token_counts):
            token_count = max(1, int(raw_tokens))
            if item_count > 0 and (
                item_count >= max_items or token_total + token_count > self._prompt_token_budget
            ):
                spans.append((start, index))
                start = index
                item_count = 0
                token_total = 0
            item_count += 1
            token_total += token_count
        spans.append((start, len(token_counts)))
        return spans

    def _retrieve_fewshots_for_segments(
        self,
        segments: Sequence[str],
        fewshot_count: int,
    ) -> list[list[dict[str, str]]]:
        if fewshot_count <= 0 or not segments:
            return [[] for _ in segments]

        excluded_sources = set(segments)
        retrieval_chunk_size = max(8, min(256, int(self.config.batch_size) * 4))
        all_fewshots: list[list[dict[str, str]]] = []
        for start in range(0, len(segments), retrieval_chunk_size):
            batch_segments = list(segments[start : start + retrieval_chunk_size])
            batch_fewshots = self.retriever.search(
                batch_segments,
                fewshot_count,
                exclude_sources=excluded_sources,
                exclude_doc_ids=set(),
            )
            all_fewshots.extend(batch_fewshots)
        return all_fewshots

    def translate_segments(
        self,
        segments: Sequence[str],
        fewshot_count: int,
        method_key: str,
    ) -> List[str]:
        clean_segments = [segment.strip() for segment in segments if isinstance(segment, str) and segment.strip()]
        if not clean_segments:
            return []

        all_outputs: List[str] = []
        if method_key == "fewshot_baseline" and fewshot_count > 0:
            all_fewshots = self._retrieve_fewshots_for_segments(clean_segments, fewshot_count)
            token_counts = [
                self._estimate_prompt_tokens_for_fewshot(segment, fewshot)
                for segment, fewshot in zip(clean_segments, all_fewshots)
            ]
            batch_spans = self._pack_dynamic_batch_spans(token_counts)
            for start, end in batch_spans:
                batch_segments = clean_segments[start:end]
                fewshots = all_fewshots[start:end]
                translation_kwargs: dict[str, object] = {}
                if self._lora_request is not None:
                    translation_kwargs["lora_request"] = self._lora_request
                raw_outputs = self.translator.fewshot_singleturn_translation(
                    batch_segments,
                    fewshots,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang,
                    multiple_path=None,
                    **translation_kwargs,
                )
                all_outputs.extend(_extract_primary_text(item) for item in raw_outputs)
        else:
            token_counts = [
                self._estimate_prompt_tokens_for_simple(segment)
                for segment in clean_segments
            ]
            batch_spans = self._pack_dynamic_batch_spans(token_counts)
            for start, end in batch_spans:
                batch_segments = clean_segments[start:end]
                translation_kwargs: dict[str, object] = {}
                if self._lora_request is not None:
                    translation_kwargs["lora_request"] = self._lora_request
                raw_outputs = self.translator.simple_translation(
                    batch_segments,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang,
                    multiple_path=None,
                    **translation_kwargs,
                )
                all_outputs.extend(_extract_primary_text(item) for item in raw_outputs)
        return all_outputs

    def close(self) -> None:
        if self._owns_translator and hasattr(self.translator, "close"):
            self.translator.close()


class _SharedLoraWorkerBackend:
    def __init__(self, direction_configs: dict[str, DirectionConfig]) -> None:
        from translation_models import GenerationConfig, ModelConfig, vllm_translator
        from vllm.lora.request import LoRARequest

        if not direction_configs:
            raise ValueError("Shared LoRA worker requires at least one direction.")

        ordered_items = list(direction_configs.items())
        model_configs: dict[str, object] = {}
        for key, config in ordered_items:
            model_configs[key] = ModelConfig(
                config.model_path,
                tokenizer_id=config.tokenizer_model,
                train_hf2=config.train_hf2,
            )

        base_paths = {str(getattr(model_config, "base_model_path", "")) for model_config in model_configs.values()}
        if len(base_paths) != 1:
            raise ValueError(
                "Shared LoRA worker requires all directions to use the same base model. "
                f"Resolved base models: {sorted(base_paths)}"
            )

        non_lora_keys = [
            key for key, model_config in model_configs.items()
            if not getattr(model_config, "use_lora", False)
        ]
        if non_lora_keys:
            raise ValueError(
                "Shared LoRA worker requires LoRA adapter models for every direction. "
                f"Non-LoRA direction(s): {', '.join(non_lora_keys)}"
            )

        primary_key, primary_config = ordered_items[0]
        generation_config = GenerationConfig(
            num_gpus=1,
            gpu_mem_util=primary_config.gpu_mem_util,
            sampling_params="greedy",
        )
        generation_config.repetition_penalty = None
        translator = vllm_translator(model_configs[primary_key], generation_config)

        self._translator = translator
        self._backends: dict[str, _DirectionWorkerBackend] = {}
        for lora_id, (key, config) in enumerate(ordered_items, start=1):
            model_config = model_configs[key]
            if getattr(model_config, "use_lora", False):
                lora_request = LoRARequest(
                    f"{key}_adapter",
                    lora_id,
                    getattr(model_config, "adapter_path"),
                )
            else:
                lora_request = None
            self._backends[key] = _DirectionWorkerBackend(
                config,
                translator=translator,
                model_config=model_config,
                lora_request=lora_request,
                owns_translator=False,
            )

    def translate_segments(
        self,
        direction_key: str,
        segments: Sequence[str],
        fewshot_count: int,
        method_key: str,
    ) -> List[str]:
        if direction_key not in self._backends:
            raise ValueError(f"Unsupported shared-worker direction: {direction_key}")
        return self._backends[direction_key].translate_segments(
            segments,
            fewshot_count,
            method_key,
        )

    def close(self) -> None:
        if hasattr(self._translator, "close"):
            self._translator.close()


def _worker_main(config: DirectionConfig, request_queue: mp.Queue, response_queue: mp.Queue) -> None:
    backend: _DirectionWorkerBackend | None = None
    try:
        _configure_worker_env(config)
        backend = _DirectionWorkerBackend(config)
        response_queue.put({"type": "ready", "direction": config.key})

        while True:
            message = request_queue.get()
            if message.get("type") == "shutdown":
                break
            request_id = message["request_id"]
            try:
                translations = backend.translate_segments(
                    message["segments"],
                    int(message["fewshot_count"]),
                    str(message.get("method_key", "fewshot_baseline")),
                )
                response_queue.put(
                    {
                        "type": "result",
                        "direction": config.key,
                        "request_id": request_id,
                        "translations": translations,
                    }
                )
            except Exception as exc:
                response_queue.put(
                    {
                        "type": "error",
                        "direction": config.key,
                        "request_id": request_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as exc:
        response_queue.put(
            {
                "type": "startup_error",
                "direction": config.key,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass


def _shared_lora_worker_main(
    direction_configs: dict[str, DirectionConfig],
    request_queue: mp.Queue,
    response_queue: mp.Queue,
) -> None:
    backend: _SharedLoraWorkerBackend | None = None
    first_config = next(iter(direction_configs.values()))
    try:
        _configure_worker_env(first_config)
        backend = _SharedLoraWorkerBackend(direction_configs)
        response_queue.put({"type": "ready", "directions": list(direction_configs)})

        while True:
            message = request_queue.get()
            if message.get("type") == "shutdown":
                break
            request_id = message["request_id"]
            direction_key = str(message.get("direction_key") or "")
            try:
                translations = backend.translate_segments(
                    direction_key,
                    message["segments"],
                    int(message["fewshot_count"]),
                    str(message.get("method_key", "fewshot_baseline")),
                )
                response_queue.put(
                    {
                        "type": "result",
                        "direction": direction_key,
                        "request_id": request_id,
                        "translations": translations,
                    }
                )
            except Exception as exc:
                response_queue.put(
                    {
                        "type": "error",
                        "direction": direction_key,
                        "request_id": request_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as exc:
        response_queue.put(
            {
                "type": "startup_error",
                "direction": ",".join(direction_configs),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass


class FewshotAppBackend:
    def __init__(
        self,
        direction_configs: dict[str, DirectionConfig],
        *,
        method_keys: Sequence[str] | None = None,
        startup_timeout_s: float = 1800.0,
        request_timeout_s: float = 900.0,
    ) -> None:
        self.direction_configs = dict(direction_configs)
        if not self.direction_configs:
            raise ValueError("At least one translation direction must be configured.")
        self.method_keys = tuple(method_keys or ("fewshot_baseline", "segment_mt"))
        invalid_methods = [key for key in self.method_keys if key not in METHOD_LABELS]
        if invalid_methods:
            raise ValueError(f"Unsupported method key(s): {', '.join(invalid_methods)}")
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self._ctx = mp.get_context("spawn")
        self._workers: dict[str, dict[str, object]] = {}
        self._job_lock = threading.RLock()
        self._jobs: dict[str, TranslationJob] = {}
        self._job_queues: dict[str, queue.Queue[str | None]] = {}
        self._dispatchers: dict[str, threading.Thread] = {}
        self._progress_chunk_size = max(1, int(DEFAULT_PROGRESS_CHUNK_SIZE))
        self._job_state_path = DEFAULT_JOB_STATE_PATH
        self._closed = False
        self._start_workers()
        self._start_dispatchers()
        self._load_persisted_jobs()
        self._recover_pending_jobs_after_restart()
        atexit.register(self.close)

    def _resolve_method_key(self, method_key: str) -> str:
        normalized = (method_key or "").strip()
        if normalized not in self.method_keys:
            supported = ", ".join(METHOD_LABELS[key] for key in self.method_keys)
            raise ValueError(f"Unsupported method: {normalized or '<empty>'}. Available method(s): {supported}.")
        return normalized

    def _get_active_job_locked(self) -> TranslationJob | None:
        active_jobs = [
            job for job in self._jobs.values()
            if job.state in ACTIVE_JOB_STATES
        ]
        if not active_jobs:
            return None
        return min(
            active_jobs,
            key=lambda item: (
                ACTIVE_JOB_STATE_PRIORITY.get(item.state, 99),
                -item.created_at,
            ),
        )

    def _raise_if_accepting_new_job_locked(self) -> None:
        active_job = self._get_active_job_locked()
        if active_job is None:
            return
        raise RuntimeError(
            f"Job {active_job.job_id} is already running. "
            "Wait for it to finish or cancel it first."
        )

    def _ensure_accepting_new_job(self) -> None:
        with self._job_lock:
            self._raise_if_accepting_new_job_locked()

    def _persist_jobs_locked(self) -> None:
        try:
            self._job_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": time.time(),
                "jobs": [_serialize_job(job) for job in self._jobs.values()],
            }
            tmp_path = self._job_state_path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            tmp_path.replace(self._job_state_path)
        except Exception:
            pass

    def _load_persisted_jobs(self) -> None:
        if not self._job_state_path.is_file():
            return
        try:
            with self._job_state_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return

        raw_jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
        if not isinstance(raw_jobs, list):
            return

        with self._job_lock:
            for item in raw_jobs:
                if not isinstance(item, dict):
                    continue
                try:
                    job = _deserialize_job(item)
                except Exception:
                    continue
                if job.job_id:
                    self._jobs[job.job_id] = job

    def _recover_pending_jobs_after_restart(self) -> None:
        recovered_ids: list[tuple[float, str, str]] = []
        with self._job_lock:
            for job in self._jobs.values():
                if job.state not in ACTIVE_JOB_STATES:
                    continue
                job.state = "queued"
                job.stage = "Recovered after restart. Waiting to resume."
                job.cancel_requested = False
                job.started_at = None
                job.finished_at = None
                job.completed_segments = 0
                job.progress_percent = 0.0
                job.translation = ""
                job.translated_file_path = None
                job.translated_pdf_path = None
                recovered_ids.append((job.created_at, job.direction_key, job.job_id))
            if recovered_ids:
                self._persist_jobs_locked()

        for _, direction_key, job_id in sorted(recovered_ids):
            if direction_key in self._job_queues:
                self._job_queues[direction_key].put(job_id)

    def _format_method_status(
        self,
        *,
        method_key: str,
        fewshot_count: int,
        segment_window_size: int,
    ) -> str:
        method_label = METHOD_LABELS.get(method_key, method_key)
        if method_key == "fewshot_baseline":
            return f"Method: {method_label} | Few-shot examples per segment: {max(0, int(fewshot_count))}"
        return f"Method: {method_label} | Segment window size: {max(1, int(segment_window_size))}"

    def _should_use_shared_lora_worker(self) -> bool:
        if len(self.direction_configs) < 2:
            return False
        enabled = os.environ.get("MFDS_GRADIO_SHARED_LORA", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return False
        visible_devices = {config.visible_devices.strip() for config in self.direction_configs.values()}
        if len(visible_devices) != 1:
            return False
        shared_device = next(iter(visible_devices))
        return bool(shared_device)

    def _start_workers(self) -> None:
        if self._should_use_shared_lora_worker():
            direction_keys = list(self.direction_configs)
            request_queue: mp.Queue = self._ctx.Queue()
            response_queue: mp.Queue = self._ctx.Queue()
            process = self._ctx.Process(
                target=_shared_lora_worker_main,
                args=(self.direction_configs, request_queue, response_queue),
                # vLLM starts child processes internally, so the worker itself
                # must not be a daemon process.
                daemon=False,
                name=f"fewshot-app-shared-{'-'.join(direction_keys)}",
            )
            process.start()
            worker_info = {
                "process": process,
                "request_queue": request_queue,
                "response_queue": response_queue,
                "worker_id": f"shared:{','.join(direction_keys)}",
            }
            for key in direction_keys:
                self._workers[key] = worker_info
            self._wait_for_worker_ready(direction_keys[0])
            return

        for key, config in self.direction_configs.items():
            request_queue: mp.Queue = self._ctx.Queue()
            response_queue: mp.Queue = self._ctx.Queue()
            process = self._ctx.Process(
                target=_worker_main,
                args=(config, request_queue, response_queue),
                # vLLM starts child processes internally, so the worker itself
                # must not be a daemon process.
                daemon=False,
                name=f"fewshot-app-{key}",
            )
            process.start()
            self._workers[key] = {
                "process": process,
                "request_queue": request_queue,
                "response_queue": response_queue,
                "worker_id": key,
            }

        for key in self.direction_configs:
            self._wait_for_worker_ready(key)

    def _start_dispatchers(self) -> None:
        worker_groups: dict[str, list[str]] = {}
        for key in self.direction_configs:
            worker_id = str(self._workers[key].get("worker_id", key))
            worker_groups.setdefault(worker_id, []).append(key)

        for worker_id, direction_keys in worker_groups.items():
            job_queue: queue.Queue[str | None] = queue.Queue()
            dispatcher = threading.Thread(
                target=self._dispatcher_main,
                args=(worker_id, job_queue),
                daemon=True,
                name=f"fewshot-dispatcher-{worker_id}",
            )
            dispatcher.start()
            for key in direction_keys:
                self._job_queues[key] = job_queue
            self._dispatchers[worker_id] = dispatcher

    def _wait_for_worker_ready(self, direction_key: str) -> None:
        worker = self._workers[direction_key]
        response_queue: mp.Queue = worker["response_queue"]  # type: ignore[assignment]
        try:
            message = response_queue.get(timeout=self.startup_timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Timed out while starting worker '{direction_key}'. "
                "Check model paths, GPU assignment, and vLLM startup logs."
            ) from exc
        if message.get("type") == "ready":
            return
        if message.get("type") in {"startup_error", "error"}:
            raise RuntimeError(
                f"Worker '{direction_key}' failed to start:\n{message.get('traceback') or message.get('error')}"
            )
        raise RuntimeError(f"Unexpected startup message for worker '{direction_key}': {message}")

    def _resolve_direction_config(self, direction_key: str) -> DirectionConfig:
        if direction_key not in self.direction_configs:
            supported = ", ".join(config.label for config in self.direction_configs.values())
            requested = "Korean -> English" if direction_key == "ko_en" else "English -> Korean"
            raise ValueError(
                f"This deployment does not support {requested}. "
                f"Available direction(s): {supported}."
            )
        return self.direction_configs[direction_key]

    def _translate_segment_batch(
        self,
        direction_key: str,
        segments: Sequence[str],
        fewshot_count: int,
        method_key: str,
    ) -> list[str]:
        if not segments:
            return []
        request_id = uuid.uuid4().hex
        worker = self._workers[direction_key]
        request_queue: mp.Queue = worker["request_queue"]  # type: ignore[assignment]
        response_queue: mp.Queue = worker["response_queue"]  # type: ignore[assignment]
        request_queue.put(
            {
                "type": "translate",
                "request_id": request_id,
                "direction_key": direction_key,
                "segments": list(segments),
                "fewshot_count": int(fewshot_count),
                "method_key": str(method_key),
            }
        )
        message = self._wait_for_response(response_queue, request_id)
        if message.get("type") == "error":
            raise RuntimeError(message.get("traceback") or message.get("error") or "Translation failed.")
        translations = [_normalize_translated_segment_text(item) for item in message.get("translations", [])]
        if len(translations) != len(segments):
            raise RuntimeError(
                f"Worker returned {len(translations)} translations for {len(segments)} input segments."
            )
        # Some model/method combinations occasionally return blank strings for
        # individual segments. For structured inputs like JSON tables, writing
        # those blanks back destroys the original content, so we keep the
        # source segment when the translated output is empty.
        translations = [
            translated if translated else _normalize_translated_segment_text(source_segment)
            for source_segment, translated in zip(segments, translations)
        ]
        return translations

    def _translate_segments_with_progress(
        self,
        *,
        direction_key: str,
        segments: Sequence[str],
        fewshot_count: int,
        method_key: str,
        progress_callback: Callable[[float, str], None] | None = None,
        job_id: str | None = None,
        partial_translation_callback: Callable[[Sequence[str]], None] | None = None,
    ) -> list[str]:
        if not segments:
            return []

        config = self._resolve_direction_config(direction_key)
        total_segments = len(segments)
        chunk_size = max(1, min(config.batch_size, self._progress_chunk_size))
        translated_segments: list[str] = []

        for start in range(0, total_segments, chunk_size):
            if job_id is not None:
                self._raise_if_job_cancelled(job_id)
            end = min(total_segments, start + chunk_size)
            batch_segments = segments[start:end]
            batch_translation = self._translate_segment_batch(
                direction_key,
                batch_segments,
                fewshot_count,
                method_key,
            )
            translated_segments.extend(batch_translation)
            if partial_translation_callback is not None:
                partial_translation_callback(translated_segments)

            completed = len(translated_segments)
            ratio = completed / total_segments
            stage = f"Translated {completed}/{total_segments} segments"

            if progress_callback is not None:
                progress_callback(0.35 + (ratio * 0.55), stage)

            if job_id is not None:
                with self._job_lock:
                    job = self._jobs.get(job_id)
                    if job is not None:
                        job.completed_segments = completed
                        job.progress_percent = min(90.0, 10.0 + (ratio * 80.0))
                        if not job.cancel_requested:
                            job.stage = stage
                        self._persist_jobs_locked()

        return translated_segments

    def translate_texts(
        self,
        texts: Sequence[str],
        fewshot_count: int,
        direction_key: str,
        method_key: str = "fewshot_baseline",
        segment_window_size: int = 1,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> List[str]:
        normalized_texts = [str(text or "").strip() for text in texts]
        if not any(normalized_texts):
            raise ValueError("Please enter source text.")

        config = self._resolve_direction_config(direction_key)
        method_key = self._resolve_method_key(method_key)
        fewshot_count = max(0, int(fewshot_count))

        if progress_callback is not None:
            progress_callback(0.15, "Preparing translation segments")

        if method_key == "segment_mt":
            units, all_segments = prepare_segment_mt_units(
                normalized_texts,
                window_size=segment_window_size,
            )
        else:
            units, all_segments = prepare_text_units(normalized_texts)
        if not all_segments:
            raise ValueError("Could not extract translation segments from the input.")

        if progress_callback is not None:
            progress_callback(0.35, f"Sending {len(all_segments)} segments to the {config.label} worker")

        translations = self._translate_segments_with_progress(
            direction_key=direction_key,
            segments=all_segments,
            fewshot_count=fewshot_count,
            method_key=method_key,
            progress_callback=progress_callback,
        )

        if progress_callback is not None:
            progress_callback(0.9, "Rebuilding translated text")
            progress_callback(1.0, "Done")
        return rebuild_prepared_units(units, translations)

    def translate(
        self,
        text: str,
        fewshot_count: int,
        direction_key: str,
        method_key: str = "fewshot_baseline",
        segment_window_size: int = 1,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TranslationResult:
        source_text = (text or "").strip()
        if not source_text:
            raise ValueError("Please enter source text.")

        segments, layout = segment_input_text(source_text)
        if not segments:
            raise ValueError("Could not extract sentence segments from the input.")
        config = self._resolve_direction_config(direction_key)
        method_key = self._resolve_method_key(method_key)
        segment_window_size = max(1, int(segment_window_size))

        merged_translation = self.translate_texts(
            [source_text],
            fewshot_count,
            direction_key,
            method_key=method_key,
            segment_window_size=segment_window_size,
            progress_callback=progress_callback,
        )[0]
        return TranslationResult(
            direction_key=direction_key,
            direction_label=config.label,
            segment_count=len(segments),
            translation=merged_translation,
            status=(
                f"Direction: {config.label} | Segments: {len(segments)} | "
                f"{self._format_method_status(method_key=method_key, fewshot_count=fewshot_count, segment_window_size=segment_window_size)} | "
                f"GPU: {config.visible_devices or 'default'}"
            ),
        )

    def translate_pdf(
        self,
        pdf_path: str,
        fewshot_count: int,
        direction_key: str,
        method_key: str = "fewshot_baseline",
        segment_window_size: int = 1,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> PdfTranslationResult:
        config = self._resolve_direction_config(direction_key)
        method_key = self._resolve_method_key(method_key)
        segment_window_size = max(1, int(segment_window_size))
        if progress_callback is not None:
            progress_callback(0.05, "Reading PDF layout and text blocks")

        pdf_file, page_count, blocks, extracted_text = extract_text_blocks_from_pdf(pdf_path)
        block_texts = [block.text for block in blocks]

        translated_blocks = self.translate_texts(
            block_texts,
            fewshot_count,
            direction_key,
            method_key=method_key,
            segment_window_size=segment_window_size,
            progress_callback=_scale_progress(progress_callback, 0.15, 0.75),
        )

        if progress_callback is not None:
            progress_callback(0.82, "Writing translated PDF")
        translated_pdf_path = render_translated_pdf(
            original_pdf_path=pdf_file,
            blocks=blocks,
            translated_blocks=build_pdf_render_blocks(blocks, translated_blocks),
            direction_key=direction_key,
        )

        if progress_callback is not None:
            progress_callback(0.95, "Preparing translated PDF download")
        merged_translation = rebuild_text_from_pdf_blocks(blocks, translated_blocks)
        if progress_callback is not None:
            progress_callback(1.0, "Done")

        return PdfTranslationResult(
            direction_key=direction_key,
            direction_label=config.label,
            page_count=page_count,
            block_count=len(blocks),
            extracted_text=extracted_text,
            translation=merged_translation,
            translated_pdf_path=translated_pdf_path,
            status=(
                f"Direction: {config.label} | Pages: {page_count} | Text blocks: {len(blocks)} | "
                f"{self._format_method_status(method_key=method_key, fewshot_count=fewshot_count, segment_window_size=segment_window_size)} | "
                f"GPU: {config.visible_devices or 'default'}"
            ),
        )

    def _wait_for_response(self, response_queue: mp.Queue, request_id: str) -> dict[str, object]:
        while True:
            try:
                message = response_queue.get(timeout=self.request_timeout_s)
            except queue.Empty as exc:
                raise TimeoutError("Timed out while waiting for translation output.") from exc
            if message.get("request_id") == request_id:
                return message

    def submit_text_job(
        self,
        text: str,
        fewshot_count: int,
        direction_key: str,
        *,
        method_key: str = "fewshot_baseline",
        segment_window_size: int = 1,
    ) -> str:
        source_text = (text or "").strip()
        if not source_text:
            raise ValueError("Please enter source text.")
        self._ensure_accepting_new_job()

        config = self._resolve_direction_config(direction_key)
        method_key = self._resolve_method_key(method_key)
        segment_window_size = max(1, int(segment_window_size))
        if method_key == "segment_mt":
            units, segments = prepare_segment_mt_units([source_text], window_size=segment_window_size)
        else:
            units, segments = prepare_text_units([source_text])
        if not segments:
            raise ValueError("Could not extract sentence segments from the input.")

        job_id = uuid.uuid4().hex[:12]
        job = TranslationJob(
            job_id=job_id,
            direction_key=direction_key,
            direction_label=config.label,
            method_key=method_key,
            method_label=METHOD_LABELS[method_key],
            input_kind="text",
            fewshot_count=max(0, int(fewshot_count)),
            segment_window_size=segment_window_size,
            created_at=time.time(),
            units=units,
            segments=segments,
            extracted_text=source_text,
            total_segments=len(segments),
            stage="Queued",
        )
        with self._job_lock:
            self._raise_if_accepting_new_job_locked()
            self._jobs[job_id] = job
            self._persist_jobs_locked()
        self._job_queues[direction_key].put(job_id)
        return job_id

    def submit_pdf_job(
        self,
        pdf_path: str,
        fewshot_count: int,
        direction_key: str,
        *,
        method_key: str = "fewshot_baseline",
        segment_window_size: int = 1,
    ) -> str:
        self._ensure_accepting_new_job()
        config = self._resolve_direction_config(direction_key)
        method_key = self._resolve_method_key(method_key)
        segment_window_size = max(1, int(segment_window_size))
        pdf_file, page_count, blocks, extracted_text = extract_text_blocks_from_pdf(pdf_path)
        block_texts = [block.text for block in blocks]
        if method_key == "segment_mt":
            units, segments = prepare_segment_mt_units(block_texts, window_size=segment_window_size)
        else:
            units, segments = prepare_text_units(block_texts)
        if not segments:
            raise ValueError("Could not extract sentence segments from the PDF text blocks.")

        job_id = uuid.uuid4().hex[:12]
        job = TranslationJob(
            job_id=job_id,
            direction_key=direction_key,
            direction_label=config.label,
            method_key=method_key,
            method_label=METHOD_LABELS[method_key],
            input_kind="pdf",
            fewshot_count=max(0, int(fewshot_count)),
            segment_window_size=segment_window_size,
            created_at=time.time(),
            units=units,
            segments=segments,
            extracted_text=extracted_text,
            pdf_source_path=str(pdf_file),
            pdf_blocks=list(blocks),
            page_count=page_count,
            block_count=len(blocks),
            total_segments=len(segments),
            stage="Queued",
        )
        with self._job_lock:
            self._raise_if_accepting_new_job_locked()
            self._jobs[job_id] = job
            self._persist_jobs_locked()
        self._job_queues[direction_key].put(job_id)
        return job_id

    def submit_json_job(
        self,
        json_path: str,
        fewshot_count: int,
        direction_key: str,
        *,
        method_key: str = "fewshot_baseline",
        segment_window_size: int = 1,
    ) -> str:
        self._ensure_accepting_new_job()
        config = self._resolve_direction_config(direction_key)
        method_key = self._resolve_method_key(method_key)
        segment_window_size = max(1, int(segment_window_size))
        json_file, payload, json_entries, extracted_text = extract_text_entries_from_json(json_path)
        entry_texts = [entry.original_text for entry in json_entries]
        if method_key == "segment_mt":
            units, segments = prepare_segment_mt_units(entry_texts, window_size=segment_window_size)
        else:
            units, segments = prepare_text_units(entry_texts)
        if not segments:
            raise ValueError("Could not extract sentence segments from the JSON string values.")

        job_id = uuid.uuid4().hex[:12]
        job = TranslationJob(
            job_id=job_id,
            direction_key=direction_key,
            direction_label=config.label,
            method_key=method_key,
            method_label=METHOD_LABELS[method_key],
            input_kind="json",
            fewshot_count=max(0, int(fewshot_count)),
            segment_window_size=segment_window_size,
            created_at=time.time(),
            units=units,
            segments=segments,
            extracted_text=extracted_text,
            json_source_path=str(json_file),
            json_payload=payload,
            json_entries=list(json_entries),
            total_segments=len(segments),
            stage="Queued",
        )
        with self._job_lock:
            self._raise_if_accepting_new_job_locked()
            self._jobs[job_id] = job
            self._persist_jobs_locked()
        self._job_queues[direction_key].put(job_id)
        return job_id

    def cancel_job(self, job_id: str) -> str:
        if not job_id:
            raise ValueError("No job is selected.")

        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError(f"Unknown job id: {job_id}")
            if job.state in {"completed", "failed", "cancelled"}:
                return f"Job {job_id} is already {job.state}."
            job.cancel_requested = True
            if job.state == "queued":
                job.state = "cancelled"
                job.stage = "Cancelled before start"
                job.finished_at = time.time()
                job.progress_percent = 0.0
                self._persist_jobs_locked()
                return f"Cancelled queued job {job_id}."
            job.state = "cancelling"
            job.stage = "Cancellation requested. Waiting for the current batch to finish."
            self._persist_jobs_locked()
            return f"Cancellation requested for job {job_id}."

    def get_job_snapshot(self, job_id: str | None) -> dict[str, Any] | None:
        if not job_id:
            return None
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return self._build_job_snapshot_locked(job)

    def resolve_current_job_id(self, job_id: str | None = None) -> str:
        with self._job_lock:
            if job_id:
                job = self._jobs.get(job_id)
                if job is not None:
                    return job.job_id

            if not self._jobs:
                return ""

            active_job = self._get_active_job_locked()
            if active_job is not None:
                return active_job.job_id

            latest_job = max(self._jobs.values(), key=lambda item: item.created_at)
            return latest_job.job_id

    def resolve_latest_result_job_id(self, job_id: str | None = None) -> str:
        with self._job_lock:
            if job_id:
                job = self._jobs.get(job_id)
                if job is not None and (
                    str(job.translation or "").strip()
                    or job.translated_file_path
                    or job.translated_pdf_path
                ):
                    return job.job_id

            candidates = [
                job
                for job in self._jobs.values()
                if (
                    str(job.translation or "").strip()
                    or job.translated_file_path
                    or job.translated_pdf_path
                )
            ]
            if not candidates:
                return ""

            latest_job = max(
                candidates,
                key=lambda item: (
                    item.finished_at or 0.0,
                    item.created_at,
                ),
            )
            return latest_job.job_id

    def get_latest_result_snapshot(self, job_id: str | None = None) -> dict[str, Any] | None:
        resolved_job_id = self.resolve_latest_result_job_id(job_id)
        if not resolved_job_id:
            return None
        return self.get_job_snapshot(resolved_job_id)

    def list_queue_rows(self) -> list[list[str]]:
        with self._job_lock:
            queue_positions = self._compute_queue_positions_locked()
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: (
                    {
                        "running": 0,
                        "cancelling": 1,
                        "queued": 2,
                        "completed": 3,
                        "failed": 4,
                        "cancelled": 5,
                    }.get(job.state, 6),
                    job.created_at,
                ),
            )
            rows: list[list[str]] = []
            for job in jobs:
                queue_position = queue_positions.get(job.job_id)
                progress_text = (
                    f"{compute_segment_progress_percent(job.completed_segments, job.total_segments):.1f}%"
                )
                segment_text = f"{job.completed_segments}/{job.total_segments}"
                rows.append(
                    [
                        job.job_id,
                        job.method_label,
                        job.direction_label,
                        job.input_kind.upper(),
                        job.state,
                        "" if queue_position is None else str(queue_position),
                        progress_text,
                        segment_text,
                        job.stage,
                        format_timestamp(job.created_at),
                    ]
                )
            return rows

    def _compute_queue_positions_locked(self) -> dict[str, int]:
        per_direction_jobs: dict[str, list[TranslationJob]] = {
            key: [] for key in self.direction_configs
        }
        for job in self._jobs.values():
            if job.state == "queued":
                per_direction_jobs.setdefault(job.direction_key, []).append(job)

        queue_positions: dict[str, int] = {}
        for jobs in per_direction_jobs.values():
            for position, job in enumerate(sorted(jobs, key=lambda item: item.created_at), start=1):
                queue_positions[job.job_id] = position
        return queue_positions

    def _build_job_snapshot_locked(self, job: TranslationJob) -> dict[str, Any]:
        queue_positions = self._compute_queue_positions_locked()
        queue_position = queue_positions.get(job.job_id)
        return {
            "job_id": job.job_id,
            "direction_label": job.direction_label,
            "method_key": job.method_key,
            "method_label": job.method_label,
            "input_kind": job.input_kind,
            "state": job.state,
            "stage": job.stage,
            "progress_percent": round(job.progress_percent, 1),
            "segment_progress_percent": round(
                compute_segment_progress_percent(job.completed_segments, job.total_segments),
                1,
            ),
            "completed_segments": job.completed_segments,
            "total_segments": job.total_segments,
            "queue_position": queue_position,
            "can_cancel": job.state in {"queued", "running", "cancelling"},
            "translation": job.translation,
            "translated_file_path": job.translated_file_path or job.translated_pdf_path,
            "translated_pdf_path": job.translated_pdf_path,
            "extracted_text": job.extracted_text,
            "error": job.error,
            "created_at": format_timestamp(job.created_at),
            "started_at": format_timestamp(job.started_at),
            "finished_at": format_timestamp(job.finished_at),
            "page_count": job.page_count,
            "block_count": job.block_count,
            "fewshot_count": job.fewshot_count,
            "segment_window_size": job.segment_window_size,
        }

    def _raise_if_job_cancelled(self, job_id: str) -> None:
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise _JobCancelledError("Job no longer exists.")
            if job.cancel_requested:
                raise _JobCancelledError(f"Job {job_id} was cancelled.")

    def _mark_job_terminal(
        self,
        job_id: str,
        *,
        state: str,
        stage: str,
        progress_percent: float | None = None,
        error: str = "",
    ) -> None:
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.state = state
            job.stage = stage
            job.finished_at = time.time()
            if progress_percent is not None:
                job.progress_percent = progress_percent
            if error:
                job.error = error
            self._persist_jobs_locked()

    def _dispatcher_main(self, direction_key: str, job_queue: queue.Queue[str | None]) -> None:
        while True:
            try:
                job_id = job_queue.get(timeout=0.5)
            except queue.Empty:
                if self._closed:
                    return
                continue

            if job_id is None or self._closed:
                return

            with self._job_lock:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                if job.state == "cancelled":
                    continue
                if job.cancel_requested:
                    job.state = "cancelled"
                    job.stage = "Cancelled before start"
                    job.finished_at = time.time()
                    self._persist_jobs_locked()
                    continue
                job.state = "running"
                job.stage = "Preparing translation"
                job.started_at = time.time()
                job.progress_percent = max(job.progress_percent, 1.0)
                self._persist_jobs_locked()

            try:
                self._run_job(job_id)
            except _JobCancelledError:
                self._mark_job_terminal(
                    job_id,
                    state="cancelled",
                    stage="Cancelled",
                )
            except Exception:
                self._mark_job_terminal(
                    job_id,
                    state="failed",
                    stage="Failed",
                    error=traceback.format_exc(),
                )

    def _run_job(self, job_id: str) -> None:
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise RuntimeError(f"Unknown job id: {job_id}")
            direction_key = job.direction_key
            method_key = job.method_key
            fewshot_count = job.fewshot_count
            segments = list(job.segments)
            units = list(job.units)
            input_kind = job.input_kind
            pdf_blocks = list(job.pdf_blocks)
            pdf_source_path = job.pdf_source_path
            json_source_path = job.json_source_path
            json_payload = copy.deepcopy(job.json_payload)
            json_entries = list(job.json_entries)

        with self._job_lock:
            job = self._jobs[job_id]
            job.stage = f"Queued segments prepared: {len(segments)} total"
            job.progress_percent = max(job.progress_percent, 5.0)
            self._persist_jobs_locked()

        last_preview_update_at = 0.0
        last_partial_file_path: str | None = None

        def update_partial_translation_preview(partial_segments: Sequence[str]) -> None:
            nonlocal last_preview_update_at, last_partial_file_path
            now = time.time()
            completed = len(partial_segments)
            should_force_update = completed >= len(segments)
            if not should_force_update and now - last_preview_update_at < 0.75:
                return

            merged_segments = list(partial_segments) + segments[completed:]
            rebuilt_partial_units = rebuild_prepared_units(units, merged_segments)
            partial_file_path: str | None = None
            if input_kind == "pdf":
                preview_text = rebuild_text_from_pdf_blocks(pdf_blocks, rebuilt_partial_units)
            elif input_kind == "json":
                preview_payload = build_translated_json_payload(
                    json_payload,
                    json_entries,
                    rebuilt_partial_units,
                )
                preview_text = json.dumps(preview_payload, ensure_ascii=False, indent=2)
                if json_source_path is not None:
                    try:
                        partial_file_path = render_translated_json(
                            original_json_path=Path(json_source_path),
                            payload=json_payload,
                            entries=json_entries,
                            translated_texts=rebuilt_partial_units,
                            direction_key=direction_key,
                            output_path=_build_json_output_path(
                                original_json_path=Path(json_source_path),
                                direction_key=direction_key,
                                job_id=job_id,
                                completed_segments=completed,
                                total_segments=len(segments),
                                final=False,
                            ),
                        )
                    except Exception:
                        partial_file_path = None
            else:
                preview_text = rebuilt_partial_units[0] if rebuilt_partial_units else ""

            with self._job_lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job.translation = preview_text
                if partial_file_path:
                    job.translated_file_path = partial_file_path
                    if input_kind == "pdf":
                        job.translated_pdf_path = partial_file_path
                self._persist_jobs_locked()

            if partial_file_path and last_partial_file_path and last_partial_file_path != partial_file_path:
                _cleanup_output_file(last_partial_file_path)
            if partial_file_path:
                last_partial_file_path = partial_file_path
            last_preview_update_at = now

        translated_segments = self._translate_segments_with_progress(
            direction_key=direction_key,
            segments=segments,
            fewshot_count=fewshot_count,
            method_key=method_key,
            job_id=job_id,
            partial_translation_callback=update_partial_translation_preview,
        )
        rebuilt_units = rebuild_prepared_units(units, translated_segments)

        self._raise_if_job_cancelled(job_id)

        translated_file_path: str | None = None
        translated_pdf_path: str | None = None
        if input_kind == "pdf":
            with self._job_lock:
                job = self._jobs[job_id]
                job.stage = "Finalizing translated segments"
                job.progress_percent = 95.0
                self._persist_jobs_locked()
            merged_translation = rebuild_text_from_pdf_blocks(pdf_blocks, rebuilt_units)
        elif input_kind == "json":
            with self._job_lock:
                job = self._jobs[job_id]
                job.stage = "Writing translated JSON"
                job.progress_percent = 95.0
                self._persist_jobs_locked()
            if json_source_path is None:
                raise RuntimeError("JSON job is missing source path.")
            translated_file_path = render_translated_json(
                original_json_path=Path(json_source_path),
                payload=json_payload,
                entries=json_entries,
                translated_texts=rebuilt_units,
                direction_key=direction_key,
                output_path=_build_json_output_path(
                    original_json_path=Path(json_source_path),
                    direction_key=direction_key,
                    job_id=job_id,
                    final=True,
                ),
            )
            merged_translation = json.dumps(
                build_translated_json_payload(json_payload, json_entries, rebuilt_units),
                ensure_ascii=False,
                indent=2,
            )
        else:
            merged_translation = rebuilt_units[0] if rebuilt_units else ""
            translated_file_path = render_translated_text(
                translation=merged_translation,
                direction_key=direction_key,
                job_id=job_id,
            )

        if translated_file_path and last_partial_file_path and translated_file_path != last_partial_file_path:
            _cleanup_output_file(last_partial_file_path)

        with self._job_lock:
            job = self._jobs[job_id]
            job.translation = merged_translation
            job.translated_file_path = translated_file_path
            job.translated_pdf_path = translated_pdf_path
            job.completed_segments = job.total_segments
            job.progress_percent = 100.0
            job.state = "completed"
            job.stage = "Completed"
            job.finished_at = time.time()
            self._persist_jobs_locked()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        seen_job_queues: set[int] = set()
        for job_queue in self._job_queues.values():
            queue_id = id(job_queue)
            if queue_id in seen_job_queues:
                continue
            seen_job_queues.add(queue_id)
            try:
                job_queue.put(None)
            except Exception:
                pass

        for dispatcher in self._dispatchers.values():
            dispatcher.join(timeout=2)

        seen_processes: set[int] = set()
        for worker in self._workers.values():
            request_queue: mp.Queue = worker["request_queue"]  # type: ignore[assignment]
            process: mp.Process = worker["process"]  # type: ignore[assignment]
            process_id = id(process)
            if process_id in seen_processes:
                continue
            seen_processes.add(process_id)
            try:
                request_queue.put({"type": "shutdown"})
            except Exception:
                pass
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
