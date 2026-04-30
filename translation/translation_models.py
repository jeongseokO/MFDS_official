from __future__ import annotations

import configparser
import inspect
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):
        return False

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers import GenerationConfig as TransformersGenerationConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import BeamSearchParams


load_dotenv()
logging.basicConfig(level=logging.INFO)


def _env_flag(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _load_hf_token_from_store(token_name: str | None = None) -> str | None:
    store_path = os.environ.get("HF_STORED_TOKENS_PATH")
    if not store_path:
        return None

    path = Path(store_path).expanduser()
    if not path.is_file():
        return None

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    section_name = token_name or os.environ.get("HF_TOKEN_NAME")
    if not section_name or not parser.has_section(section_name):
        return None

    raw = parser.get(section_name, "hf_token", fallback="").strip()
    return raw.strip("'\"") or None


def _load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_READ_TOKEN")
    if token:
        if token.startswith("hf_"):
            return token
        resolved = _load_hf_token_from_store(token_name=token)
        return resolved or token

    token_path = Path.home() / ".config" / "hf_token"
    if token_path.is_file():
        raw = token_path.read_text(encoding="utf-8").strip()
        if raw.startswith("hf_"):
            return raw
        resolved = _load_hf_token_from_store(token_name=raw)
        return resolved or raw or None

    return _load_hf_token_from_store()


def _ensure_writable_root(env_name: str, suffix: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []

    env_value = os.environ.get(env_name)
    if env_value:
        candidates.append(Path(env_value).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / suffix)

    candidates.extend(
        [
            repo_root / ".cache" / suffix,
            Path.cwd() / ".cache" / suffix,
            Path(tempfile.gettempdir()) / suffix,
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(candidate, os.W_OK):
            os.environ[env_name] = str(candidate)
            return str(candidate)

    raise RuntimeError(f"Unable to create a writable {env_name} directory.")


_ensure_writable_root("VLLM_CACHE_ROOT", "vllm")
_ensure_writable_root("VLLM_CONFIG_ROOT", "vllm_config")


def _iter_hf_cache_dirs() -> list[str]:
    candidates: list[Path] = []
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.getenv(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")

    xdg_cache = os.getenv("XDG_CACHE_HOME")
    if xdg_cache:
        candidates.append(Path(xdg_cache).expanduser() / "huggingface" / "hub")

    seen: set[str] = set()
    existing: list[str] = []
    for candidate in candidates:
        try:
            if not candidate.exists() or not candidate.is_dir():
                continue
        except OSError:
            continue
        value = str(candidate)
        if value in seen:
            continue
        seen.add(value)
        existing.append(value)
    return existing


def _resolve_cached_snapshot(repo_id: str, token: str | None) -> str | None:
    kwargs = {"repo_id": repo_id, "local_files_only": True}
    if token:
        kwargs["token"] = token

    for cache_dir in [None, *_iter_hf_cache_dirs()]:
        try:
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            elif "cache_dir" in kwargs:
                kwargs.pop("cache_dir", None)
            return snapshot_download(**kwargs)
        except Exception:
            continue
    return None


def _download_or_path(repo_or_path: str, token: str | None) -> str:
    if os.path.exists(repo_or_path):
        return repo_or_path
    candidate = Path(repo_or_path).expanduser()
    if candidate.is_absolute() or repo_or_path.startswith(("./", "../", "~")):
        raise FileNotFoundError(
            f"Model or adapter path does not exist: {repo_or_path}. "
            "Set FEWSHOT_BASELINE_MODEL_KO_EN and FEWSHOT_BASELINE_MODEL_EN_KO "
            "in .env to valid LoRA adapter directories or Hugging Face repo ids."
        )

    cached = _resolve_cached_snapshot(repo_or_path, token)
    if cached:
        return cached

    kwargs = {"repo_id": repo_or_path}
    if token:
        kwargs["token"] = token
    return snapshot_download(**kwargs)


def _prefer_cached_or_remote(repo_or_path: str | None, token: str | None) -> str | None:
    if not repo_or_path:
        return repo_or_path
    if os.path.exists(repo_or_path):
        return repo_or_path
    cached = _resolve_cached_snapshot(repo_or_path, token)
    return cached or repo_or_path


class GenerationConfig(TransformersGenerationConfig):
    def __init__(
        self,
        num_gpus: int,
        gpu_mem_util: float,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_retries: int = 10,
        sampling_params: str = "greedy",
        beam_width: int = 3,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_gpus = num_gpus
        self.gpu_mem_util = gpu_mem_util
        self.max_retries = max_retries
        self.sampling_params = sampling_params
        self.beam_width = beam_width


class ModelConfig:
    def __init__(
        self,
        model_id: str,
        tokenizer_id: str | None = None,
        adapter_path: str | None = None,
        *,
        train_hf2: bool = False,
    ) -> None:
        self.original_model_id = model_id
        self.model_id = model_id
        self.base_model_path = model_id
        self.adapter_path = adapter_path
        self.use_lora = adapter_path is not None
        self.adapter_config: dict[str, Any] | None = None

        token = _load_hf_token()
        if train_hf2:
            self._configure_peft_adapter(model_id, tokenizer_id, token)
        else:
            self.base_model_path = _prefer_cached_or_remote(model_id, token)
            self.tokenizer_id = _prefer_cached_or_remote(tokenizer_id or model_id, token)

        prompt_reference = str(self.base_model_path or self.model_id)
        self.requires_system_prompt = not (
            "TowerInstruct" in prompt_reference or "vaiv" in prompt_reference.lower()
        )

    def _configure_peft_adapter(
        self,
        adapter_id: str,
        tokenizer_id: str | None,
        token: str | None,
    ) -> None:
        adapter_dir = Path(_download_or_path(adapter_id, token))
        adapter_config_path = adapter_dir / "adapter_config.json"
        if not adapter_config_path.exists():
            raise FileNotFoundError(
                f"adapter_config.json not found in {adapter_dir}. "
                "FEWSHOT_BASELINE_MODEL_* must point to a PEFT/LoRA adapter directory "
                "or Hugging Face adapter repository."
            )

        import json

        with adapter_config_path.open("r", encoding="utf-8") as handle:
            self.adapter_config = json.load(handle)

        base_model_name = self.adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(f"Missing base_model_name_or_path in {adapter_config_path}.")

        override_base = os.getenv("HF2_BASE_MODEL") or os.getenv("HF2_BASE_MODEL_PATH")
        if override_base:
            base_model_name = override_base

        base_local_candidate = adapter_dir / str(base_model_name)
        if base_local_candidate.exists():
            resolved_base = str(base_local_candidate)
        else:
            resolved_base = _prefer_cached_or_remote(str(base_model_name), token)

        resolved_path = Path(str(resolved_base))
        if resolved_path.is_dir() and not (resolved_path / "config.json").exists():
            subdir = resolved_path / "pretrained_models"
            if (subdir / "config.json").exists():
                resolved_base = str(subdir)

        self.base_model_path = str(resolved_base)
        self.adapter_path = str(adapter_dir)
        self.use_lora = True
        self.tokenizer_id = (
            tokenizer_id
            or self.adapter_config.get("tokenizer_name_or_path")
            or self.base_model_path
        )

        logging.info(
            "Configured LoRA adapter '%s' with base '%s'.",
            self.adapter_path,
            self.base_model_path,
        )


def _flat_tail_ratio(token_ids: list[int] | None) -> float | None:
    if not token_ids:
        return None
    cleaned = [tok for tok in token_ids if isinstance(tok, int)]
    if not cleaned:
        return None
    unique_total = len(set(cleaned))
    seen: set[int] = set()
    hit_idx = None
    for idx, tok in enumerate(cleaned):
        if tok not in seen:
            seen.add(tok)
            if len(seen) == unique_total:
                hit_idx = idx
                break
    if hit_idx is None:
        return None
    return float((len(cleaned) - hit_idx) / len(cleaned))


class vllm_translator:
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        device: str | None = None,
    ) -> None:
        self.model_config = model_config
        self.generation_config = generation_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        hf_token = _load_hf_token()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer_id,
            token=hf_token,
        )

        terminators = [self.tokenizer.eos_token_id]
        if "OpenBioLLM" in self.model_config.model_id:
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        self._base_sampling_kwargs = {
            "max_tokens": generation_config.max_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "stop_token_ids": terminators,
        }
        if _env_flag("STOP_STRING_NEWLINE"):
            self._base_sampling_kwargs["stop"] = ["\n"]
        repetition_penalty = getattr(generation_config, "repetition_penalty", None)
        if repetition_penalty is not None:
            self._base_sampling_kwargs["repetition_penalty"] = repetition_penalty

        if self.generation_config.sampling_params == "greedy":
            self.sampling_params = SamplingParams(**self._base_sampling_kwargs)
        else:
            beam_kwargs = dict(self._base_sampling_kwargs)
            beam_kwargs.pop("top_p", None)
            beam_kwargs.pop("stop_token_ids", None)
            beam_kwargs.pop("stop", None)
            beam_kwargs["beam_width"] = self.generation_config.beam_width
            beam_kwargs["length_penalty"] = 0.5
            self.sampling_params = BeamSearchParams(**beam_kwargs)

        llm_kwargs: dict[str, Any] = {
            "model": model_config.base_model_path,
            "dtype": torch.bfloat16,
            "tensor_parallel_size": generation_config.num_gpus,
            "gpu_memory_utilization": generation_config.gpu_mem_util,
            "enable_lora": model_config.use_lora,
            "max_lora_rank": int(os.environ.get("VLLM_MAX_LORA_RANK", "64")),
            "enforce_eager": _env_flag("VLLM_ENFORCE_EAGER", "1"),
            "max_model_len": int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")),
        }

        attention_backend = os.getenv("VLLM_ATTENTION_BACKEND", "").strip()
        if attention_backend:
            try:
                from vllm.engine.arg_utils import EngineArgs

                if "attention_config" in inspect.signature(EngineArgs).parameters:
                    llm_kwargs["attention_config"] = {"backend": attention_backend}
            except Exception as exc:
                logging.warning("Could not inspect vLLM attention_config support: %s", exc)

        self.llm = LLM(**llm_kwargs)
        self.supports_multi_path = True
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        llm_engine = getattr(getattr(self, "llm", None), "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            try:
                engine_core.shutdown()
            except Exception as exc:
                logging.warning("Failed to shut down vLLM engine core: %s", exc)
        output_processor = getattr(llm_engine, "output_processor", None)
        if output_processor is not None and hasattr(output_processor, "close"):
            try:
                output_processor.close()
            except Exception as exc:
                logging.warning("Failed to close vLLM output processor: %s", exc)
        self.llm = None
        self._closed = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def apply_chat_template(self, messages_list: list[list[dict[str, str]]]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_list
        ]

    def _default_lora_request(self):
        if self.model_config.use_lora:
            return LoRARequest(
                str(self.model_config.model_id),
                1,
                str(self.model_config.adapter_path),
            )
        return None

    def _make_sampling_params(
        self,
        n: int = 1,
        *,
        logprobs: int | None = None,
    ) -> SamplingParams:
        params_kwargs = dict(self._base_sampling_kwargs)
        params_kwargs["n"] = max(1, int(n))
        if logprobs is not None:
            params_kwargs["logprobs"] = logprobs
        return SamplingParams(**params_kwargs)

    def _truncate_prompts(self, text_list: list[str]) -> list[str]:
        prompts = list(text_list)
        max_model_len = int(self.llm.llm_engine.model_config.max_model_len)
        for idx, prompt in enumerate(prompts):
            token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(token_ids) > max_model_len - 10:
                prompts[idx] = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": ""}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return prompts

    def _generate_request_outputs(
        self,
        text_list: list[str],
        *,
        num_paths: int = 1,
        request_logprobs: bool = False,
        lora_request=None,
    ):
        lora_request = self._default_lora_request() if lora_request is None else lora_request
        prompts = self._truncate_prompts(text_list)
        sampling_params = self._make_sampling_params(
            n=num_paths,
            logprobs=1 if request_logprobs else None,
        )
        return self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

    @staticmethod
    def _extract_assistant(text: str) -> str:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>", 1)[-1]
        return text.split("<|eot_id|>", 1)[0].strip()

    def _generate_output(self, text_list: list[str], *, lora_request=None) -> list[str]:
        if self.generation_config.sampling_params != "greedy":
            return [
                result["mt_paths"][0] if result.get("mt_paths") else ""
                for result in self._generate_beam_outputs(text_list, 1, lora_request=lora_request)
            ]

        request_outputs = self._generate_request_outputs(text_list, lora_request=lora_request)
        return [req.outputs[0].text if req.outputs else "" for req in request_outputs]

    def _generate_multi_path_outputs(
        self,
        text_list: list[str],
        num_paths: int,
        *,
        lora_request=None,
    ) -> list[dict[str, Any]]:
        request_outputs = self._generate_request_outputs(
            text_list,
            num_paths=num_paths,
            request_logprobs=True,
            lora_request=lora_request,
        )
        results: list[dict[str, Any]] = []
        for request in request_outputs:
            sorted_outputs = sorted(
                getattr(request, "outputs", []) or [],
                key=lambda out: getattr(out, "index", 0),
            )
            mt_paths = [out.text for out in sorted_outputs]
            token_lengths = []
            flat_tail_ratios = []
            cum_logprobs = []
            for out in sorted_outputs:
                token_ids = getattr(out, "token_ids", None) or getattr(out, "output_token_ids", None)
                token_ids = list(token_ids) if token_ids is not None else None
                token_lengths.append(len(token_ids) if token_ids is not None else None)
                flat_tail_ratios.append(_flat_tail_ratio(token_ids))
                cumulative = getattr(out, "cumulative_logprob", None)
                cum_logprobs.append(float(cumulative) if cumulative is not None else None)
            results.append(
                {
                    "mt_paths": mt_paths,
                    "cum_logprobs": cum_logprobs,
                    "token_lengths": token_lengths,
                    "flat_tail_ratios": flat_tail_ratios,
                }
            )
        return results

    def _generate_beam_outputs(
        self,
        text_list: list[str],
        num_paths: int,
        *,
        lora_request=None,
    ) -> list[dict[str, Any]]:
        lora_request = self._default_lora_request() if lora_request is None else lora_request
        inputs = [{"prompt": text} for text in self._truncate_prompts(text_list)]
        outputs = self.llm.beam_search(
            prompts=inputs,
            params=self.sampling_params,
            lora_request=lora_request,
        )

        results: list[dict[str, Any]] = []
        for output in outputs:
            sequences = sorted(
                getattr(output, "sequences", []) or [],
                key=lambda seq: getattr(seq, "index", 0),
            )
            mt_paths = []
            cum_logprobs = []
            token_lengths = []
            flat_tail_ratios = []
            for seq in sequences[:num_paths]:
                text = self._extract_assistant(getattr(seq, "text", ""))
                mt_paths.append(text.split("\n", 1)[0].strip())
                token_ids = list(getattr(seq, "tokens", []) or [])
                token_lengths.append(len(token_ids))
                flat_tail_ratios.append(_flat_tail_ratio(token_ids))
                cumulative = getattr(seq, "cum_logprob", None)
                cum_logprobs.append(float(cumulative) if cumulative is not None else None)
            results.append(
                {
                    "mt_paths": mt_paths,
                    "cum_logprobs": cum_logprobs,
                    "token_lengths": token_lengths,
                    "flat_tail_ratios": flat_tail_ratios,
                }
            )
        return results

    def create_simple_translation_messages_list(
        self,
        src_list: list[str],
        source_lang: str,
        target_lang: str,
        **_unused,
    ) -> list[list[dict[str, str]]]:
        messages_list: list[list[dict[str, str]]] = []
        for src in src_list:
            instruction_line = (
                f"You will be provided with a document in {source_lang}, "
                f"and your task is to translate it into {target_lang}."
            )
            translate_line = "Just translate the document and don't provide any additional comments."
            user_payload = "\n".join([instruction_line, translate_line, str(src)])
            if self.model_config.requires_system_prompt:
                messages = [
                    {"role": "system", "content": "You are a translator."},
                    {"role": "user", "content": user_payload},
                ]
            else:
                messages = [{"role": "user", "content": user_payload}]
            messages_list.append(messages)
        return messages_list

    def simple_translation(
        self,
        src_list: list[str],
        source_lang: str = "Korean",
        target_lang: str = "English",
        multiple_path: int | None = None,
        *,
        lora_request=None,
        **_unused,
    ):
        messages_list = self.create_simple_translation_messages_list(
            src_list,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        prompts = self.apply_chat_template(messages_list)

        if multiple_path is not None and multiple_path > 1:
            if self.generation_config.sampling_params == "greedy":
                return self._generate_multi_path_outputs(
                    prompts,
                    multiple_path,
                    lora_request=lora_request,
                )
            return self._generate_beam_outputs(
                prompts,
                multiple_path,
                lora_request=lora_request,
            )
        return self._generate_output(prompts, lora_request=lora_request)

    def fewshot_singleturn_translation(
        self,
        src_list: list[str],
        fewshot_list: list[list[dict[str, str]]],
        source_lang: str = "Korean",
        target_lang: str = "English",
        multiple_path: int | None = None,
        *,
        lora_request=None,
        **_unused,
    ):
        messages_list: list[list[dict[str, str]]] = []
        for src_sent, fewshot in zip(src_list, fewshot_list):
            system_prompt = "You are a professional translator."
            user_prompt = (
                f"I will give you one or more examples of text fragments, where the first one is in "
                f"{source_lang} and the second one is the translation of the first fragment into "
                f"{target_lang}. These sentences will be displayed below."
            )
            for idx, demo in enumerate(fewshot):
                user_prompt += (
                    f"\n{idx + 1}. {source_lang} text: {demo['src']}\n"
                    f"{target_lang} translation: {demo['mt']}"
                )
            user_prompt += (
                f"\nAfter the example pairs, I will provide a/an {source_lang} sentence and I would like "
                f"you to translate it into {target_lang}. Please provide only the translation result "
                f"without any additional comments, formatting, or chat content. Translate the text from "
                f"{source_lang} to {target_lang}."
            )
            user_prompt += f"\nTranslate the following sentence: {src_sent}"

            if self.model_config.requires_system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            else:
                messages = [{"role": "user", "content": system_prompt + user_prompt}]
            messages_list.append(messages)

        prompts = self.apply_chat_template(messages_list)
        if multiple_path is not None and multiple_path > 1:
            if self.generation_config.sampling_params == "greedy":
                return self._generate_multi_path_outputs(
                    prompts,
                    multiple_path,
                    lora_request=lora_request,
                )
            return self._generate_beam_outputs(
                prompts,
                multiple_path,
                lora_request=lora_request,
            )
        return self._generate_output(prompts, lora_request=lora_request)
