#!/bin/bash
#SBATCH --job-name=mfds_gradio
#SBATCH --nodes=1
# Request gpu:1 to run both directions on one GPU with shared LoRA.
# Request gpu:2 to split ko_en/en_ko across two GPUs.
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=mfds_gradio_%j.out
#SBATCH --error=mfds_gradio_error_%j.err

set -euo pipefail

REPO_ROOT=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm_fewshot_gradio.sbatch" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
unset TRANSFORMERS_CACHE

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export MFDS_DISABLE_VLLM_FLASHINFER="${MFDS_DISABLE_VLLM_FLASHINFER:-1}"
export MFDS_FAISS_DB_ROOT="${MFDS_FAISS_DB_ROOT:-${REPO_ROOT}/resources/faiss/dev_with_doc_id}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN in .env, for example /path/to/conda/envs/mfds/bin/python" >&2
    exit 1
fi

GRADIO_HOST="${GRADIO_HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
MAX_FEWSHOT="${MAX_FEWSHOT:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PROGRESS_CHUNK_SIZE="${PROGRESS_CHUNK_SIZE:-32}"
AUTO_BATCH_TOKEN_BUDGET="${AUTO_BATCH_TOKEN_BUDGET:-12288}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"
APP_DIRECTIONS="${APP_DIRECTIONS:-ko_en,en_ko}"
APP_METHODS="${APP_METHODS:-fewshot_baseline,segment_mt}"
FEWSHOT_BASELINE_MODEL_KO_EN="${FEWSHOT_BASELINE_MODEL_KO_EN:-SKIML/mfds-vaivgem-ko-en-fewshot-lora}"
FEWSHOT_BASELINE_MODEL_EN_KO="${FEWSHOT_BASELINE_MODEL_EN_KO:-SKIML/mfds-vaivgem-en-ko-fewshot-lora}"
GRADIO_SHARE="${GRADIO_SHARE:-0}"

export FEWSHOT_APP_PROGRESS_CHUNK_SIZE="${FEWSHOT_APP_PROGRESS_CHUNK_SIZE:-${PROGRESS_CHUNK_SIZE}}"
export FEWSHOT_APP_AUTO_BATCH_TOKEN_BUDGET="${FEWSHOT_APP_AUTO_BATCH_TOKEN_BUDGET:-${AUTO_BATCH_TOKEN_BUDGET}}"

if [[ ! -e "${FEWSHOT_BASELINE_MODEL_KO_EN}" && "${FEWSHOT_BASELINE_MODEL_KO_EN}" != */* ]]; then
    echo "FEWSHOT_BASELINE_MODEL_KO_EN does not look like a path or Hugging Face repo id: ${FEWSHOT_BASELINE_MODEL_KO_EN}" >&2
    exit 1
fi
if [[ ! -e "${FEWSHOT_BASELINE_MODEL_EN_KO}" && "${FEWSHOT_BASELINE_MODEL_EN_KO}" != */* ]]; then
    echo "FEWSHOT_BASELINE_MODEL_EN_KO does not look like a path or Hugging Face repo id: ${FEWSHOT_BASELINE_MODEL_EN_KO}" >&2
    exit 1
fi

USE_KO_EN=0
USE_EN_KO=0
IFS=',' read -r -a APP_DIRECTION_LIST <<< "${APP_DIRECTIONS}"
for direction in "${APP_DIRECTION_LIST[@]}"; do
    case "${direction}" in
        ko_en) USE_KO_EN=1 ;;
        en_ko) USE_EN_KO=1 ;;
        "") ;;
        *)
            echo "Unsupported APP_DIRECTIONS entry: ${direction}" >&2
            echo "Supported values: ko_en,en_ko" >&2
            exit 1
            ;;
    esac
done
if [[ "${USE_KO_EN}" == "0" && "${USE_EN_KO}" == "0" ]]; then
    echo "APP_DIRECTIONS must include at least one of: ko_en, en_ko" >&2
    exit 1
fi

RAW_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${RAW_VISIBLE_DEVICES}" && -n "${SLURM_JOB_GPUS:-}" ]]; then
    RAW_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
fi

CUDA_DEVICE_LIST=()
if [[ -n "${RAW_VISIBLE_DEVICES}" ]]; then
    IFS=',' read -r -a CUDA_DEVICE_LIST <<< "${RAW_VISIBLE_DEVICES}"
fi

if [[ -z "${KO_EN_GPU:-}" ]]; then
    if [[ "${#CUDA_DEVICE_LIST[@]}" -ge 1 && -n "${CUDA_DEVICE_LIST[0]}" ]]; then
        KO_EN_GPU="${CUDA_DEVICE_LIST[0]}"
    else
        KO_EN_GPU="0"
    fi
fi

if [[ -z "${EN_KO_GPU:-}" ]]; then
    if [[ "${#CUDA_DEVICE_LIST[@]}" -ge 2 && -n "${CUDA_DEVICE_LIST[1]}" ]]; then
        EN_KO_GPU="${CUDA_DEVICE_LIST[1]}"
    elif [[ "${#CUDA_DEVICE_LIST[@]}" -ge 1 && -n "${CUDA_DEVICE_LIST[0]}" ]]; then
        EN_KO_GPU="${CUDA_DEVICE_LIST[0]}"
    else
        EN_KO_GPU="0"
    fi
fi

ACTIVE_DIRECTION_COUNT=$((USE_KO_EN + USE_EN_KO))
GPU_MODE="dual_gpu"
if [[ "${ACTIVE_DIRECTION_COUNT}" -eq 1 ]]; then
    GPU_MODE="single_direction"
elif [[ "${KO_EN_GPU}" == "${EN_KO_GPU}" ]]; then
    GPU_MODE="single_gpu_shared"
fi

SHARED_LORA_VALUE="${MFDS_GRADIO_SHARED_LORA:-1}"
SHARED_LORA_LOWER="${SHARED_LORA_VALUE,,}"
SHARED_LORA_ENABLED=1
case "${SHARED_LORA_LOWER}" in
    0|false|no|off) SHARED_LORA_ENABLED=0 ;;
esac

if [[ -z "${GPU_MEM_UTIL}" ]]; then
    if [[ "${GPU_MODE}" == "single_gpu_shared" && "${SHARED_LORA_ENABLED}" == "0" ]]; then
        GPU_MEM_UTIL="0.3"
    else
        GPU_MEM_UTIL="0.6"
    fi
fi

LAUNCH_ARGS=(
    --host "${GRADIO_HOST}"
    --port "${PORT}"
    --directions "${APP_DIRECTIONS}"
    --methods "${APP_METHODS}"
    --ko-en-model "${FEWSHOT_BASELINE_MODEL_KO_EN}"
    --en-ko-model "${FEWSHOT_BASELINE_MODEL_EN_KO}"
    --ko-en-gpu "${KO_EN_GPU}"
    --en-ko-gpu "${EN_KO_GPU}"
    --db-root "${MFDS_FAISS_DB_ROOT}"
    --gpu-mem-util "${GPU_MEM_UTIL}"
    --batch-size "${BATCH_SIZE}"
    --max-fewshot "${MAX_FEWSHOT}"
    --startup-timeout "${STARTUP_TIMEOUT}"
    --request-timeout "${REQUEST_TIMEOUT}"
)

if [[ -n "${TOKENIZER_MODEL}" ]]; then
    LAUNCH_ARGS+=(--tokenizer-model "${TOKENIZER_MODEL}")
fi

if [[ "${GRADIO_SHARE}" == "1" ]]; then
    LAUNCH_ARGS+=(--share)
fi

NODE_HOST="$(hostname -f 2>/dev/null || hostname)"
echo "[gradio] repo_root=${REPO_ROOT}"
echo "[gradio] job_id=${SLURM_JOB_ID:-N/A}"
echo "[gradio] node=${NODE_HOST}"
echo "[gradio] host=${GRADIO_HOST} port=${PORT}"
echo "[gradio] directions=${APP_DIRECTIONS}"
echo "[gradio] methods=${APP_METHODS}"
echo "[gradio] gpu_mode=${GPU_MODE} gpu_mem_util=${GPU_MEM_UTIL}"
echo "[gradio] shared_lora=${SHARED_LORA_VALUE}"
echo "[gradio] ko_en_gpu=${KO_EN_GPU} en_ko_gpu=${EN_KO_GPU}"
echo "[gradio] ko_en_model=${FEWSHOT_BASELINE_MODEL_KO_EN}"
echo "[gradio] en_ko_model=${FEWSHOT_BASELINE_MODEL_EN_KO}"
echo "[gradio] faiss_db_root=${MFDS_FAISS_DB_ROOT}"
echo "[gradio] hf_home=${HF_HOME}"
echo "[gradio] launcher=${REPO_ROOT}/launch_fewshot_gradio"
echo "[gradio] access_hint=ssh -N -L ${PORT}:${NODE_HOST}:${PORT} <login-node>"

if [[ "${GPU_MODE}" == "single_gpu_shared" && "${SHARED_LORA_ENABLED}" == "1" ]]; then
    echo "[gradio] model_preload=shared_base_model_with_lora_switching"
else
    echo "[gradio] model_preload=eager_per_direction"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gradio] nvidia_smi_begin"
    nvidia-smi -L || true
    echo "[gradio] nvidia_smi_end"
fi

exec bash "${REPO_ROOT}/launch_fewshot_gradio" "${LAUNCH_ARGS[@]}"
