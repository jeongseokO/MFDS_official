# MFDS Official Gradio Service

MFDS 번역 Gradio 웹사이트를 다른 서버에서 실행하기 위한 공유용 repo입니다.

이 repo에는 웹사이트 실행에 필요한 코드만 포함되어 있습니다. 실험 결과, 평가 코드, 학습 데이터, 추론 데이터셋, 모델 가중치, FAISS 인덱스, 개인 토큰은 포함하지 않습니다.

## 1. 구조

```text
MFDS_official/
  README.md
  requirements.txt
  .env.example
  launch_fewshot_gradio
  slurm_fewshot_gradio.sbatch
  gradio_app/
    fewshot_gradio_app.py
    fewshot_app_backend.py
  translation/
    translation_models.py
  utils/
    retriever.py
```

실행 구조는 다음과 같습니다.

```text
SLURM GPU node에서 Gradio 실행
  -> ngrok 또는 Cloudflare Tunnel로 외부 URL 생성
  -> Google Sites / 기관 웹사이트에 iframe으로 연결
```

주의할 점:

- `#SBATCH --time=48:00:00`이면 SLURM job은 최대 48시간 뒤 종료됩니다.
- 공개 URL을 만들면 링크를 아는 사람이 접속할 수 있습니다. 민감한 문서는 넣지 않도록 안내하세요.
- 이 repo는 인증/로그인 기능을 포함하지 않습니다.

## 2. 서버 준비물

필요한 것은 네 가지입니다.

1. GPU가 있는 Linux 서버 또는 SLURM 클러스터
2. Korean -> English LoRA adapter
3. English -> Korean LoRA adapter
4. Few-shot retrieval FAISS 인덱스

LoRA adapter는 각각 `adapter_config.json`을 포함해야 합니다. `adapter_config.json` 안의 `base_model_name_or_path`가 target 서버에서 접근 가능한 base model 경로 또는 Hugging Face repo id를 가리켜야 합니다.

FAISS 인덱스는 `MFDS_FAISS_DB_ROOT` 값 뒤에 방향 suffix가 붙는 구조를 사용합니다.

예시:

```text
MFDS_FAISS_DB_ROOT=/data/shared/mfds/faiss/dev_with_doc_id

필요한 실제 경로:
/data/shared/mfds/faiss/dev_with_doc_id_ko_to_en/
/data/shared/mfds/faiss/dev_with_doc_id_en_to_ko/
```

## 3. Repo 받기

```bash
cd /data/shared
git clone <MFDS_official_REPO_URL> MFDS_official
cd MFDS_official
```

아직 remote repo에 올리기 전이라면, 이 디렉터리를 그대로 서버에 복사해도 됩니다.

## 4. Conda 환경 만들기

Python 3.10을 권장합니다.

```bash
conda create -n mfds_official python=3.10 -y
conda activate mfds_official
python -m pip install --upgrade pip
pip install -r requirements.txt
```

클러스터에서 PyTorch/vLLM CUDA build가 별도로 정해져 있으면, 관리자 문서에 맞춰 `torch`, `vllm`, `flashinfer-python`을 먼저 설치한 뒤 나머지를 설치하세요.

```bash
pip install -r requirements.txt
```

PDF 업로드 기능을 쓰려면 `pdftotext`도 필요할 수 있습니다.

```bash
which pdftotext
```

없으면 서버 관리자에게 `poppler-utils` 설치를 요청하세요.

## 5. 환경 변수 설정

샘플 파일을 복사합니다.

```bash
cp .env.example .env
```

`.env`를 열어서 서버에 맞게 수정합니다.

```bash
nano .env
```

필수 항목:

```bash
PYTHON_BIN=/absolute/path/to/conda/envs/mfds_official/bin/python
HF_HOME=/absolute/path/to/huggingface_cache
FEWSHOT_BASELINE_MODEL_KO_EN=/absolute/path/to/ko_en_lora_adapter
FEWSHOT_BASELINE_MODEL_EN_KO=/absolute/path/to/en_ko_lora_adapter
MFDS_FAISS_DB_ROOT=/absolute/path/to/faiss/dev_with_doc_id
```

모델이나 adapter가 private Hugging Face repo에 있으면 `HF_TOKEN`도 넣습니다.

```bash
HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

`.env`는 절대 git에 올리지 마세요. `.gitignore`에 이미 제외되어 있습니다.

## 6. 로컬 실행 테스트

GPU node에 직접 접속해서 테스트할 수 있는 환경이면 다음을 실행합니다.

```bash
conda activate mfds_official
./launch_fewshot_gradio \
  --host 0.0.0.0 \
  --port 7860 \
  --directions ko_en,en_ko \
  --methods fewshot_baseline,segment_mt
```

로그에 다음과 비슷한 메시지가 나오면 실행 중입니다.

```text
Running on local URL:  http://0.0.0.0:7860
```

서버 내부에서 확인:

```bash
curl http://127.0.0.1:7860
```

## 7. SLURM으로 실행

일반적으로는 SLURM job으로 Gradio를 띄웁니다.

```bash
cd /data/shared/MFDS_official
sbatch slurm_fewshot_gradio.sbatch
```

job id 확인:

```bash
squeue -u "$USER"
```

로그 확인:

```bash
tail -f mfds_gradio_<JOB_ID>.out
tail -f mfds_gradio_error_<JOB_ID>.err
```

기본값은 `#SBATCH --gres=gpu:1`입니다. 이 경우 Korean -> English와 English -> Korean 두 방향이 같은 GPU에서 하나의 base model을 공유하고, 요청마다 LoRA adapter를 바꿔 끼웁니다.

GPU 2개를 쓰고 싶으면 `slurm_fewshot_gradio.sbatch`에서 다음 줄을 바꿉니다.

```bash
#SBATCH --gres=gpu:2
```

실행 옵션을 임시로 바꾸고 싶으면 `sbatch` 앞에 환경 변수를 붙이면 됩니다.

```bash
GPU_MEM_UTIL=0.5 BATCH_SIZE=32 sbatch slurm_fewshot_gradio.sbatch
```

한 방향만 실행:

```bash
APP_DIRECTIONS=ko_en sbatch slurm_fewshot_gradio.sbatch
```

## 8. ngrok 설치

외부에서 접속 가능한 URL을 만들 때 ngrok을 사용할 수 있습니다. root 권한이 없어도 홈 디렉터리에 설치할 수 있습니다.

```bash
mkdir -p "$HOME/.local/bin"
cd "$HOME/.local/bin"
wget -O ngrok.tgz https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok.tgz
rm ngrok.tgz
chmod +x ngrok

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
ngrok version
```

ngrok 계정에서 authtoken을 발급받은 뒤 등록합니다.

```bash
ngrok config add-authtoken '<YOUR_NGROK_AUTHTOKEN>'
```

## 9. ngrok으로 Gradio 공개

SLURM job 로그에서 compute node 이름을 확인합니다.

```text
[gradio] node=n04
[gradio] host=0.0.0.0 port=7860
```

로그인 노드 또는 compute node에 접속 가능한 서버에서 다음을 실행합니다.

```bash
ngrok http http://n04:7860
```

성공하면 다음과 같은 URL이 나옵니다.

```text
Forwarding  https://xxxx.ngrok-free.app -> http://n04:7860
```

여기서 `https://xxxx.ngrok-free.app`가 외부 공개 URL입니다.

터미널을 닫아도 유지하려면 background로 실행합니다.

```bash
nohup ngrok http http://n04:7860 > ngrok_mfds.log 2>&1 &
echo $! > ngrok_mfds.pid
```

URL 확인:

```bash
grep -o 'https://[^ ]*ngrok[^ ]*' ngrok_mfds.log | head
```

중지:

```bash
kill "$(cat ngrok_mfds.pid)"
```

ngrok static domain이 있으면 `.env`에 넣고 다음처럼 실행합니다.

```bash
ngrok http --domain "$NGROK_STATIC_DOMAIN" http://n04:7860
```

## 10. 웹사이트에 연결

Google Sites나 기존 웹사이트에서 `/translate_test` 페이지를 만들고 iframe을 넣습니다.

```html
<iframe
  src="https://xxxx.ngrok-free.app"
  style="width:100%; height:900px; border:0;"
  allow="clipboard-read; clipboard-write">
</iframe>
```

`src`에는 ngrok이 출력한 `Forwarding` URL을 넣습니다.

예시:

```html
<iframe
  src="https://abc123.ngrok-free.app"
  style="width:100%; height:900px; border:0;"
  allow="clipboard-read; clipboard-write">
</iframe>
```

## 11. 재시작 절차

SLURM job은 시간 제한이 끝나면 내려갑니다. 다시 올릴 때는 다음 순서로 진행합니다.

1. 기존 job 확인

```bash
squeue -u "$USER"
```

2. 필요하면 기존 job 종료

```bash
scancel <JOB_ID>
```

3. 새 job 실행

```bash
sbatch slurm_fewshot_gradio.sbatch
```

4. 새 node 이름 확인

```bash
tail -f mfds_gradio_<NEW_JOB_ID>.out
```

5. ngrok 재실행

```bash
ngrok http http://<NEW_NODE>:7860
```

static domain이 없으면 ngrok URL이 바뀌므로 웹사이트 iframe의 `src`도 새 URL로 바꿔야 합니다.

## 12. 자주 나는 오류

### `HF_TOKEN` 오류

private 모델이나 adapter를 쓰는데 token이 없을 때 발생합니다.

해결:

```bash
nano .env
# HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

### `adapter_config.json not found`

`FEWSHOT_BASELINE_MODEL_KO_EN` 또는 `FEWSHOT_BASELINE_MODEL_EN_KO`가 LoRA adapter 디렉터리가 아닙니다.

해결:

```bash
ls "$FEWSHOT_BASELINE_MODEL_KO_EN"
ls "$FEWSHOT_BASELINE_MODEL_EN_KO"
```

각 경로에 `adapter_config.json`이 있어야 합니다.

### `Few-shot retrieval index was not found`

FAISS 인덱스 경로가 잘못되었습니다.

해결:

```bash
echo "$MFDS_FAISS_DB_ROOT"
ls "${MFDS_FAISS_DB_ROOT}_ko_to_en"
ls "${MFDS_FAISS_DB_ROOT}_en_to_ko"
```

### `No available memory for the cache blocks`

GPU 메모리가 부족합니다.

해결 예시:

```bash
GPU_MEM_UTIL=0.5 BATCH_SIZE=16 sbatch slurm_fewshot_gradio.sbatch
```

또는 한 방향만 실행합니다.

```bash
APP_DIRECTIONS=ko_en sbatch slurm_fewshot_gradio.sbatch
```

### ngrok 명령어가 없음

`PATH`가 갱신되지 않았을 수 있습니다.

```bash
export PATH="$HOME/.local/bin:$PATH"
which ngrok
```

### ngrok `authentication failed`

authtoken 등록이 필요합니다.

```bash
ngrok config add-authtoken '<YOUR_NGROK_AUTHTOKEN>'
```

## 13. 공유 전 확인 checklist

공유하기 전에 반드시 확인하세요.

```bash
grep -R "HF_TOKEN=" -n .
grep -R "TOKEN=" -n .
grep -R "/data[0-9]/" -n .
git status --short
```

정상적인 공유 repo라면 실제 토큰 값, 개인 절대 경로, 실험 데이터 파일이 나오면 안 됩니다.
