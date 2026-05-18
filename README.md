# MFDS Official Gradio Service

MFDS 번역 Gradio 웹사이트를 다른 서버에서 실행하기 위한 공유용 repo입니다.

이 repo에는 웹사이트 실행에 필요한 코드만 포함되어 있습니다. 실험 결과, 평가 코드, 학습 데이터, 추론 데이터셋, 모델 가중치, FAISS 인덱스, 개인 토큰은 포함하지 않습니다.

## 1. 구조

```text
MFDS_official/
  README.md
  requirements.txt
  .env.example
  check_ocr_env
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
2. Korean -> English LoRA adapter: `SKIML/mfds-vaivgem-ko-en-fewshot-lora`
3. English -> Korean LoRA adapter: `SKIML/mfds-vaivgem-en-ko-fewshot-lora`
4. Few-shot retrieval FAISS 인덱스

LoRA adapter는 각각 private Hugging Face repo에 올라가 있습니다. 실행 계정의 `HF_TOKEN`은 adapter repo와 base model `SKIML/vaivgem-mfds-9b_original_backbone`을 읽을 권한이 있어야 합니다.

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

Python 3.10을 권장합니다. PDF OCR은 서버 로컬에서만 수행하며 PDF 내용은 외부로 전송하지 않습니다.

설치 역할을 분리합니다.

- `conda`: Tesseract, qpdf, Ghostscript, unpaper, poppler 같은 native 실행파일
- `pip -r requirements.txt`: Gradio/번역 앱과 OCRmyPDF Python 패키지
- 직접 다운로드 또는 내부 배포: Tesseract 언어 데이터 `kor`, `eng`, `osd`

새 환경은 다음 순서로 만듭니다.

```bash
conda create -n mfds_official -c conda-forge --override-channels -y \
  python=3.10 \
  tesseract \
  ghostscript \
  qpdf \
  unpaper \
  poppler

conda activate mfds_official
python -m pip install --upgrade pip
pip install -r requirements.txt
```

클러스터에서 PyTorch/vLLM CUDA build가 별도로 정해져 있으면, 관리자 문서에 맞춰 `torch`, `vllm`, `flashinfer-python`을 먼저 설치한 뒤 `pip install -r requirements.txt`를 실행하세요.

Tesseract 언어 데이터를 설치합니다. 인터넷 다운로드가 허용되지 않는 서버에서는 같은 파일을 내부 저장소나 오프라인 패키지로 복사하세요.

```bash
mkdir -p "$CONDA_PREFIX/share/tessdata"

curl -fL -o "$CONDA_PREFIX/share/tessdata/kor.traineddata" \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/kor.traineddata
curl -fL -o "$CONDA_PREFIX/share/tessdata/eng.traineddata" \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
curl -fL -o "$CONDA_PREFIX/share/tessdata/osd.traineddata" \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/osd.traineddata

export TESSDATA_PREFIX="$CONDA_PREFIX/share/tessdata"
```

OCR 설치를 검증합니다.

```bash
./check_ocr_env
```

직접 확인하려면 다음 명령을 사용합니다.

```bash
which python
which ocrmypdf
which tesseract
ocrmypdf --version
tesseract --list-langs
python -c "from PIL import Image; import cryptography, ocrmypdf, pikepdf; print('OCR Python deps OK')"
pip check
```

`tesseract --list-langs`에 `kor`, `eng`, `osd`가 보여야 합니다. `pip check`가 OCR 관련 오류를 출력하면 같은 환경에서 `pip install --force-reinstall ocrmypdf==16.13.0 pikepdf==10.6.0 pillow==12.1.0 cryptography==47.0.0 cffi==2.0.0`를 먼저 시도하세요.

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
HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
HF_HOME=/absolute/path/to/huggingface_cache
GRADIO_TEMP_DIR=/absolute/path/to/MFDS_official/.cache/gradio_tmp
FEWSHOT_BASELINE_MODEL_KO_EN=SKIML/mfds-vaivgem-ko-en-fewshot-lora
FEWSHOT_BASELINE_MODEL_EN_KO=SKIML/mfds-vaivgem-en-ko-fewshot-lora
MFDS_FAISS_DB_ROOT=/absolute/path/to/faiss/dev_with_doc_id
TESSDATA_PREFIX=/absolute/path/to/conda/envs/mfds_official/share/tessdata
MFDS_OCR_LANGUAGES=kor+eng
MFDS_OCR_MODE=force
```

`HF_TOKEN`은 `.env` 안의 빈 칸에 입력합니다. 이 token은 SKIML private model repo에 read 권한이 있어야 합니다.

`launch_fewshot_gradio`와 SLURM 스크립트는 `PYTHON_BIN`이 있는 디렉터리를 자동으로 `PATH` 앞에 붙입니다. 따라서 `ocrmypdf`와 `tesseract`가 같은 conda env에 설치되어 있으면 별도 `PATH` 설정 없이 앱 subprocess에서 찾을 수 있습니다.

`GRADIO_TEMP_DIR`은 파일 업로드 임시 저장소입니다. 기본값은 repo 안의 `.cache/gradio_tmp`이며, 공유 서버의 `/tmp/gradio` 권한 충돌을 피하려면 실행 계정이 쓸 수 있는 로컬 경로로 둡니다.

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

정상적으로 시작되면 로그에 다음과 같은 줄이 나옵니다.

```text
[gradio] node=n04
[gradio] host=0.0.0.0 port=7860
```

여기서 `n04`는 예시입니다. 실제로는 SLURM이 이번 job에 배정한 GPU 노드명입니다. 서버마다 `n03`, `n04`, `gpu01`, `node-a12`처럼 다를 수 있고, job을 다시 실행하면 바뀔 수도 있습니다. 아래 ngrok 명령에서는 로그에 나온 자기 노드명을 `NODE_NAME` 값으로 넣으면 됩니다.

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

SLURM job 로그에서 compute node 이름을 확인합니다. 아래 예시에서 `n04`는 고정값이 아니라, 사용자가 받은 job이 실행 중인 노드명입니다.

```text
[gradio] node=n04
[gradio] host=0.0.0.0 port=7860
```

로그에 `node=n04`라고 나오면 `NODE_NAME="n04"`처럼 적습니다. 다른 이름이 나오면 그 이름을 넣습니다.

```bash
NODE_NAME="n04"
ngrok http "http://${NODE_NAME}:7860"
```

예를 들어 로그가 `[gradio] node=n04`이면 실제 명령은 다음입니다.

```bash
ngrok http http://n04:7860
```

이 명령은 로그인 노드 또는 compute node에 접속 가능한 서버에서 실행합니다. 현재 서버에서 compute node 이름이 접근되지 않으면, Gradio job이 떠 있는 compute node에 접속해서 ngrok을 실행해야 합니다.

성공하면 다음과 같은 URL이 나옵니다.

```text
Forwarding  https://xxxx.ngrok-free.app -> http://n04:7860
```

여기서 `https://xxxx.ngrok-free.app`가 외부 공개 URL입니다.

터미널을 닫아도 유지하려면 background로 실행합니다.

```bash
NODE_NAME="n04"
nohup ngrok http "http://${NODE_NAME}:7860" > ngrok_mfds.log 2>&1 &
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
NODE_NAME="n04"
ngrok http --domain "$NGROK_STATIC_DOMAIN" "http://${NODE_NAME}:7860"
```

## 10. 웹사이트에 연결 (SKIML 문의)

이 단계는 서버 운영자가 직접 처리할 수 없는 단계입니다. Gradio와 ngrok을 정상적으로 실행한 뒤, ngrok이 출력한 `Forwarding` URL을 SKIML 담당자에게 전달하세요.

SKIML 담당자는 Google Sites나 기존 웹사이트의 `/translate_test` 페이지에 아래처럼 iframe을 넣습니다.

```html
<iframe
  src="https://xxxx.ngrok-free.app"
  style="width:100%; height:900px; border:0;"
  allow="clipboard-read; clipboard-write">
</iframe>
```

`src`에는 서버 운영자가 전달한 ngrok `Forwarding` URL을 넣습니다.

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
NEW_NODE_NAME="n04"
ngrok http "http://${NEW_NODE_NAME}:7860"
```

`NEW_NODE_NAME`에는 새 로그에 나온 `[gradio] node=...` 값을 넣습니다. static domain이 없으면 ngrok URL이 바뀌므로, 새 URL을 SKIML 담당자에게 다시 전달해야 합니다.

## 12. 자주 나는 오류

### `HF_TOKEN` 오류

private 모델이나 adapter를 쓰는데 token이 없을 때 발생합니다.

해결:

```bash
nano .env
# HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

### `adapter_config.json not found`

`FEWSHOT_BASELINE_MODEL_KO_EN` 또는 `FEWSHOT_BASELINE_MODEL_EN_KO`가 LoRA adapter repo/id가 아니거나, private repo 접근 권한이 없을 때 발생합니다.

해결:

```bash
grep -n "FEWSHOT_BASELINE_MODEL" .env
grep -n "HF_TOKEN" .env
```

기본값은 `SKIML/mfds-vaivgem-ko-en-fewshot-lora`, `SKIML/mfds-vaivgem-en-ko-fewshot-lora`입니다. 실행 계정의 Hugging Face token이 두 adapter repo와 base model repo를 읽을 수 있어야 합니다.

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

### `ocrmypdf: command not found`

`ocrmypdf`가 앱 실행 환경의 `PATH`에 없습니다.

해결:

```bash
conda activate mfds_official
which ocrmypdf
grep -n "^PYTHON_BIN=" .env
```

`.env`의 `PYTHON_BIN`이 `ocrmypdf`를 설치한 conda env의 Python을 가리켜야 합니다.

### `/tmp/gradio/... Permission denied`

Gradio가 업로드 파일을 `/tmp/gradio`에 임시 저장하려다 권한에 막힌 것입니다. 공유 서버에서 다른 사용자가 먼저 만든 `/tmp/gradio` 디렉터리 때문에 자주 발생합니다.

해결:

```bash
mkdir -p .cache/gradio_tmp
grep -n "^GRADIO_TEMP_DIR=" .env || echo "GRADIO_TEMP_DIR=$(pwd)/.cache/gradio_tmp" >> .env
sed -i "s|^GRADIO_TEMP_DIR=.*|GRADIO_TEMP_DIR=$(pwd)/.cache/gradio_tmp|" .env
```

앱 또는 SLURM job을 재시작하세요.

### `cannot import name 'PdfMatrix' from 'pikepdf'`

구버전 conda `ocrmypdf`와 신버전 `pikepdf`가 섞였을 때 발생합니다. 이 repo는 `ocrmypdf`를 pip requirements에서 설치하는 방식을 기준으로 합니다.

해결:

```bash
conda activate mfds_official
pip uninstall -y ocrmypdf pikepdf
conda remove -y ocrmypdf pikepdf || true
pip install --force-reinstall ocrmypdf==16.13.0 pikepdf==10.6.0
ocrmypdf --version
```

### `libqpdf.so... cannot open shared object file`

`pikepdf`와 native `qpdf` 조합이 깨졌거나 conda env의 library path가 보이지 않을 때 발생합니다.

해결:

```bash
conda activate mfds_official
conda install -c conda-forge --override-channels -y qpdf
pip install --force-reinstall pikepdf==10.6.0
find "$CONDA_PREFIX" -name 'libqpdf.so*'
ocrmypdf --version
```

필요하면 앱 실행 전에 다음을 추가합니다.

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

### `No module named 'cryptography'` 또는 `cannot import name 'Image' from 'PIL'`

OCRmyPDF의 Python 의존성이 일부 빠졌거나 깨진 상태입니다.

해결:

```bash
conda activate mfds_official
pip install --force-reinstall \
  ocrmypdf==16.13.0 \
  pikepdf==10.6.0 \
  pillow==12.1.0 \
  cryptography==47.0.0 \
  cffi==2.0.0
python -c "from PIL import Image; import cryptography, ocrmypdf, pikepdf; print('OCR Python deps OK')"
ocrmypdf --version
```

### Tesseract 언어 데이터 오류

`kor`, `eng`, `osd` 중 하나가 `tesseract --list-langs`에 보이지 않으면 OCR 품질 파이프라인이 시작되지 않습니다.

해결:

```bash
echo "$TESSDATA_PREFIX"
ls "$TESSDATA_PREFIX"/{kor,eng,osd}.traineddata
tesseract --list-langs
```

파일이 없으면 4장의 Tesseract 언어 데이터 설치 단계를 다시 수행하세요.

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
