# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG HF_TOKEN

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache

RUN --mount=type=secret,id=HF_TOKEN \
    set -eu; \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HUGGING_FACE_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    elif [ -n "$HF_TOKEN" ]; then \
        export HUGGING_FACE_TOKEN="$HF_TOKEN"; \
    else \
        echo "Error: Hugging Face token not found." >&2; \
        exit 1; \
    fi; \
    python3.11 - <<'PY'
import os
import sys
import shutil
from transformers import AutoModel, AutoProcessor

token = os.environ.get("HUGGING_FACE_TOKEN", "")
if not token:
    print("Error: HUGGING_FACE_TOKEN not found", file=sys.stderr)
    sys.exit(1)

model_name = "google/medgemma-4b-it"
cache_dir = "/app/huggingface_cache"
print(f"Downloading {model_name}...")

# Загружаем модель
model = AutoModel.from_pretrained(model_name, token=token, cache_dir=cache_dir)
processor = AutoProcessor.from_pretrained(model_name, token=token, cache_dir=cache_dir)

# Сразу удаляем модель из памяти
del model
del processor

print("Model downloaded successfully!")
print("Cleaning up cache...")

# Агрессивная очистка кэша
import glob
import gc

# Принудительная сборка мусора
gc.collect()

# Находим и удаляем дубликаты и временные файлы
cache_path = f"{cache_dir}/hub"
if os.path.exists(cache_path):
    # Удаляем .incomplete файлы
    for incomplete in glob.glob(f"{cache_path}/**/*.incomplete", recursive=True):
        os.remove(incomplete)
        print(f"Removed incomplete: {incomplete}")
    
    # Удаляем blobs (они дублируются в snapshots)
    blobs_path = os.path.join(cache_path, "blobs")
    if os.path.exists(blobs_path):
        shutil.rmtree(blobs_path)
        print("Removed blobs directory")
    
    # Удаляем временные папки
    for tmp_dir in glob.glob(f"{cache_path}/models--*/refs"):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # Удаляем .lock файлы
    for lock_file in glob.glob(f"{cache_path}/**/*.lock", recursive=True):
        os.remove(lock_file)

print("Cache cleanup complete.")

# Показываем размер итогового кэша
total_size = 0
for dirpath, dirnames, filenames in os.walk(cache_dir):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        if os.path.exists(fp):
            total_size += os.path.getsize(fp)
print(f"Final cache size: {total_size / (1024**3):.2f} GB")
PY

# Очистка pip кэша и временных файлов
RUN pip cache purge && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/* && \
    find /opt/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/huggingface_cache /app/huggingface_cache

COPY ./app ./app
COPY ./.streamlit ./app/.streamlit
COPY requirements.txt .

ENV PATH="/opt/venv/bin:$PATH"
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache

EXPOSE 8501
EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]