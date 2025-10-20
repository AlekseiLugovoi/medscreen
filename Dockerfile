# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG HF_TOKEN

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Копируем приложение
COPY ./app ./app
COPY ./.streamlit ./app/.streamlit

ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache

# Создаем entrypoint скрипт правильно
COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
set -e

if [ ! -d "/app/huggingface_cache/hub" ]; then
  echo "Downloading model..."
  python3.11 -c '
from transformers import AutoModel, AutoProcessor
import os

token = os.environ.get("HF_TOKEN")
if token:
    print(f"Using token: {token[:10]}...")
    AutoModel.from_pretrained("google/medgemma-4b-it", token=token, cache_dir="/app/huggingface_cache")
    AutoProcessor.from_pretrained("google/medgemma-4b-it", token=token, cache_dir="/app/huggingface_cache")
    print("Model downloaded successfully!")
else:
    print("Warning: HF_TOKEN not set, model will be downloaded on first inference")
'
fi

exec "$@"
EOF

RUN chmod +x /app/entrypoint.sh

EXPOSE 8501 8502

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 0

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]