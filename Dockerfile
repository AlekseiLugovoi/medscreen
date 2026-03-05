# syntax=docker/dockerfile:1

# vllm/vllm-openai already has torch, triton, vllm, cuda — no need to install them
FROM vllm/vllm-openai:v0.8.2

ARG HF_TOKEN

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only lightweight app dependencies (streamlit, fastapi, dicom, etc.)
COPY requirements-app.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements-app.txt

# Copy application
COPY ./app ./app
COPY ./.streamlit ./.streamlit

ENV HF_HOME=/app/huggingface_cache

# Temp directory for vLLM file:// images
RUN mkdir -p /app/tmp

# Entrypoint: download model if needed, then start both services
COPY <<'ENTRY' /app/entrypoint.sh
#!/bin/bash
set -e

# Download model on first run
if [ ! -d "/app/huggingface_cache/hub" ]; then
  echo "Downloading model..."
  python3 -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN')
if token:
    snapshot_download('google/medgemma-1.5-4b-it', token=token, cache_dir='/app/huggingface_cache')
    print('Model downloaded successfully!')
else:
    print('Warning: HF_TOKEN not set')
"
fi

# Start API backend (model lives here)
echo "Starting API backend on port 8502..."
uvicorn app.api:app --host 0.0.0.0 --port 8502 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to start..."
for i in $(seq 1 60); do
  if curl -s http://localhost:8502/health > /dev/null 2>&1; then
    echo "API is ready!"
    break
  fi
  sleep 2
done

# Start Streamlit frontend
echo "Starting Streamlit on port 8501..."
python -m streamlit run app/main.py \
  --server.port=8501 \
  --server.address=0.0.0.0 &
ST_PID=$!

# Wait for either process to exit
wait -n $API_PID $ST_PID
ENTRY

RUN chmod +x /app/entrypoint.sh

EXPOSE 8501 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/health || exit 0

ENTRYPOINT ["/app/entrypoint.sh"]
