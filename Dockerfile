# syntax=docker/dockerfile:1

# Сборка зависимостей и скачивание модели
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Принимаем токен как аргумент сборки (для Paperspace) или через secret (для локальной сборки)
ARG HF_TOKEN

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Установка torch и torchvision
RUN pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
# Установка остальных зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Скачивание весов модели
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache

# Попробуем использовать secret (для локальной сборки), если не получится - используем ARG (для Paperspace)
RUN --mount=type=secret,id=hf_token \
    if [ -f /run/secrets/hf_token ]; then \
        export HF_TOKEN=$(cat /run/secrets/hf_token); \
    fi; \
    if [ -z "$HF_TOKEN" ]; then \
        echo "Error: HF_TOKEN is not set. Pass it as --build-arg or via secret mount" && exit 1; \
    fi; \
    python3.11 -c "\
from transformers import AutoModel, AutoProcessor; \
import os; \
model_name = 'google/medgemma-4b-it'; \
print(f'Downloading {model_name}...'); \
AutoModel.from_pretrained(model_name, token=os.environ['HF_TOKEN'], cache_dir='/app/huggingface_cache'); \
AutoProcessor.from_pretrained(model_name, token=os.environ['HF_TOKEN'], cache_dir='/app/huggingface_cache'); \
print('Model downloaded successfully!')"

# Финальный образ
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем готовое окружение
COPY --from=builder /opt/venv /opt/venv
# Копируем кэш с весами модели
COPY --from=builder /app/huggingface_cache /app/huggingface_cache

# Копируем код приложения
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