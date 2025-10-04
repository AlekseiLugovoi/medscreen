# syntax=docker/dockerfile:1

# --- Этап 1: Сборка зависимостей ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Установка torch и torchvision
RUN pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Установка остальных зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# --- Предварительное скачивание модели ---
# ЭТОТ БЛОК НУЖНО ПОЛНОСТЬЮ УДАЛИТЬ
# RUN --mount=type=secret,id=HF_TOKEN,dst=/run/secrets/hf_token \
#     export HF_TOKEN=$(cat /run/secrets/hf_token) && \
#     python -c "from transformers import pipeline; pipeline('image-text-to-text', model='google/medgemma-4b-it', model_kwargs={'torch_dtype': 'bfloat16'})"

# --- Этап 2: Финальный образ ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 curl && rm -rf /var/lib/apt/lists/*

# --- ИСПРАВЛЕНИЕ: Основная рабочая директория ---
WORKDIR /app

# Копируем готовое окружение
COPY --from=builder /opt/venv /opt/venv

# Копируем код приложения
COPY ./app ./app
COPY ./.streamlit ./app/.streamlit
COPY requirements.txt .

# Указываем путь к venv
ENV PATH="/opt/venv/bin:$PATH"

# Открываем порты
EXPOSE 8501
EXPOSE 8502

# Проверка состояния для Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# --- ИСПРАВЛЕНИЕ: Команда по умолчанию для Streamlit ---
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]