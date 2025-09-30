# --- Этап 1: Сборка зависимостей ---
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# --- Этап 2: Финальный образ ---
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем готовое окружение и код приложения
COPY --from=builder /opt/venv /opt/venv
COPY app/ ./app/
# Streamlit ищет конфиг в домашней директории пользователя. Для root это /root/.
COPY .streamlit/ /root/.streamlit/

# Указываем путь к venv и запускаем приложение
ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]