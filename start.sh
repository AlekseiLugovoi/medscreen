#!/bin/sh

# Записываем в лог попытку запуска
echo "--- Attempting to start Streamlit app ---"

# Запускаем приложение и перенаправляем ВЕСЬ вывод (и ошибки) в лог
streamlit run app/main.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
    
# Если streamlit упадет, выполнится эта часть
echo "--- Streamlit process exited unexpectedly. Keeping container alive for debugging. ---"

# Держим контейнер живым 5 минут, чтобы успеть посмотреть логи
sleep 300