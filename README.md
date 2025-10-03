# 🩺 medscreen
**MedScreen** — это сервис для анализа компьютерных томографий органов грудной клетки с целью выявления **исследований без патологий**.  
Решение помогает снизить нагрузку на врачей-радиологов, автоматически отбирая нормальные КТ, чтобы врачи могли сосредоточиться на случаях с возможными аномалиями.  
В основе — мультимодальная LLM **MedGemma-4b-it**, адаптированная для медицинских изображений.


## 🚀 Quick Start

### Online Demo
Попробовать сервис можно здесь:  
👉 [Запустить онлайн](https://d75658572430f4f78b2972d1c74f592ca.clg07azjl.paperspacegradient.com)  

### Local Setup

1.  **Клонируйте репозиторий:**
    ```sh
    git clone https://github.com/AlekseiLugovoi/medscreen.git
    cd medscreen
    ```

2.  **Запустите приложение (выберите один из способов):**

    <details>
        <summary>Запуск через Docker (рекомендуется)</summary>

    ```sh
    # Создайте файл .env рядом с docker-compose.yml 
    # (или воспользуйтесь нашим: hf _nNvENhqrQTVFgRxSGUUpYNudvpEgQaNOWZ для работы моделей с hf)
    echo "HF_TOKEN=ваш_huggingface_token" > .env
    ```

    ```sh
    # Сборка и запуск контейнера
    docker compose up --build
    ```
    Команда запустит два сервиса:
    - **Веб-интерфейс:** `http://localhost:8501`
    - **REST API:** `http://localhost:8502`

    </details>

    <details>
        <summary>Запуск через Conda</summary>

    ```sh
    # Создание и активация окружения
    conda create -n medscreen python=3.11 --yes
    conda activate medscreen

    # Установка зависимостей
    pip install -r requirements.txt
    ```
    
    Для локального запуска вам потребуется два терминала:

    **Терминал 1: Запуск веб-интерфейса**
    ```sh
    streamlit run app/main.py --server.port 8501
    ```

    **Терминал 2: Запуск REST API**
    ```sh
    uvicorn api:app --port 8502
    ```

    - **Веб-интерфейс** будет доступен по адресу: `http://localhost:8501`
    - **REST API** будет доступен по адресу: `http://localhost:8502`

    </details>

3.  **Протестируйте сервис:**
    Для проверки работоспособности используйте [**демо-данные**](https://disk.yandex.ru/d/2ddI6aLMkoIYrA). \
    Вы можете взаимодействовать с сервисом двумя способами:

    <details>
    <summary>Вариант 1: Через веб-интерфейс</summary>

    Откройте `http://localhost:8501` в браузере. Интерфейс позволяет:
    - **Интерактивно анализировать** одно исследование в режиме "Превью".
    - **Обрабатывать несколько исследований** и скачивать CSV-отчет в режиме "Пакетная обработка".
    
    </details>

    <details>
    <summary>Вариант 2: Через REST API</summary>

    API предназначен для автоматизации пакетной обработки.
    - **Интерактивная документация (Swagger):** `http://localhost:8502/docs`

    **Пример запроса:**
    Отправьте один или несколько ZIP-архивов на эндпоинт `/api/v1/upload`. Сервис вернет CSV-файл с результатами.

    ```bash
    # Пример отправки двух архивов и сохранения результата в report.csv
    curl -X POST "http://localhost:8502/api/v1/upload" \
         -H "Content-Type: multipart/form-data" \
         -F "files=@/путь/к/вашему/study1.zip" \
         -F "files=@/путь/к/вашему/study2.zip" \
         --output report.csv
    ```
    </details>


## 📥 Поддерживаемые форматы

Демо данные для тестирования сервиса: [link](https://disk.yandex.ru/d/2ddI6aLMkoIYrA) \
Приложение спроектировано для работы с одним исследованием за раз и поддерживает следующие форматы данных:

</details>

<details>
    <summary>Серия DICOM-файлов в ZIP-архиве</summary>

*   **Описание:** Стандартный клинический случай, когда каждый срез представлен отдельным `.dcm` файлом. Все файлы исследования должны быть упакованы в один `.zip` архив.
*   **Структура:**
    ```
    исследование.zip
    ├── slice-001.dcm
    ├── slice-002.dcm
    └── ...
    ```

</details>

<details>
    <summary>Многокадровый (Multi-frame) DICOM</summary>

*   **Описание:** Редкий случай, когда все срезы исследования содержатся в одном `.dcm` файле.
*   **Структура:**
    ```
    исследование.dcm
    ```

</details>

<details>
    <summary>NIfTI (Neuroimaging Informatics Technology Initiative)</summary>

*   **Описание:** Популярный формат в научных исследованиях. Приложение принимает как сжатые (`.nii.gz`), так и несжатые (`.nii`) файлы.
*   **Структура:**
    ```
    исследование.nii.gz
    ```
    *или*
    ```
    исследование.nii
    ```
</details>


## 📂 Структура проекта

```
.
├── app/                 # Исходный код Streamlit-приложения
├── dev/                 # Ноутбуки и файлы для разработки
├── .env                 # Файл с секретами (HF_TOKEN)
├── api.py               # Исходный код FastAPI
├── docker-compose.yml   # Конфигурация Docker Compose
├── Dockerfile           # Конфигурация Docker-образа
└── requirements.txt     # Зависимости Python
```

## 🔗 Ссылки

- **Презентация Проекта:** https://disk.yandex.ru/d/LpKu44Kq0Xa_0w
- **Онлайн сервис:** https://d75658572430f4f78b2972d1c74f592ca.clg07azjl.paperspacegradient.com
