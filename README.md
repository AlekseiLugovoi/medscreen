# 🩺 medscreen
**MedScreen** — это сервис для анализа компьютерных томографий органов грудной клетки с целью выявления **исследований без патологий**.  
Решение помогает снизить нагрузку на врачей-радиологов, автоматически отбирая нормальные КТ, чтобы врачи могли сосредоточиться на случаях с возможными аномалиями.  
В основе — мультимодальная LLM **MedGemma-4b-it**, адаптированная для медицинских изображений.


## 🚀 Quick Start


### Online Demo
Попробовать сервис можно здесь: 👉 [Запустить онлайн](https://d75658572430f4f78b2972d1c74f592ca.clg07azjl.paperspacegradient.com/)

> Онлайн доступна только веб-версия сервиса \
Для использования api - необходима локальная установка

### Local Setup

<details>
<summary><strong>⚠️ Предварительная настройка NVIDIA Container Toolkit (только для GPU)</strong></summary>

Если планируете использовать GPU, установите **NVIDIA Container Toolkit**:

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Проверка
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Для CPU:** пропустите этот шаг и закомментируйте `runtime: nvidia` в `docker-compose.yml`

</details>

1.  **Клонируйте репозиторий:**
    ```sh
    git clone https://github.com/AlekseiLugovoi/medscreen.git
    cd medscreen
    ```

2.  **Создайте файл .env:**
    ```sh
    # Замените ... вашим токеном от Hugging Face
    # или воспользуйтесь нашим: hf_ nNvENhqrQTVFgRxSGUUpYNudvpEgQaNOWZ
    echo "HF_TOKEN=hf_..." > .env
    ```

3.  **Запустите приложение (выберите один из вариантов):**

    > **⚠️ Важно:** Для запуска с GPU (режим по умолчанию) убедитесь, что вы выполнили предварительную настройку **NVIDIA Container Toolkit**, описанную выше. Если у вас нет GPU, закомментируйте строку `runtime: nvidia` в файле `docker-compose.yml` для запуска на CPU.

    <details>
    <summary><strong>Запуск через Docker (рекомендуется)</strong></summary>
    
    > **Примечание:** Веса модели скачиваются при первом запуске контейнера, что может занять до 5 минут в зависимости от скорости интернета. Последующие запуски будут мгновенными, так как модель кэшируется внутри контейнера. Для безопасной передачи Hugging Face токена используется `DOCKER_BUILDKIT=1`.

    <details>
    <summary>🌐 Только веб-интерфейс (по умолчанию)</summary>
    
    Запускает автономное веб-приложение с локальной моделью.
    ```sh
    # Эта команда запустит сервис с профилем "default"
    DOCKER_BUILDKIT=1 docker compose --profile default up --build
    ```
    **Доступ:** `http://localhost:8501`
    </details>

    <details>
    <summary>⚙️ Только REST API</summary>
    
    Запускает только API-сервис для автоматизации и пакетной обработки.
    ```sh
    # Явно указываем профиль "api"
    DOCKER_BUILDKIT=1 docker compose --profile api up --build
    ```
    **Доступ:**
    - **API:** `http://localhost:8502`
    - **Документация (Swagger):** `http://localhost:8502/docs`

    **Пример запроса:**
    ```bash
    # Отправка двух архивов для анализа
    curl -X POST "http://localhost:8502/process" \
         -H "Content-Type: multipart/form-data" \
         -F "files=@/путь/к/study1.zip" \
         -F "files=@/путь/к/study2.zip"
    ```
    
    **Пример ответа (JSON):**
    ```json
    {
      "results": [
        {
          "archive_name": "study1.zip",
          "series_uid": "1.2.840.113704.1.111.4980...",
          "source_format": "DICOM Series",
          "modality": "CT",
          "body_part": "CHEST",
          "orientation": "Axial",
          "num_frames": 120,
          "is_valid": true,
          "has_any_pathology": false,
          "pneumonia": false,
          "lung_cancer": false,
          "aortic_dilation": false,
          "ml_processing_time": "5.12s"
        },
        {
          "archive_name": "study2.zip",
          "series_uid": "1.3.6.1.4.1.14519.5.2.1...",
          "source_format": "DICOM Series",
          "modality": "CT",
          "body_part": "CHEST",
          "orientation": "Axial",
          "num_frames": 95,
          "is_valid": true,
          "has_any_pathology": true,
          "pneumonia": true,
          "lung_cancer": false,
          "aortic_dilation": false,
          "ml_processing_time": "4.31s"
        }
      ]
    }
    ```
    </details>


    <details>
    <summary>🔗 Оба сервиса вместе</summary>
    
    Запускает и веб-интерфейс, и REST API.
    ```sh
    # Указываем оба профиля
    DOCKER_BUILDKIT=1 docker compose --profile default --profile api up --build
    ```
    **Доступ:**
    - **Веб-интерфейс:** `http://localhost:8501`
    - **REST API:** `http://localhost:8502`
    </details>

    </details>


    <details>
    <summary><strong>Локальная разработка (Conda)</strong></summary>

    > **Примечание:** Веса модели будут скачаны при первом запуске приложения (может занять 3-5 минут).

    ```sh
    # Создание окружения
    conda create -n medscreen python=3.11 --yes
    conda activate medscreen
    pip install -r requirements.txt

    # Запуск веб-интерфейса
    cd medscreen
    PYTHONPATH=$PWD streamlit run app/main.py --server.port 8501

    # Или запуск API (в отдельном терминале)
    PYTHONPATH=$PWD uvicorn app.api:app --host 0.0.0.0 --port 8502
    ```
    </details>

4.  **Протестируйте сервис:**
    Используйте [**демо-данные**](https://disk.yandex.ru/d/2ddI6aLMkoIYrA) для проверки работоспособности.

## 💻 Системные требования

<details>
<summary><strong>Минимальные требования</strong></summary>

- **GPU:** NVIDIA с поддержкой CUDA (минимум 8GB VRAM) *или CPU (медленнее в 10-50 раз)*
- **RAM:** 16GB системной памяти
- **Диск:** 20GB свободного места (для модели и зависимостей)
- **ОС:** Linux с поддержкой Docker

</details>

<details>
<summary><strong>Рекомендуемые требования</strong></summary>

- **GPU:** NVIDIA RTX 3080/4080 или выше (12GB+ VRAM)
- **RAM:** 32GB системной памяти
- **CPU:** 8+ ядер

</details>

## 🗂️ Поддерживаемые форматы
Сервис принимает на вход **ZIP-архив**, содержащий одно исследование в одном из следующих форматов:
- **Серия DICOM:** множество файлов (часто с расширением `.dcm` или без него).
- **Многокадровый DICOM:** один `.dcm` файл, содержащий все срезы.
- **NIfTI:** один файл (`.nii` или `.nii.gz`).
- **Серия изображений:** множество файлов (`.png`, `.jpg`).

## 📂 Архитектура

```
medscreen/
├── app/                    # Код приложения
│   ├── main.py            # Streamlit интерфейс  
│   ├── api.py             # FastAPI сервис
│   ├── ml_inference.py    # ML-модель (MedGemma)
│   └── ...                # Другие модули
├── Dockerfile             # Единый образ для обоих сервисов
├── docker-compose.yml     # Конфигурация запуска
└── requirements.txt       # Зависимости
```
**Принцип:** Один Docker-образ, два независимых сервиса с разными командами запуска.

## 🔗 Ссылки
- **Презентация Проекта:** [https://disk.yandex.ru/d/LpKu44Kq0Xa_0w](https://disk.yandex.ru/d/LpKu44Kq0Xa_0w)
- **Онлайн-сервис:** [https://d75658572430f4f78b2972d1c74f592ca.clg07azjl.paperspacegradient.com](https://d75658572430f4f78b2972d1c74f592ca.clg07azjl.paperspacegradient.com)
