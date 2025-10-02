# 🩺 medscreen
Сервис для выявления компьютерных томографий органов грудной клетки без патологий

## 🚀 Quick Start

### Online Demo
Попробовать сервис можно здесь:  
👉 [Запустить онлайн](https://d848d1e027dd94c969d950ddf81efe6c9.clg07azjl.paperspacegradient.com)  

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
    # Сборка и запуск контейнера
    docker-compose up --build
    ```
    Сервис будет доступен по адресу `http://localhost:8501`.

    </details>
    <details>
        <summary>Запуск через Conda</summary>

    ```sh
    # Создание и активация окружения
    conda create -n medscreen python=3.11 --yes
    conda activate medscreen

    # Установка зависимостей
    pip install -r requirements.txt

    # Запуск приложения
    streamlit run app/main.py
    ```
    </details>

3.  **Протестируйте сервис:**
    - Для проверки работоспособности используйте [**демо-данные**](https://disk.yandex.ru/d/2ddI6aLMkoIYrA).

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

├── .github/workflows/   # CI/CD для публикации Docker-образа
├── .streamlit/          # Конфигурация Streamlit
├── app/                 # Исходный код Streamlit-приложения
├── dev/                 # Файлы для разработки и исследований
├── Dockerfile           # Инструкции для сборки Docker-образа
├── README.md            # Этот файл
└── requirements.txt     # Зависимости Python
```

## 🔗 Ссылки

- **Презентация Проекта:** https://disk.yandex.ru/d/LpKu44Kq0Xa_0w
