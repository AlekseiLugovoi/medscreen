# 🩺 medscreen
Сервис для выявления компьютерных томографий органов грудной клетки без патологий

## 🚀 Quick Start

### Online Demo
Попробовать сервис можно здесь:  
👉 [Запустить онлайн](https://d848d1e027dd94c969d950ddf81efe6c9.clg07azjl.paperspacegradient.com)  

### Local Setup

Clone the Repository
```sh
git clone https://github.com/AlekseiLugovoi/medscreen.git
```

<details>
    <summary>Run Conda</summary>

```sh
ENV=medscreen
PY_VERSION=3.11
conda create -n $ENV python=$PY_VERSION --yes

conda activate $ENV
pip install -r requirements.txt
```
```sh
streamlit run app/main.py
```

</details>

<details>
    <summary>Run Docker</summary>

```sh
docker-compose up
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

├── .github/workflows/   # CI/CD для публикации Docker-образа
├── .streamlit/          # Конфигурация Streamlit
├── app/                 # Исходный код Streamlit-приложения
├── dev/                 # Файлы для разработки и исследований
├── Dockerfile           # Инструкции для сборки Docker-образа
├── README.md            # Этот файл
└── requirements.txt     # Зависимости Python
```

## 🔗 Ссылки

- **Веса модели:** [Ссылка на ваш лендинг]
- **Презентация проекта:** [Ссылка на вашу презентацию]
