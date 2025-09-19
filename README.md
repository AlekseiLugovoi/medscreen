# 🩺 medscreen
Сервис для выявления компьютерных томографий органов грудной клетки без патологий

## 🚀 Quick Start

### Online Demo
Попробовать сервис можно здесь:  
👉 [[TODO] Запустить онлайн](https://d848d1e027dd94c969d950ddf81efe6c9.clg07azjl.paperspacegradient.com)  

### Local Setup

- Clone the Repository
    ```sh
    git clone https://github.com/AlekseiLugovoi/medscreen.git
    ```

- Run Conda
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

- or Docker
    ```sh
    docker build -t medscreen .
    docker run -p 8501:8501 medscreen
    ```

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
