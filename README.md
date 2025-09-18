# medscreen
Сервис для выявления компьютерных томографий органов грудной клетки без патологий

## Quick Start

### Clone the Repository

```sh
git clone https://github.com/AlekseiLugovoi/medscreen.git
```

### Local Setup

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