import streamlit as st
import requests
import os

# --- НОВЫЙ БЛОК: Клиент для API ---
API_URL = os.getenv("API_URL", "http://medscreen-api:8502")

@st.cache_data(show_spinner="Анализ срезов моделью...")
def run_inference_via_api(file_content: bytes, filename: str) -> dict:
    """Отправляет файл в API и возвращает результат."""
    files = {'file': (filename, file_content, 'application/zip')}
    try:
        response = requests.post(f"{API_URL}/api/v1/process_single", files=files, timeout=300)
        response.raise_for_status()
        # Возвращаем только результаты инференса
        return response.json().get("inference_results", {})
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения с API: {e}")
        return {}

# --- СТАРЫЙ КОД (БОЛЬШЕ НЕ НУЖЕН В STREAMLIT) ---
# @st.cache_resource
# def get_model() -> PathologyClassifier:
#     """Загружает и кэширует ML-модель."""
#     return PathologyClassifier()

# @st.cache_data(show_spinner="Анализ срезов моделью...")
# def run_pathology_inference(_model: PathologyClassifier, volume_3d: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
#     """
#     Кэшируемая обертка для запуска инференса модели.
#     Возвращает словарь с результатами.
#     """
#     return _model.run_inference(volume_3d, threshold=threshold)