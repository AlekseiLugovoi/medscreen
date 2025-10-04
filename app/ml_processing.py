import streamlit as st
from app.ml_inference import PathologyClassifier
from typing import Dict, Any
import numpy as np

@st.cache_resource
def get_model() -> PathologyClassifier:
    """Загружает и кэширует ML-модель."""
    return PathologyClassifier()

@st.cache_data(show_spinner="Анализ срезов моделью...")
def run_pathology_inference(_model: PathologyClassifier, volume_3d: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
    """
    Кэшируемая обертка для запуска инференса модели.
    Возвращает словарь с результатами.
    """
    return _model.run_inference(volume_3d, threshold=threshold)