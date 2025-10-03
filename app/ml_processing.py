# --- Модуль для взаимодействия с ML-моделью ---
#
# Флоу:
# 1. `get_model` загружает и кэширует ML-модель при первом запуске.
# 2. `run_pathology_inference` является кэшируемой оберткой,
#    которая принимает 3D-объем и запускает на нем анализ.

import streamlit as st
import numpy as np
from typing import Dict, Any
from ml_inference import PathologyClassifier

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