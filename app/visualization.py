# --- Модуль для подготовки изображений к отображению ---
#
# Флоу:
# 1. Функции получают "сырые" 3D-данные (numpy array).
# 2. Применяют необходимые преобразования:
#    - `apply_ct_window` для КТ-окон.
#    - `normalize_to_uint8` для приведения к 8-битному формату (0-255).
# 3. `create_gif` создает анимацию из готовых 8-битных кадров.

import io
import numpy as np
import imageio
import streamlit as st

def apply_ct_window(volume: np.ndarray, center: int, width: int) -> np.ndarray:
    """Применяет оконное преобразование (windowing) к КТ-объему."""
    lo, hi = center - width / 2, center + width / 2
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo + 1e-6)
    return volume

def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    """Конвертирует нормализованный массив (0-1) в 8-битное изображение (0-255)."""
    volume = np.clip(volume, 0, 1)
    return (volume * 255).astype(np.uint8)

@st.cache_data(show_spinner="Создание анимации...")
def create_gif(frames: list, duration_ms: int = 50) -> bytes:
    """Создает GIF-анимацию из списка 8-битных кадров."""
    with io.BytesIO() as buffer:
        imageio.mimsave(buffer, frames, format='GIF', duration=duration_ms, loop=0)
        return buffer.getvalue()

@st.cache_data(show_spinner="Подготовка кадров для просмотра...")
def prepare_frames_for_display(_series_data: dict, window_name: str, ct_windows: dict) -> list:
    """
    Готовит все кадры серии к отображению: применяет окно и конвертирует в uint8.
    Функция обернута в st.cache_data для кэширования результата.
    """
    raw_frames = _series_data["frames"]
    meta = _series_data["meta"]
    
    if meta.get("Modality") == "CT":
        center, width = ct_windows[window_name]
        processed_frames = apply_ct_window(raw_frames, center, width)
    else:
        # Для не-КТ данных просто нормализуем по всему диапазону
        lo, hi = raw_frames.min(), raw_frames.max()
        processed_frames = (raw_frames - lo) / (hi - lo + 1e-6)
        
    uint8_frames = normalize_to_uint8(processed_frames)
    # Возвращаем как список отдельных кадров
    return [frame for frame in uint8_frames]