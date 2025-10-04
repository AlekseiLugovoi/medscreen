# ---
# Модуль для классификации медицинских изображений с использованием MedGemma.
#
# Основной метод:
#   run_inference(volume_3d: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]
#
# Вход:
#   - volume_3d: 3D-массив numpy (срезы, высота, ширина).
#   - threshold: Порог для бинаrizации вероятностей (0.0-1.0).
#
# Выход (словарь):
#   - "preds": list[bool] - Бинарные предсказания для каждого среза.
#   - "raw_probs": list[float] - "Сырые" вероятности патологии для каждого среза.
#   - "processing_time": float - Общее время обработки в секундах.
# ---

import logging
import time
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from typing import Dict, List, Any
import re

# --- ЛОГИРОВАНИЕ ---
model_logger = logging.getLogger('model_logger')
model_logger.setLevel(logging.INFO)
model_logger.propagate = False
if not model_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [MODEL] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    model_logger.addHandler(handler)

def get_gpu_memory_usage_str() -> str:
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return f"GPU Mem: {allocated:.1f}MB (Peak: {peak:.1f}MB)"

# --- Вспомогательные функции для выборки срезов ---
def select_step(n_slices: int) -> int:
    if n_slices < 50:   return 1
    if n_slices < 100:  return 2
    if n_slices < 200:  return 4
    if n_slices < 400:  return 6
    if n_slices < 600:  return 8
    return 10

def quartile_sample_indices(n_files: int, n: int) -> list[int]:
    if n == 0: n = 1
    q1, q2, q3 = n_files // 4, n_files // 2, (3 * n_files) // 4
    idx = set()
    idx.update(range(0, q1, n))
    idx.update(range(q1, q2, max(1, n//2)))
    idx.update(range(q2, q3, max(1, n//2)))
    idx.update(range(q3, n_files, n))
    if n_files > 0:
        idx.add(n_files - 1)
    return sorted(list(idx))

class PathologyClassifier:
    def __init__(self, model_name: str = "google/medgemma-4b-it"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        start_time = time.time()
        model_logger.info(f"Инициализация модели '{model_name}' на устройстве: {self.device}")

        try:
            self.pipe = pipeline(
                "image-text-to-text",
                model=model_name,
                model_kwargs={"torch_dtype": self.torch_dtype},
                device=self.device
            )
            init_time = time.time() - start_time
            model_logger.info(f"Модель инициализирована за {init_time:.1f}с")
        except Exception as e:
            model_logger.error(f"Ошибка инициализации модели: {e}")
            raise

        # --- ИЗМЕНЕНИЕ: Промпт от коллеги ---
        self.user_prompt = """Task: classify chest CT scan for pulmonary abnormalities.
Steps:
1) Examine both lungs for opacities, consolidations, ground-glass changes, pleural effusion, pneumothorax, fibrosis, or nodules.
2) If no abnormalities are visible, output label: normal.
3) If any abnormality is suspected, output label: anomaly.
4) Output format:
   - label: normal OR label: anomaly"""
        self.system_prompt = "You are an expert radiologist."

    def _prepare_slice(self, slice_2d: np.ndarray) -> Image.Image:
        """Конвертирует 2D срез в PIL Image."""
        # Применяем легочное окно, как в скрипте коллеги
        center, width = -600, 1500
        lo, hi = center - width / 2, center + width / 2
        slice_2d = np.clip(slice_2d, lo, hi)
        slice_2d = (slice_2d - lo) / (hi - lo + 1e-6)
        
        # Нормализуем в uint8
        slice_2d = np.clip(slice_2d, 0, 1)
        img_array = (slice_2d * 255).astype(np.uint8)
        return Image.fromarray(img_array).convert("L")

    @torch.inference_mode()
    def run_inference(self, volume_3d: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Выборка и подготовка срезов
        num_total_slices = volume_3d.shape[0]
        step = select_step(num_total_slices)
        indices_to_process = quartile_sample_indices(num_total_slices, step)
        
        slices_to_process = [self._prepare_slice(volume_3d[i]) for i in indices_to_process]

        if not slices_to_process:
            return {
                'study_has_pathology': False, 'study_prob_pathology': 0.0,
                'study_processing_time': 0.0, 'pred_slices': []
            }

        # 2. --- ИЗМЕНЕНИЕ: Формирование батча в формате чата ---
        batch_messages = []
        for image in slices_to_process:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": self.user_prompt}, {"type": "image", "image": image}]}
            ]
            batch_messages.append(messages)

        # 3. Запуск инференса
        outputs = self.pipe(
            batch_messages,
            max_new_tokens=10, # Достаточно для "label: anomaly"
            batch_size=4 # Оставляем батчинг для скорости
        )

        # 4. Парсинг результатов
        slice_preds = []
        for output in outputs:
            # Извлекаем последний ответ модели
            text_content = output[0]['generated_text'][-1]['content']
            # Ищем 'anomaly' в ответе, это надежнее, чем парсить 'label:'
            is_anomaly = 'anomaly' in text_content.lower()
            slice_preds.append(is_anomaly)

        # 5. Агрегация и возврат результата в старом формате
        num_pathology_slices = sum(slice_preds)
        total_processed = len(slice_preds)
        
        study_prob_pathology = (num_pathology_slices / total_processed) if total_processed > 0 else 0.0
        study_has_pathology = study_prob_pathology >= threshold

        # Создаем полный список предсказаний для всех срезов (False по умолчанию)
        full_preds = [False] * num_total_slices
        for i, pred_idx in enumerate(indices_to_process):
            if slice_preds[i]:
                full_preds[pred_idx] = True

        processing_time = time.time() - start_time

        return {
            "study_has_pathology": study_has_pathology,
            "study_prob_pathology": study_prob_pathology,
            "study_processing_time": processing_time,
            "pred_slices": full_preds
        }