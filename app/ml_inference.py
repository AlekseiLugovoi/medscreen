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
        model_logger.info(f"Инициализация модели '{model_name}' на устройстве: {self.device}")

        try:
            # ИСПРАВЛЕНИЕ: используем правильный тип pipeline как у коллеги
            self.pipe = pipeline(
                "image-text-to-text",  # Не "image-to-text"!
                model=model_name,
                model_kwargs={"torch_dtype": self.torch_dtype},
                device=self.device,
            )
            model_logger.info("Модель MedGemma успешно загружена.")
        except Exception as e:
            model_logger.error(f"Критическая ошибка загрузки модели MedGemma: {e}")
            raise

        # ИСПРАВЛЕНИЕ: используем тот же промпт что и у коллеги
        self.prompt = """Task: classify chest CT scan for pulmonary abnormalities.
Steps:
1) Examine both lungs for opacities, consolidations, ground-glass changes, pleural effusion, pneumothorax, fibrosis, or nodules.
2) If no abnormalities are visible, output label: normal.
3) If any abnormality is suspected, output label: anomaly.
4) Output format:
   - label: normal OR label: anomaly"""
        
    def _prepare_slice(self, slice_2d: np.ndarray) -> Image.Image:
        center, width = -600, 1500
        lo, hi = center - width / 2.0, center + width / 2.0
        img_windowed = np.clip(slice_2d, lo, hi)
        img_normalized = (img_windowed - lo) / (hi - lo + 1e-6)
        img_uint8 = (img_normalized * 255.0).astype(np.uint8)
        return Image.fromarray(img_uint8, mode="L").convert("RGB")

    # --- ИЗМЕНЕННАЯ ЛОГИКА ---
    @torch.inference_mode()
    def run_inference(self, volume_3d: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
        start_time = time.perf_counter()
        if torch.cuda.is_available(): 
            torch.cuda.reset_peak_memory_stats()

        n_slices = volume_3d.shape[0]
        model_logger.info(f"Инференс стартовал для {n_slices} срезов. {get_gpu_memory_usage_str()}")

        step = select_step(n_slices)
        indices_to_check = quartile_sample_indices(n_slices, step)
        images_to_check = [self._prepare_slice(volume_3d[i]) for i in indices_to_check]

        # Этап 1: Сбор "голосов" от модели (как у коллеги)
        counts = {"normal": 0, "anomaly": 0}
        slice_labels = {} # Сохраняем метку для каждого среза

        for i, image in enumerate(images_to_check):
            slice_idx = indices_to_check[i]
            try:
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
                    {"role": "user", "content": [{"type": "text", "text": self.prompt}, {"type": "image", "image": image}]}
                ]
                output = self.pipe(text=messages, max_new_tokens=256)
                
                if isinstance(output, list) and len(output) > 0:
                    generated_text = output[0]["generated_text"][-1]["content"]
                    label = "unknown"
                    try:
                        parsed_label = generated_text.split("label:")[1].split("\n")[0].strip().lower()
                        if "anomaly" in parsed_label:
                            label = "anomaly"
                        elif "normal" in parsed_label:
                            label = "normal"
                    except Exception:
                        pass # label остается "unknown"
                    
                    if label in counts:
                        counts[label] += 1
                    slice_labels[slice_idx] = label

            except Exception as e:
                model_logger.warning(f"Ошибка при классификации среза {slice_idx}: {e}")

        # Этап 2: Агрегация и принятие решения (как у коллеги)
        total_votes = counts["normal"] + counts["anomaly"]
        final_probability = (counts["anomaly"] / total_votes) if total_votes > 0 else 0.0
        
        has_pathology = final_probability >= threshold

        # Этап 3: Формирование результата
        preds = [False] * n_slices
        raw_probs = [0.0] * n_slices # Теперь это просто для информации, не для решения

        if has_pathology:
            # Если общая вероятность высокая, помечаем все срезы, где модель нашла аномалию
            for slice_idx, label in slice_labels.items():
                if label == "anomaly":
                    # Распространяем на соседние срезы для лучшей визуализации
                    for j in range(max(0, slice_idx - 1), min(n_slices, slice_idx + 2)):
                        preds[j] = True
        
        # Заполняем raw_probs для отладки
        for slice_idx, label in slice_labels.items():
            if label == "anomaly":
                raw_probs[slice_idx] = 0.9
            elif label == "normal":
                raw_probs[slice_idx] = 0.05

        duration = time.perf_counter() - start_time
        model_logger.info(f"Инференс закончен за {duration:.2f}с. Итоговая вер-ть: {final_probability:.4f}. {get_gpu_memory_usage_str()}")
        
        # --- ИСПРАВЛЕНИЕ: Возвращаем словарь с новыми, информативными ключами ---
        return {
            "pred_slices": preds,                   # list[bool] - Бинарные предсказания для каждого среза
            "prob_slices": raw_probs,               # list[float] - "Сырые" вероятности для каждого среза (для отладки)
            "study_processing_time": duration,      # float - Время обработки исследования моделью
            "study_has_pathology": has_pathology,   # bool - Итоговый бинарный ответ по всему исследованию
            "study_prob_pathology": final_probability # float - Итоговая вероятность патологии для исследования
        }