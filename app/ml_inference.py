# {
#     "pathologies_found": ["Пневмония"],
#     "study_processing_time": 25.3,
#     "pred_slices": [False, ..., True, True, True, ..., False],
#     "pneumonia": {
#         "name": "Пневмония",
#         "has_pathology": True,
#         "share_pathology_slices": 0.20,
#         "pathology_slices": [88, 90, 92, 95]
#     },
#     "lung_cancer": {
#         "name": "Рак легких",
#         "has_pathology": False,
#         "share_pathology_slices": 0.05,
#         "pathology_slices": [150]
#     }
#     "aortic_dilation": {
#         "name": "Расширение брюшной аорты",
#         "has_pathology": False,
#         "share_pathology_slices": 0.05,
#         "pathology_slices": [150]
#     }
# }

import logging
import time
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from typing import Dict, List, Any
import yaml
import os
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


config_path = os.path.join(os.path.dirname(__file__), 'pathology_config.yml')
with open(config_path, 'r', encoding='utf-8') as f:
    PATHOLOGY_CONFIG = yaml.safe_load(f)
model_logger.info(f"Загружено {len(PATHOLOGY_CONFIG)} патологий из конфигурации")

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

    def _prepare_slice(self, slice_2d: np.ndarray, cfg: dict) -> Image.Image:
        """
        Конвертирует 2D-срез в PIL Image
        с учётом окна/кроп-коэффициента, заданных в конфиге.
        """
        center  = cfg.get("window_center",  -600)
        width   = cfg.get("window_width",   1500)
        crop_fr = cfg.get("crop_fraction")          # None → без кропа

        lo, hi = center - width / 2, center + width / 2
        sl = np.clip(slice_2d, lo, hi)
        sl = (sl - lo) / (hi - lo + 1e-6)

        if crop_fr:
            h, w = sl.shape[-2:]
            ch, cw = int(h*crop_fr), int(w*crop_fr)
            y0, x0 = (h - ch)//2, (w - cw)//2
            sl = sl[y0:y0+ch, x0:x0+cw]

        img = (np.clip(sl, 0, 1)*255).astype(np.uint8)
        return Image.fromarray(img).convert("L")

    def _run_inference_for_prompt(self, slices: List[Image.Image], user_prompt: str, system_prompt: str) -> List[bool]:
        """Запускает инференс для набора срезов с заданным промптом."""
        batch_messages = []
        for image in slices:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image", "image": image}]}
            ]
            batch_messages.append(messages)

        outputs = self.pipe(
            batch_messages,
            max_new_tokens=30,
            batch_size=4
        )

        slice_preds = []
        for output in outputs:
            # output: [{'generated_text': [ {role: 'system'...}, {role: 'user'...}, {role: 'assistant', content: [...]} ]}]
            gen = output[0].get("generated_text", "")
            text_content = ""
            try:
                if isinstance(gen, list):
                    # берём последнее сообщение ассистента
                    for msg in reversed(gen):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                text_content = " ".join(
                                    c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                                )
                            else:
                                text_content = str(content)
                            break
                elif isinstance(gen, str):
                    text_content = gen
                else:
                    text_content = str(gen)
            except Exception:
                text_content = ""

            lc = text_content.lower()
            m = re.search(r"\blabel\s*:\s*(anomaly|normal)\b", lc)
            if m:
                label = m.group(1)
            else:
                # запасной вариант — строго искать метки, а не любое слово anomaly из инструкции
                if "label: anomaly" in lc:
                    label = "anomaly"
                elif "label: normal" in lc:
                    label = "normal"
                else:
                    label = "normal"  # по умолчанию — нормальный, чтобы избегать ложных срабатываний

            is_anomaly = (label == "anomaly")
            slice_preds.append(is_anomaly)

        return slice_preds

    @torch.inference_mode()
    def run_inference(self, volume_3d: np.ndarray) -> Dict[str, Any]:
        start_time = time.time()
        
        num_total_slices = volume_3d.shape[0]
        step = select_step(num_total_slices)
        indices_to_process = quartile_sample_indices(num_total_slices, step)
        
        model_logger.info(f"Начало инференса: {len(indices_to_process)}/{num_total_slices} срезов (шаг={step}) для {len(PATHOLOGY_CONFIG)} патологий.")
        
        if not indices_to_process:
            return {
                'study_has_pathology': False, 'pathologies_found': [],
                'study_processing_time': 0.0,
                'pred_slices': [False] * num_total_slices
            }

        # --- ЗАПУСК ИНФЕРЕНСА ДЛЯ КАЖДОЙ ПАТОЛОГИИ ---
        pathology_details = {}
        all_positive_indices = set()

        for key, config in PATHOLOGY_CONFIG.items():
            model_logger.info(f"Проверка на: {config['name']}…")
            slices = [self._prepare_slice(volume_3d[i], config)
                      for i in indices_to_process]

            slice_preds = self._run_inference_for_prompt(
                slices, config['prompt'], config['system_prompt']
            )
            
            num_pathology_slices = sum(slice_preds)
            total_processed = len(slice_preds)
            
            pathology_share = (num_pathology_slices / total_processed) if total_processed > 0 else 0.0
            has_pathology = pathology_share >= config['threshold']
            
            positive_indices = [indices_to_process[i] for i, pred in enumerate(slice_preds) if pred]
            if has_pathology:
                all_positive_indices.update(positive_indices)

            pathology_details[key] = {
                "name": config['name'],
                "has_pathology": has_pathology,
                "share_pathology_slices": pathology_share,
                "pathology_slices": positive_indices
            }

        # --- АГРЕГАЦИЯ РЕЗУЛЬТАТОВ ---
        study_has_pathology = any(details['has_pathology'] for details in pathology_details.values())
        pathologies_found = [details['name'] for details in pathology_details.values() if details['has_pathology']]

        full_preds = [False] * num_total_slices
        for idx in all_positive_indices:
            full_preds[idx] = True

        processing_time = time.time() - start_time
        
        model_logger.info(
            f"Инференс завершен за {processing_time:.2f}с | "
            f"Патология: {'Да' if study_has_pathology else 'Нет'} ({', '.join(pathologies_found) or 'N/A'}) | "
            f"{get_gpu_memory_usage_str()}"
        )
        
        # --- ФОРМИРОВАНИЕ ИТОГОВОГО СЛОВАРЯ ---
        result = {
            "pathologies_found": pathologies_found,
            "study_processing_time": processing_time,
            "pred_slices": full_preds
        }
        # Добавляем детали по каждой патологии в основной словарь
        result.update(pathology_details)

        return result

