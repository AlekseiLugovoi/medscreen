import logging
import time
import numpy as np
import torch
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer

<<<<<<< HEAD
=======
# import os
# os.environ["HF_TOKEN"] = "" # Public

>>>>>>> 80cab49 (Refactoring)
# --- ЛОГИРОВАНИЕ ---
model_logger = logging.getLogger('model_logger')
model_logger.setLevel(logging.INFO)
model_logger.propagate = False
if not model_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [MODEL] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    model_logger.addHandler(handler)

# --- Вспомогательные функции для выборки срезов ---
def select_step(n_slices: int) -> int:
    """Выбор базового шага n по числу срезов."""
    if n_slices < 50:   return 1
    if n_slices < 100:  return 2
    if n_slices < 200:  return 4
    if n_slices < 400:  return 6
    if n_slices < 600:  return 8
    return 10

def quartile_sample_indices(n_files: int, n: int) -> list[int]:
    """Q1/Q4 шаг n; Q2/Q3 шаг n//2."""
    if n == 0: n = 1
    q1, q2, q3 = n_files // 4, n_files // 2, (3 * n_files) // 4
    
    idx = set()
    idx.update(range(0,   q1, n))
    idx.update(range(q1,  q2, max(1, n//2)))
    idx.update(range(q2,  q3, max(1, n//2)))
    idx.update(range(q3, n_files, n))
    
    if n_files > 0:
        idx.add(n_files - 1)
        
    return sorted(list(idx))

class PathologyClassifier:
    """Класс-обертка для ML-модели BiomedCLIP."""
    def __init__(self, model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_logger.info(f"Инициализация модели '{model_name}' на устройстве: {self.device}")

        try:
            self.model, _, self.preprocess = create_model_and_transforms(f'hf-hub:{model_name}')
            self.tokenizer = get_tokenizer(f'hf-hub:{model_name}')
            self.model = self.model.to(self.device)
            self.model.eval()
            model_logger.info("Модель BiomedCLIP успешно загружена.")
        except Exception as e:
            model_logger.error(f"Критическая ошибка загрузки модели BiomedCLIP: {e}")
            raise

        self.pathology_texts = [
            "chest CT scan with obvious lung pathology and disease",
            "abnormal chest computed tomography showing clear pulmonary lesions", 
            "chest CT with visible lung nodules or masses",
        ]
        self.normal_texts = [
            "completely normal chest CT scan with healthy lungs",
            "chest CT with perfectly clear lung fields and no abnormalities",
            "normal chest computed tomography showing healthy pulmonary tissue",
        ]
        self.all_texts = self.pathology_texts + self.normal_texts
        self.n_pathology_texts = len(self.pathology_texts)
        
        self.confidence_threshold = 0.5
        self.min_consensus_group_size = 2
        
        model_logger.info(f"Порог уверенности: {self.confidence_threshold}, мин. размер группы: {self.min_consensus_group_size}")

    def _prepare_slice(self, slice_2d: np.ndarray) -> Image.Image:
        """Подготавливает один 2D срез: применяет окно и конвертирует в PIL.Image."""
        center, width = -600, 1500
        lo, hi = center - width / 2.0, center + width / 2.0
        img_windowed = np.clip(slice_2d, lo, hi)
        img_normalized = (img_windowed - lo) / (hi - lo + 1e-6)
        img_uint8 = (img_normalized * 255.0).astype(np.uint8)
        return Image.fromarray(img_uint8, mode="L").convert("RGB")

    def _classify_batch(self, images: list[Image.Image]) -> list[bool]:
        """Классифицирует батч изображений."""
        try:
            image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            text_inputs = self.tokenizer(self.all_texts).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_inputs).float()
                text_features = self.model.encode_text(text_inputs).float()
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                logits = image_features @ text_features.T
                probs = logits.softmax(dim=-1)
                
                pathology_probs = probs[:, :self.n_pathology_texts].sum(dim=1)
                normal_probs = probs[:, self.n_pathology_texts:].sum(dim=1)
                
                predictions = (pathology_probs > normal_probs) & (pathology_probs > self.confidence_threshold)
                return predictions.cpu().tolist()
                
        except Exception as e:
            model_logger.error(f"Ошибка при классификации батча: {e}")
            return [False] * len(images)

    def _apply_consensus_filtering(self, slice_predictions: dict) -> dict:
        """Применяет консенсусную фильтрацию - требует несколько положительных срезов подряд."""
        filtered_predictions = {}
        positive_indices = sorted([i for i, pred in slice_predictions.items() if pred])
        
        if not positive_indices:
            return filtered_predictions
        
        groups = []
        if positive_indices:
            current_group = [positive_indices[0]]
            for i in range(1, len(positive_indices)):
                if positive_indices[i] - positive_indices[i-1] <= 2:
                    current_group.append(positive_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [positive_indices[i]]
            groups.append(current_group)
        
        for group in groups:
            if len(group) >= self.min_consensus_group_size:
                for idx in group:
                    filtered_predictions[idx] = True
        
        return filtered_predictions

    def run_inference(self, volume_3d: np.ndarray) -> list[bool]:
        """Запускает инференс для всего 3D-объема."""
        start_time = time.perf_counter()
        n_slices = volume_3d.shape[0]
        model_logger.info(f"Начало инференса для объема из {n_slices} срезов.")

        step = select_step(n_slices)
        indices_to_check = quartile_sample_indices(n_slices, step)
        model_logger.info(f"Будет проанализировано {len(indices_to_check)} из {n_slices} срезов.")

        images_to_check = [self._prepare_slice(volume_3d[i]) for i in indices_to_check]

        batch_size = 8
        slice_predictions = {}
        model_logger.info(f"Запуск классификации батчами (размер батча: {batch_size})...")
        
        for i in range(0, len(images_to_check), batch_size):
            batch_images = images_to_check[i:i+batch_size]
            batch_indices = indices_to_check[i:i+batch_size]
            batch_results = self._classify_batch(batch_images)
            slice_predictions.update(zip(batch_indices, batch_results))
            
        filtered_predictions = self._apply_consensus_filtering(slice_predictions)

        predictions = [False] * n_slices
        for pathology_idx in filtered_predictions.keys():
            for j in range(max(0, pathology_idx - 1), min(n_slices, pathology_idx + 2)):
                predictions[j] = True
        
        duration = time.perf_counter() - start_time
        model_logger.info(
            f"Инференс завершен за {duration:.3f}с. "
            f"Первичных положительных: {sum(slice_predictions.values())}/{len(indices_to_check)}. "
            f"После фильтрации: {len(filtered_predictions)}. "
            f"Итого: {sum(predictions)}/{n_slices}."
        )
        
        return predictions