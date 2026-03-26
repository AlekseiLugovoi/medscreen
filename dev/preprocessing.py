"""
Утилиты предобработки КТ-объёмов для CTScreener.

Пайплайн:
  1. extract_planes()  — извлечь проекции (axial/coronal/sagittal) из 3D-объёма
  2. prepare_uniform() — отобрать срезы с адаптивным шагом и сгруппировать в пачки
  3. model.run()       — инференс (в model.py)

Функции можно использовать и для визуализации (hu_to_pil).
"""

import math
import numpy as np
from PIL import Image


def hu_to_pil(slice_2d: np.ndarray, center: float = -600, width: float = 1500) -> Image.Image:
    """
    HU → 8-bit RGB PIL. Если данные уже 0-255 (PNG), пропускаем windowing.

    Стандартные HU-окна:
      Lung:        center=-600, width=1500  (лёгочная ткань)
      Narrow lung: center=-500, width=1000  (subtle GGO)
      Mediastinal: center=40,   width=350   (мягкие ткани, сосуды)
      Bone:        center=300,  width=2000  (кости, кальцинаты)
    """
    if slice_2d.max() <= 255 and slice_2d.min() >= 0:
        img = slice_2d.astype(np.uint8)
    else:
        lo, hi = center - width / 2, center + width / 2
        s = np.clip(slice_2d, lo, hi)
        img = ((s - lo) / (hi - lo) * 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def extract_planes(volume_3d: np.ndarray) -> dict:
    """
    Извлекает 3 проекции из (H, W, D) объёма. Все срезы, без кропа.

    Args:
        volume_3d: shape (H, W, D) — сырой 3D-объём (NIfTI/DICOM).

    Returns:
        {"axial": (D,H,W), "coronal": (H,D,W), "sagittal": (W,D,H)}
    """
    H, W, D = volume_3d.shape

    # Axial: (H,W,D) → (D,H,W)
    axial = np.transpose(volume_3d, (2, 1, 0))

    # Coronal: все H слоёв
    coronal = np.stack([np.rot90(volume_3d[j, :, :], 1) for j in range(H)])

    # Sagittal: все W слоёв
    sagittal = np.stack([np.rot90(volume_3d[:, i, :], 1) for i in range(W)])

    return {"axial": axial, "coronal": coronal, "sagittal": sagittal}


def select_step(n_slices: int, base: int = 50) -> int:
    """
    Адаптивный шаг выборки срезов, растёт логарифмически.

    Args:
        n_slices: кол-во срезов в проекции.
        base: объёмы до этого размера берутся целиком.
              Чем больше base, тем позже начинается прореживание.

    Примеры (base=50):
      50  → step=1 (все)
      100 → step=1 (все)
      200 → step=2 (каждый 2-й)
      300 → step=2 (каждый 2-й)
      500 → step=3 (каждый 3-й)
    """
    return max(1, int(math.log2(n_slices / base)))


def prepare_uniform(
    volume: np.ndarray,
    center: float = -600,
    width: float = 1500,
    group_size: int = 12,
    group_stride: int = None,
    base: int = 50,
) -> tuple[list[list[Image.Image]], list[tuple[int, int]]]:
    n = volume.shape[0]
    step = select_step(n, base)
    indices = list(range(0, n, step))

    if group_stride is None:
        group_stride = group_size

    groups, ranges = [], []
    for i in range(0, max(1, len(indices) - group_size + 1), group_stride):
        chunk = indices[i:i + group_size]
        if len(chunk) < min(group_size, 3):
            continue
        images = [hu_to_pil(volume[idx], center=center, width=width) for idx in chunk]
        groups.append(images)
        ranges.append((chunk[0], chunk[-1]))

    # последнее окно, если stride не дотянул до конца
    if groups and indices[-1] > ranges[-1][1]:
        chunk = indices[-group_size:]
        if len(chunk) >= 3:
            images = [hu_to_pil(volume[idx], center=center, width=width) for idx in chunk]
            groups.append(images)
            ranges.append((chunk[0], chunk[-1]))

    return groups, ranges
