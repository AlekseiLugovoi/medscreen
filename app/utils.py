# app/utils.py

import numpy as np
import pydicom

def get_meta(ds):
    """Извлекает метаданные из DICOM-файла (более надежная версия)."""
    def get_value(tag, default="N/A"):
        # Используем числовой тег для большей надежности
        val = ds.get(tag, default)
        if isinstance(val, pydicom.dataelem.DataElement) and val.value is None:
            return default
        if isinstance(val, pydicom.multival.MultiValue):
            return str([v for v in val])
        return str(getattr(val, 'value', val)) # Получаем значение, если это объект DataElement

    return {
        "Modality": get_value(0x00080060), # Модальность
        "PatientID": get_value(0x00100020), # ID Пациента
        "StudyInstanceUID": get_value(0x0020000D), # UID Исследования
        "SeriesInstanceUID": get_value(0x0020000E), # UID Серии
        "PixelSpacing": get_value(0x00280030), # Размер пикселя
        "SliceThickness": get_value(0x00180050), # Толщина среза
        "Shape": str(ds.pixel_array.shape if "PixelData" in ds else "N/A"),
    }

# Остальные функции (to_uint8, norm_ct, normalize_dicom_and_get_frames)
# остаются такими же, как в прошлой версии.
def to_uint8(x):
    x = np.clip(x, 0, 1); return (x * 255).astype(np.uint8)

def norm_ct(ds, center=-600, width=1500):
    arr = ds.pixel_array.astype(np.float32)
    arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    lo, hi = center - width / 2, center + width / 2
    arr = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    return arr

# ... (и другие norm_* функции, если они у вас есть) ...

def normalize_dicom_and_get_frames(ds):
    mod = getattr(ds, "Modality", "").upper()
    if mod == "CT": processed_array = norm_ct(ds)
    # ... (и другие модальности) ...
    else:
        processed_array = ds.pixel_array.astype(np.float32)
        lo, hi = processed_array.min(), processed_array.max()
        processed_array = (processed_array - lo) / (hi - lo + 1e-6)
    if processed_array.ndim == 2: processed_array = np.expand_dims(processed_array, axis=0)
    return processed_array