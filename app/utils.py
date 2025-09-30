import streamlit as st
import pydicom
import numpy as np
import io
import zipfile
from collections import defaultdict
import imageio
import nibabel as nib
from ml_inference import PathologyClassifier
import time

# --- Инициализация ML-модели ---
@st.cache_resource
def get_model():
    return PathologyClassifier()

model = get_model()

# --- Вспомогательные функции ---

def apply_window(volume, center, width):
    """Применяет оконное преобразование (windowing) к объему."""
    lo, hi = center - width / 2, center + width / 2
    volume = (np.clip(volume, lo, hi) - lo) / (hi - lo + 1e-6)
    return volume

def to_uint8(x):
    """Конвертирует нормализованный массив (0-1) в 8-битное изображение (0-255)."""
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)

@st.cache_data
def get_windowed_frames(series_uid: str, center: int, width: int) -> np.ndarray:
    """
    Кэширует результат применения оконного преобразования.
    Принимает UID серии, а не сам массив, для эффективного кэширования.
    """
    raw_frames = st.session_state.processed_data[series_uid]["frames"]
    return apply_window(raw_frames, center, width)

@st.cache_data(show_spinner="Подготовка кадров для просмотра...")
def precompute_uint8_frames(series_uid: str, window_name: str, ct_windows: dict) -> list:
    """
    Предварительно конвертирует все кадры в uint8.
    Принимает UID и имя окна для кэширования.
    """
    series = st.session_state.processed_data[series_uid]
    meta = series["meta"]
    
    if meta.get("Modality") == "CT":
        center, width = ct_windows[window_name]
        frames = get_windowed_frames(series_uid, center, width)
    else:
        raw_frames = series["frames"]
        lo, hi = raw_frames.min(), raw_frames.max()
        frames = (raw_frames - lo) / (hi - lo + 1e-6)
        
    return [to_uint8(frame) for frame in frames]

def get_dicom_orientation(ds):
    """Определяет ориентацию срезов DICOM."""
    try:
        orient = ds.ImageOrientationPatient
        row_vec, col_vec = np.array(orient[:3]), np.array(orient[3:])
        normal_vec = np.cross(row_vec, col_vec)
        main_axis = np.argmax(np.abs(normal_vec))
        if main_axis == 0: return "Sagittal"
        if main_axis == 1: return "Coronal"
        if main_axis == 2: return "Axial"
    except Exception:
        return "Unknown"

# --- Обработчики форматов с отчетом о статусе ---

def _process_dicom_zip(uploaded_file, status):
    """Обрабатывает ZIP-архив с отчетом о прогрессе."""
    series_dict = defaultdict(list)
    with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as zf:
        filenames = [name for name in zf.namelist() if not name.endswith('/')]
        total_files = len(filenames)
        status.write(f"Распаковка архива... Найдено файлов: {total_files}")
        time.sleep(1)

        for i, filename in enumerate(filenames):
            if (i + 1) % 25 == 0:
                status.write(f"Чтение DICOM-файлов ({i+1}/{total_files})...")
            try:
                with zf.open(filename) as f:
                    ds = pydicom.dcmread(f, force=True)
                    if 'PixelData' in ds:
                        series_dict[ds.SeriesInstanceUID].append(ds)
            except Exception:
                continue

    if not series_dict: return {}
    
    processed_series = {}
    for series_uid, datasets in series_dict.items():
        if len(datasets) <= 1: continue
        status.write(f"Сортировка {len(datasets)} срезов...")
        datasets.sort(key=lambda ds: int(getattr(ds, 'InstanceNumber', 0)))
        
        status.write("Сборка 3D-объема...")
        time.sleep(1)
        proxy_ds = datasets[0]
        volume = np.stack([ds.pixel_array for ds in datasets]).astype(np.float32)
        volume = volume * float(getattr(proxy_ds, "RescaleSlope", 1.0)) + float(getattr(proxy_ds, "RescaleIntercept", 0.0))

        meta = {
            "SourceFormat": "DICOM Series (ZIP)", "SeriesInstanceUID": series_uid,
            "Modality": str(getattr(proxy_ds, "Modality", "N/A")),
            "PixelSpacing": str(getattr(proxy_ds, "PixelSpacing", "N/A")),
            "SliceThickness": str(getattr(proxy_ds, "SliceThickness", "N/A")),
            "num_frames": len(datasets), "orientation": get_dicom_orientation(proxy_ds),
            "missing_slices": sum(np.diff(sorted([d.SliceLocation for d in datasets if 'SliceLocation' in d.dir()])) > (1.5 * float(getattr(proxy_ds, 'SliceThickness', 1.0))))
        }
        processed_series[series_uid] = {"frames": volume, "meta": meta}
    return processed_series

def _process_multiframe_dicom(uploaded_file, status):
    """Обрабатывает многокадровый DICOM файл."""
    status.write("Чтение многокадрового DICOM...")
    try:
        ds = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()), force=True)
        num_frames = int(getattr(ds, "NumberOfFrames", 1))
        if num_frames <= 1: return {}

        volume = ds.pixel_array.astype(np.float32)
        volume = volume * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        
        series_uid = getattr(ds, "SeriesInstanceUID", "N/A")
        meta = {
            "SourceFormat": "Multi-frame DICOM", "SeriesInstanceUID": series_uid,
            "Modality": str(getattr(ds, "Modality", "N/A")),
            "PixelSpacing": str(getattr(ds, "PixelSpacing", "N/A")),
            "SliceThickness": str(getattr(ds, "SliceThickness", "N/A")),
            "num_frames": num_frames, "orientation": get_dicom_orientation(ds), "missing_slices": 0
        }
        return {series_uid: {"frames": volume, "meta": meta}}
    except Exception:
        return {}

def _process_nifti_file(uploaded_file, status):
    """Обрабатывает файл формата .nii или .nii.gz."""
    status.write("Чтение файла NIfTI...")
    try:
        with io.BytesIO(uploaded_file.getvalue()) as bio:
            bio.name = uploaded_file.name
            img = nib.load(bio)

        volume = img.get_fdata(dtype=np.float32)
        if volume.ndim == 3: volume = np.transpose(volume, (2, 0, 1))
        
        header = img.header
        zooms = header.get_zooms()
        series_uid = f"nii_{uploaded_file.name.split('.')[0]}"

        meta = {
            "SourceFormat": "NIfTI", "SeriesInstanceUID": series_uid, "Modality": "NIFTI",
            "PixelSpacing": f"[{zooms[0]:.2f}, {zooms[1]:.2f}]", "SliceThickness": f"{zooms[2]:.2f}",
            "num_frames": volume.shape[0], "orientation": "".join(nib.aff2axcodes(img.affine)), "missing_slices": 0
        }
        return {series_uid: {"frames": volume, "meta": meta}}
    except Exception as e:
        st.error(f"Ошибка обработки NIfTI: {e}")
        return {}

# --- Главный диспетчер и валидация ---

def process_uploaded_file(uploaded_file, status):
    """Главная функция-диспетчер: определяет тип файла и вызывает нужный обработчик."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".zip"):
        return _process_dicom_zip(uploaded_file, status)
    elif filename.endswith(".dcm"):
        return _process_multiframe_dicom(uploaded_file, status)
    elif filename.endswith((".nii", ".nii.gz")):
        return _process_nifti_file(uploaded_file, status)
    else:
        st.error("Неподдерживаемый формат файла.")
        return {}

def validate_series(meta):
    """Проводит универсальную валидацию серии по заданным критериям."""
    checks = []
    source_format = meta.get("SourceFormat", "Unknown")

    is_valid_modality = meta.get("Modality") in ["CT", "NIFTI"]
    checks.append({
        "text": f"Модальность исследования: {meta.get('Modality')}", "status": is_valid_modality,
        "desc": "Проверяем, что исследование является компьютерной томографией (CT) или файлом NIfTI."
    })

    orientation = meta.get("orientation", "")
    is_axial = "Axial" in orientation or "AS" in orientation
    checks.append({
        "text": f"Ориентация срезов: {orientation}", "status": is_axial,
        "desc": "Проверяем, что срезы расположены в стандартной аксиальной проекции (поперечные срезы тела)."
    })

    missing_slices = meta.get("missing_slices", -1)
    is_complete = missing_slices == 0
    desc_integrity = "Для DICOM-серий проверяем отсутствие пропусков между срезами. Для цельных форматов (NIfTI, Multi-frame DICOM) проверка не требуется."
    details = f" (найдено пропусков: {missing_slices})" if not is_complete and missing_slices > 0 else ""
    checks.append({
        "text": f"Целостность серии{details}", "status": is_complete, "desc": desc_integrity
    })

    is_single_series = source_format != "DICOM Series (ZIP)"
    desc_single = "Для DICOM-архивов проверяем, что все файлы относятся к одной серии. Для цельных форматов проверка не требуется."
    checks.append({
        "text": "Одна серия в файле", "status": is_single_series, "desc": desc_single
    })
    
    return checks

# --- Функции для ML и GIF ---

@st.cache_data(show_spinner="Анализ срезов моделью...")
def run_pathology_inference(_model, volume_3d):
    """Кэшируемая обертка для запуска инференса модели."""
    return _model.run_inference(volume_3d)

@st.cache_data(show_spinner="Создание анимации...")
def create_gif_from_frames(series_uid: str, window_name: str, ct_windows: dict, duration_ms: int = 50) -> bytes:
    """
    Создает GIF-анимацию. Принимает UID и имя окна для кэширования.
    """
    uint8_frames = precompute_uint8_frames(series_uid, window_name, ct_windows)
    with io.BytesIO() as buffer:
        imageio.mimsave(buffer, uint8_frames, format='GIF', duration=duration_ms, loop=0)
        return buffer.getvalue()