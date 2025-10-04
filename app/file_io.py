# --- Модуль для парсинга входных ZIP-архивов ---
#
# Основная функция: parse_zip_archive(uploaded_file)
#
# Флоу:
# 1. Получает на вход загруженный ZIP-файл.
# 2. Анализирует содержимое, определяет тип данных (DICOM, NIfTI, PNG/JPG).
# 3. Вызывает соответствующий внутренний обработчик.
# 4. Возвращает стандартизированный результат в виде кортежа: (data, error_message).
#
# Структура успешного вывода (data):
# {
#     "SeriesInstanceUID_1": {
#         "frames": np.ndarray,  # 3D массив (срезы, высота, ширина)
#         "meta": {
#             "SourceFormat": str,      # "DICOM Series", "NIfTI", "Image Series"
#             "Modality": str,          # "CT", "NIFTI", "IMAGE"
#             "orientation": str,       # "Axial", "Sagittal", "Coronal", "Unknown"
#             "num_frames": int,        # Фактическое количество срезов
#             "StudyInstanceUID": str,  # UID исследования (если есть)
#             "PixelSpacing": str,      # Расстояние между центрами пикселей
#             "SliceThickness": str,    # Толщина среза
#             "BodyPartExamined": str   # Исследуемая часть тела (если есть)
#         }
#     },
#     # ... могут быть и другие серии, если найдены в DICOM
# }
#
# При ошибке `data` будет `None`, а `error_message` - строкой с описанием.

import io
import zipfile
import numpy as np
import pydicom
import nibabel
from PIL import Image
from collections import defaultdict

def _get_dicom_orientation(ds):
    """Определяет ориентацию срезов DICOM (Axial, Sagittal, Coronal)."""
    try:
        orient = ds.ImageOrientationPatient
        normal_vec = np.cross(np.array(orient[:3]), np.array(orient[3:]))
        main_axis = np.argmax(np.abs(normal_vec))
        return ["Sagittal", "Coronal", "Axial"][main_axis]
    except Exception:
        return "Unknown"

def _parse_dicom_series(zf, dcm_files):
    """Парсит серию DICOM-файлов из архива."""
    if not dcm_files:
        return None, "В архиве не найдено DICOM-файлов."

    # Обработка многокадрового DICOM (если он один в архиве)
    if len(dcm_files) == 1:
        with zf.open(dcm_files[0]) as f:
            ds = pydicom.dcmread(io.BytesIO(f.read()), force=True)
        if int(getattr(ds, "NumberOfFrames", 1)) > 1:
            volume = ds.pixel_array.astype(np.float32)
            volume = volume * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
            series_uid = getattr(ds, "SeriesInstanceUID", "MultiFrame_DICOM")
            meta = {
                "SourceFormat": "Multi-frame DICOM",
                "Modality": getattr(ds, "Modality", "N/A"),
                "orientation": _get_dicom_orientation(ds),
                "num_frames": ds.NumberOfFrames,
                "StudyInstanceUID": getattr(ds, "StudyInstanceUID", "N/A"),
                "PixelSpacing": str(getattr(ds, "PixelSpacing", "N/A")),
                "SliceThickness": str(getattr(ds, "SliceThickness", "N/A")),
                "BodyPartExamined": getattr(ds, "BodyPartExamined", "N/A"),
            }
            return {series_uid: {"frames": volume, "meta": meta}}, None

    # Обработка серии однокадровых DICOM
    series_dict = defaultdict(list)
    for filename in dcm_files:
        with zf.open(filename) as f:
            try:
                ds = pydicom.dcmread(f, force=True)
                if 'PixelData' in ds:
                    # Если UID отсутствует, используем "default_series" как ключ
                    uid = getattr(ds, 'SeriesInstanceUID', 'default_series')
                    series_dict[uid].append(ds)
            except Exception:
                continue
    
    if not series_dict:
        return None, "Не удалось прочитать DICOM-серии в архиве."

    processed_series = {}
    for series_uid, datasets in series_dict.items():
        datasets.sort(key=lambda ds: int(getattr(ds, 'InstanceNumber', 0)))
        proxy_ds = datasets[0]
        volume = np.stack([ds.pixel_array for ds in datasets]).astype(np.float32)
        volume = volume * float(getattr(proxy_ds, "RescaleSlope", 1.0)) + float(getattr(proxy_ds, "RescaleIntercept", 0.0))
        meta = {
            "SourceFormat": "DICOM Series",
            "Modality": getattr(proxy_ds, "Modality", "N/A"),
            "orientation": _get_dicom_orientation(proxy_ds),
            "num_frames": len(datasets),
            "StudyInstanceUID": getattr(proxy_ds, "StudyInstanceUID", "N/A"),
            "PixelSpacing": str(getattr(proxy_ds, "PixelSpacing", "N/A")),
            "SliceThickness": str(getattr(proxy_ds, "SliceThickness", "N/A")),
            "BodyPartExamined": getattr(proxy_ds, "BodyPartExamined", "N/A"),
        }
        processed_series[series_uid] = {"frames": volume, "meta": meta}
        
    return processed_series, None

def _parse_nifti(zf, nii_files):
    """Парсит NIfTI-файл из архива."""
    if len(nii_files) > 1:
        return None, "Архив должен содержать только один NIfTI-файл."
    
    nii_filename = nii_files[0]
    try:
        with zf.open(nii_filename) as f:
            # Nibabel работает с файлоподобными объектами
            nii_img = nibabel.Nifti1Image.from_stream(io.BytesIO(f.read()))
        
        volume = nii_img.get_fdata().astype(np.float32)
        zooms = nii_img.header.get_zooms()
        
        orientation_code = ''.join(nibabel.aff2axcodes(nii_img.affine))
        if orientation_code.startswith(('L','R')) and orientation_code[1] in ('A','P'):
             volume = np.transpose(volume, (2, 1, 0))
             # Соответственно меняем местами размеры вокселя
             pixel_spacing = f"[{zooms[1]:.4f}, {zooms[0]:.4f}]"
             slice_thickness = f"{zooms[2]:.4f}"
        else: # Предполагаем, что уже в нужной ориентации
             pixel_spacing = f"[{zooms[0]:.4f}, {zooms[1]:.4f}]"
             slice_thickness = f"{zooms[2]:.4f}"


        series_uid = "NIfTI_" + nii_filename
        meta = {
            "SourceFormat": "NIfTI",
            "Modality": "NIFTI",
            "orientation": "Axial",
            "num_frames": volume.shape[0],
            "StudyInstanceUID": "N/A",
            "PixelSpacing": pixel_spacing,
            "SliceThickness": slice_thickness,
            "BodyPartExamined": "N/A",
        }
        return {series_uid: {"frames": volume, "meta": meta}}, None
    except Exception as e:
        return None, f"Ошибка чтения NIfTI файла: {e}"

def _parse_image_series(zf, img_files):
    """Парсит серию изображений (PNG/JPG) из архива."""
    img_files.sort() # Сортировка по имени файла для правильного порядка срезов
    frames = []
    for filename in img_files:
        with zf.open(filename) as f:
            img = Image.open(io.BytesIO(f.read())).convert('L') # 'L' = grayscale
            frames.append(np.array(img))
    
    if not frames:
        return None, "Не найдено изображений в архиве."

    volume = np.stack(frames).astype(np.float32)
    series_uid = "ImageSeries_" + zf.filename.split('/')[-1]
    meta = {
        "SourceFormat": "Image Series",
        "Modality": "IMAGE",
        "orientation": "Unknown",
        "num_frames": len(frames),
        "StudyInstanceUID": "N/A",
        "PixelSpacing": "N/A",
        "SliceThickness": "N/A",
        "BodyPartExamined": "N/A",
    }
    return {series_uid: {"frames": volume, "meta": meta}}, None

def parse_zip_archive(file_input):
    """Определяет тип данных в ZIP и вызывает соответствующий парсер."""
    try:
        if hasattr(file_input, 'read'):
            file_content = file_input.read()
        else:
            file_content = file_input

        # Добавляем проверку размера
        if len(file_content) > 500 * 1024 * 1024:  # 500MB лимит
            return None, "Файл слишком большой (>500MB)"
        
        with zipfile.ZipFile(io.BytesIO(file_content)) as zf:
            file_list = [f for f in zf.namelist() if not f.startswith('__MACOSX/') and not f.endswith('/')]

            if not file_list:
                return None, "Архив пуст."

            # Сначала проверяем на явные форматы, чтобы избежать ложных срабатываний
            nii_files = [f for f in file_list if f.lower().endswith(('.nii', '.nii.gz'))]
            if nii_files:
                return _parse_nifti(zf, nii_files)

            img_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if img_files:
                return _parse_image_series(zf, img_files)

            # Если нет явных форматов, пробуем прочитать как DICOM
            try:
                with zf.open(file_list[0]) as f:
                    pydicom.dcmread(f, stop_before_pixels=True, force=True)
                # Если чтение успешно, считаем все файлы в архиве DICOM-серией
                return _parse_dicom_series(zf, file_list)
            except pydicom.errors.InvalidDicomError:
                # Если первый файл не DICOM, выдаем ошибку
                return None, "В архиве не найдены поддерживаемые файлы (.nii, .png, .jpg) и он не является DICOM-серией."

    except zipfile.BadZipFile:
        return None, "Загруженный файл не является ZIP-архивом или поврежден."
    except Exception as e:
        return None, f"Произошла непредвиденная ошибка: {e}"