import io
import tempfile
import zipfile
import numpy as np
import pydicom
import nibabel
from PIL import Image
from collections import defaultdict


def _get_dicom_orientation(ds):
    """Determine DICOM slice orientation (Axial, Sagittal, Coronal)."""
    try:
        orient = ds.ImageOrientationPatient
        normal_vec = np.cross(np.array(orient[:3]), np.array(orient[3:]))
        main_axis = np.argmax(np.abs(normal_vec))
        return ["Sagittal", "Coronal", "Axial"][main_axis]
    except Exception:
        return "Unknown"


def _parse_dicom_series(zf, dcm_files):
    """Parse DICOM files from a ZIP archive."""
    if not dcm_files:
        return None, "No DICOM files found in archive."

    # Multi-frame DICOM (single file with multiple frames)
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

    # Single-frame DICOM series
    series_dict = defaultdict(list)
    for filename in dcm_files:
        with zf.open(filename) as f:
            try:
                ds = pydicom.dcmread(f, force=True)
                if 'PixelData' in ds:
                    uid = getattr(ds, 'SeriesInstanceUID', 'default_series')
                    series_dict[uid].append(ds)
            except Exception:
                continue

    if not series_dict:
        return None, "Failed to read DICOM series from archive."

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


def _parse_single_nifti(zf, nii_filename):
    """Parse a single NIfTI file, return (series_uid, data_dict) or raise."""
    suffix = ".nii.gz" if nii_filename.lower().endswith(".nii.gz") else ".nii"
    with zf.open(nii_filename) as f:
        raw = f.read()
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(raw)
        tmp.flush()
        nii_img = nibabel.load(tmp.name)
        volume = nii_img.get_fdata().astype(np.float32)
        zooms = nii_img.header.get_zooms()
        orientation_code = ''.join(nibabel.aff2axcodes(nii_img.affine))
    if orientation_code.startswith(('L', 'R')) and orientation_code[1] in ('A', 'P'):
        volume = np.transpose(volume, (2, 1, 0))
        pixel_spacing = f"[{zooms[1]:.4f}, {zooms[0]:.4f}]"
        slice_thickness = f"{zooms[2]:.4f}"
    else:
        pixel_spacing = f"[{zooms[0]:.4f}, {zooms[1]:.4f}]"
        slice_thickness = f"{zooms[2]:.4f}"

    # Use filename (without path) as series UID
    name = nii_filename.rsplit("/", 1)[-1]
    series_uid = f"NIfTI_{name}"

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
    return series_uid, {"frames": volume, "meta": meta}


def _parse_nifti(zf, nii_files):
    """Parse one or more NIfTI files from a ZIP archive."""
    processed = {}
    errors = []

    for nii_filename in sorted(nii_files):
        try:
            series_uid, data = _parse_single_nifti(zf, nii_filename)
            processed[series_uid] = data
        except Exception as e:
            errors.append(f"{nii_filename}: {e}")

    if not processed:
        return None, f"Failed to read NIfTI files: {'; '.join(errors)}"

    return processed, None


def _parse_image_series(zf, img_files):
    """Parse image series (PNG/JPG) from a ZIP archive."""
    img_files.sort()
    frames = []
    for filename in img_files:
        with zf.open(filename) as f:
            img = Image.open(io.BytesIO(f.read())).convert('L')
            frames.append(np.array(img))

    if not frames:
        return None, "No images found in archive."

    volume = np.stack(frames).astype(np.float32)
    series_uid = "ImageSeries"
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
    """Detect data type in ZIP and dispatch to the appropriate parser."""
    try:
        if hasattr(file_input, 'read'):
            file_content = file_input.read()
        else:
            file_content = file_input

        if len(file_content) > 500 * 1024 * 1024:
            return None, "File too large (>500MB)."

        with zipfile.ZipFile(io.BytesIO(file_content)) as zf:
            file_list = [f for f in zf.namelist()
                         if not f.startswith('__MACOSX/') and not f.endswith('/')]

            if not file_list:
                return None, "Archive is empty."

            nii_files = [f for f in file_list if f.lower().endswith(('.nii', '.nii.gz'))]
            if nii_files:
                return _parse_nifti(zf, nii_files)

            img_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if img_files:
                return _parse_image_series(zf, img_files)

            # Try DICOM as fallback
            try:
                with zf.open(file_list[0]) as f:
                    pydicom.dcmread(f, stop_before_pixels=True, force=True)
                return _parse_dicom_series(zf, file_list)
            except pydicom.errors.InvalidDicomError:
                return None, "No supported files found (.nii, .png, .jpg, DICOM)."

    except zipfile.BadZipFile:
        return None, "File is not a valid ZIP archive."
    except Exception as e:
        return None, f"Unexpected error: {e}"
