import pandas as pd
import io
from typing import List
from fastapi import FastAPI, UploadFile, File

from app.file_io import parse_zip_archive
from app.data_validation import validate_series
from app.ml_inference import PathologyClassifier, model_logger, get_gpu_memory_usage_str


app = FastAPI(
    title="MedScreen API",
    description="API для пакетной обработки медицинских исследований.",
    version="1.0.0"
)

# Загружаем модель при старте
model = PathologyClassifier()


@app.post("/process", tags=["Processing"])
async def process(files: List[UploadFile] = File(...)):
    """
    Принимает один или несколько ZIP-архивов, обрабатывает их
    и возвращает результат в виде JSON.
    """
    all_results = []

    for file in files:
        file_content = await file.read()
        series_data, error_message = parse_zip_archive(file_content)

        if not series_data or error_message:
            all_results.append({
                'archive_name': file.filename, 'series_uid': error_message or "Parsing error",
                'is_valid': False, 'has_pathology': False, 'pred_pathology': "0.0000",
                'ml_processing_time': "0.00s", 'source_format': 'N/A', 'modality': 'N/A',
                'body_part': 'N/A', 'orientation': 'N/A', 'num_frames': 0
            })
            continue

        for series_uid, data in series_data.items():
            meta = data['meta']
            validation_checks = validate_series(meta)
            is_valid = all(check['status'] for check in validation_checks)

            if is_valid and len(data['frames']) > 0:
                inference_results = model.run_inference(data['frames'])
                has_pathology_flag = inference_results.get('study_has_pathology', False)
                final_prob = inference_results.get('study_prob_pathology', 0.0)
                ml_time = inference_results.get('study_processing_time', 0.0)
            else:
                has_pathology_flag, final_prob, ml_time = False, 0.0, 0.0

            all_results.append({
                'archive_name': file.filename,
                'series_uid': series_uid,
                'source_format': meta.get('SourceFormat', 'N/A'),
                'modality': meta.get('Modality', 'N/A'),
                'body_part': meta.get('BodyPartExamined', 'N/A'),
                'orientation': meta.get('orientation', 'N/A'),
                'num_frames': meta.get('num_frames', 0),
                'is_valid': is_valid,
                'has_pathology': has_pathology_flag,
                'pred_pathology': f"{final_prob:.4f}",
                'ml_processing_time': f"{ml_time:.2f}s"
            })

    return {"results": all_results}

# Команда для локального запуска:
# uvicorn api:app --host 0.0.0.0 --port 8000 --reload