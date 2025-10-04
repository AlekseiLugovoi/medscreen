import pandas as pd
import io
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

# --- ИСПРАВЛЕНИЕ: Правильные импорты ---
from app.file_io import parse_zip_archive
from app.data_validation import validate_series
from app.ml_inference import PathologyClassifier, model_logger, get_gpu_memory_usage_str
# --- КОНЕЦ ИСПРАВЛЕНИЯ ---

app = FastAPI(
    title="MedScreen API",
    description="API для пакетной обработки медицинских исследований.",
    version="1.0.0"
)

# Загружаем модель при старте, чтобы она была готова к использованию
model = PathologyClassifier()


@app.post("/api/v1/process_single", tags=["Processing"])
async def process_single_archive(file: UploadFile = File(...)):
    """
    Принимает один ZIP-архив, проводит полный анализ (валидация + ML)
    и возвращает результат в виде JSON.
    """
    # --- ИСПРАВЛЕНИЕ: Используем file.file для передачи файлового объекта ---
    series_data, error_message = parse_zip_archive(file.file)

    if not series_data or error_message:
        return {"error": error_message or "Parsing error"}

    # Берем первую серию
    series_uid = list(series_data.keys())[0]
    data = series_data[series_uid]
    meta = data['meta']
    
    validation_checks = validate_series(meta)
    is_valid = all(check['status'] for check in validation_checks)

    if is_valid and len(data['frames']) > 0:
        model_logger.info(f"Запуск инференса для {file.filename}. {get_gpu_memory_usage_str()}")
        # --- ИСПРАВЛЕНИЕ: Вызываем метод модели ---
        inference_results = model.run_inference(data['frames'])
        model_logger.info(f"Инференс завершен. {get_gpu_memory_usage_str()}")
    else:
        inference_results = {
            'study_has_pathology': False,
            'study_prob_pathology': 0.0,
            'study_processing_time': 0.0,
            'pred_slices': []
        }

    return {
        "validation_passed": is_valid,
        "validation_checks": validation_checks,
        "inference_results": inference_results
    }


@app.post("/api/v1/upload", tags=["Processing"])
async def process_archives(files: List[UploadFile] = File(...)):
    """
    Принимает один или несколько ZIP-архивов, обрабатывает их
    и возвращает результат в виде CSV-файла.
    """
    csv_data = []
    fieldnames = [
        'archive_name', 'series_uid', 'source_format', 'modality',
        'body_part', 'orientation', 'num_frames', 'is_valid',
        'has_pathology', 'pred_pathology', 'ml_processing_time'
    ]

    for file in files:
        # --- ИСПРАВЛЕНИЕ: Используем file.file ---
        series_data, error_message = parse_zip_archive(file.file)

        if not series_data or error_message:
            csv_data.append({'archive_name': file.filename, 'is_valid': False, 'series_uid': error_message or "Parsing error"})
            continue

        for series_uid, data in series_data.items():
            meta = data['meta']
            validation_checks = validate_series(meta)
            is_valid = all(check['status'] for check in validation_checks)

            if is_valid and len(data['frames']) > 0:
                # --- ИСПРАВЛЕНИЕ: Вызываем метод модели ---
                inference_results = model.run_inference(data['frames'])
                has_pathology_flag = inference_results.get('study_has_pathology', False)
                final_prob = inference_results.get('study_prob_pathology', 0.0)
                ml_time = inference_results.get('study_processing_time', 0.0)
            else:
                has_pathology_flag = False
                final_prob = 0.0
                ml_time = 0.0

            csv_data.append({
                'archive_name': file.filename,
                'series_uid': series_uid,
                'source_format': meta.get('SourceFormat', 'N/A'),
                'modality': meta.get('Modality', 'N/A'),
                'body_part': meta.get('BodyPartExamined', 'N/A'),
                'orientation': meta.get('orientation', 'N/A'),
                'num_frames': meta.get('num_frames', 'N/A'),
                'is_valid': is_valid,
                'has_pathology': has_pathology_flag,
                'pred_pathology': f"{final_prob:.4f}",
                'ml_processing_time': f"{ml_time:.2f}s"
            })

    df = pd.DataFrame(csv_data, columns=fieldnames).fillna("N/A")
    stream = io.StringIO()
    df.to_csv(stream, index=False)

    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=report.csv"}
    )
    return response

# Команда для локального запуска:
# uvicorn api:app --host 0.0.0.0 --port 8000 --reload