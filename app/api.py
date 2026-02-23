from typing import List
from fastapi import FastAPI, UploadFile, File

from app.file_io import parse_zip_archive
from app.data_validation import validate_series
from app.ml_inference import CTScreener


app = FastAPI(
    title="MedScreen API",
    description="Chest CT pathology screening API powered by MedGemma + vLLM.",
    version="2.0.0",
)

model = CTScreener()

# All response keys — always present, None when not applicable
_EMPTY_RESULT = {
    "archive_name": None,
    "series_uid": None,
    "source_format": None,
    "modality": None,
    "body_part": None,
    "num_frames": None,
    "is_valid": False,
    "verdict": None,
    "abnormal_ratio": None,
    "window_verdicts": None,
    "window_reasonings": None,
    "window_slices": None,
    "total_slices": None,
    "slices_processed": None,
    "coverage_pct": None,
    "inference_time": None,
}


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/process", tags=["Processing"])
async def process(files: List[UploadFile] = File(...)):
    """
    Accept one or more ZIP archives with CT studies,
    return screening results as JSON.

    Every result dict has the same set of keys.
    If the study is invalid, inference keys are None.
    """
    all_results = []

    for file in files:
        file_content = await file.read()
        series_data, error_message = parse_zip_archive(file_content)

        if not series_data or error_message:
            all_results.append({
                **_EMPTY_RESULT,
                "archive_name": file.filename,
                "series_uid": error_message or "Parsing error",
            })
            continue

        for series_uid, data in series_data.items():
            meta = data["meta"]
            validation_checks = validate_series(meta)
            is_valid = all(check["status"] for check in validation_checks)

            entry = {
                **_EMPTY_RESULT,
                "archive_name": file.filename,
                "series_uid": series_uid,
                "source_format": meta.get("SourceFormat", "N/A"),
                "modality": meta.get("Modality", "N/A"),
                "body_part": meta.get("BodyPartExamined", "N/A"),
                "num_frames": meta.get("num_frames", 0),
                "is_valid": is_valid,
            }

            if is_valid and len(data["frames"]) > 0:
                result = model.run(data["frames"])
                entry.update({
                    "verdict": result["verdict"],
                    "abnormal_ratio": result["abnormal_ratio"],
                    "window_verdicts": result["window_verdicts"],
                    "window_reasonings": result["window_reasonings"],
                    "window_slices": result["window_slices"],
                    "total_slices": result["total_slices"],
                    "slices_processed": result["slices_processed"],
                    "coverage_pct": result["coverage_pct"],
                    "inference_time": f"{result['inference_time']:.1f}s",
                })

            all_results.append(entry)

    return {"results": all_results}
