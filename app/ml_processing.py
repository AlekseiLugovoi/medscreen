import os
import requests
import streamlit as st

API_URL = os.environ.get("MEDSCREEN_API_URL", "http://localhost:8502")


def check_api_health() -> bool:
    """Check if the API backend is available."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _call_api(zip_bytes: bytes, filename: str) -> list | None:
    """Send ZIP to API, return list of results."""
    try:
        resp = requests.post(
            f"{API_URL}/process",
            files={"files": (filename, zip_bytes, "application/zip")},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["results"]
    except requests.RequestException:
        return None


@st.cache_data(show_spinner="Running ML screening...")
def run_inference(zip_bytes: bytes, filename: str = "study.zip") -> dict:
    """Single study inference (for preview). Returns first result."""
    results = _call_api(zip_bytes, filename)
    return results[0] if results else None


@st.cache_data(show_spinner="Processing archive...")
def run_inference_batch(zip_bytes: bytes, filename: str = "study.zip") -> list:
    """Batch inference. Returns all results (one per study in ZIP)."""
    return _call_api(zip_bytes, filename)
