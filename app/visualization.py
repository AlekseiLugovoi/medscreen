import io
import numpy as np
import imageio
import streamlit as st


def apply_ct_window(volume: np.ndarray, center: int, width: int) -> np.ndarray:
    """Apply HU windowing to a CT volume."""
    lo, hi = center - width / 2, center + width / 2
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo + 1e-6)
    return volume


def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    """Convert normalized (0-1) array to 8-bit (0-255)."""
    volume = np.clip(volume, 0, 1)
    return (volume * 255).astype(np.uint8)


@st.cache_data(show_spinner="Creating animation...")
def create_gif(_frames: list, series_uid: str, window_name: str,
               max_frames: int = 100, duration_ms: int = 50) -> bytes:
    """Create a GIF animation from a list of uint8 frames.

    Subsamples to max_frames if needed. Cache key: (series_uid, window_name).
    """
    frames = _frames
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [_frames[int(i * step)] for i in range(max_frames)]

    with io.BytesIO() as buffer:
        imageio.mimsave(buffer, frames, format='GIF', duration=duration_ms, loop=0)
        return buffer.getvalue()


@st.cache_data(show_spinner="Preparing frames...")
def prepare_frames_for_display(_series_data: dict, series_uid: str,
                               window_name: str, ct_windows: dict) -> list:
    """Apply CT window and convert frames to uint8 for display.

    Cache key: (series_uid, window_name, ct_windows).
    """
    raw_frames = _series_data["frames"]
    meta = _series_data["meta"]

    if meta.get("Modality") == "CT":
        center, width = ct_windows[window_name]
        processed_frames = apply_ct_window(raw_frames, center, width)
    else:
        lo, hi = raw_frames.min(), raw_frames.max()
        processed_frames = (raw_frames - lo) / (hi - lo + 1e-6)

    uint8_frames = normalize_to_uint8(processed_frames)
    return [frame for frame in uint8_frames]
