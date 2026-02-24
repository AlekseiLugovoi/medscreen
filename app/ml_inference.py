import os
import json
import time
import logging
import tempfile
import numpy as np
from PIL import Image
from typing import Literal
from pydantic import BaseModel, constr
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


# --- Логирование ---
logger = logging.getLogger("medscreen.model")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [MODEL] %(levelname)s: %(message)s"))
    logger.addHandler(handler)


# --- Схема structured output ---
class CTVerdict(BaseModel):
    reasoning: constr(max_length=150)
    verdict: Literal["NORMAL", "ABNORMAL"]


# --- Промпты ---
SYSTEM_PROMPT = (
    "You are an expert chest CT radiologist. "
    "Analyze the provided CT slices carefully. "
    "Be concise and precise in your reasoning."
)

USER_PROMPT_TEMPLATE = """\
Review these {n_images} consecutive axial chest CT slices
(slices {start}-{end} of {total} total).

Look carefully for ANY pathological findings, for example:
ground-glass opacity, consolidation, pleural effusion,
pneumothorax, atelectasis, nodules, masses,
or any other abnormality.

Important:
- Normal anatomical variants are NOT pathology
- Age-related changes alone are NOT pathology
- Report ABNORMAL only for true pathological findings

Decision rules:
- If ANY true pathology is present → verdict: ABNORMAL
- If lungs and structures are completely normal → verdict: NORMAL

Reasoning: one sentence, max 15 words, only what you see."""


class CTScreener:
    """
    Study-level КТ-скринер на базе MedGemma + vLLM.

    Подход:
      1. Объём делится на n_windows равных зон.
      2. Из центра каждой зоны берётся окно из window_size соседних срезов.
      3. Все окна батчатся в один вызов vLLM.
         Модель возвращает JSON: {"reasoning": "...", "verdict": "NORMAL/ABNORMAL"}.
      4. Если доля ABNORMAL-окон >= threshold — исследование патологическое.

    Для маленьких объёмов параметры автоподстраиваются:
      - n <= window_size → одно окно со всеми срезами
      - n < window_size * n_windows → увеличиваем кол-во окон
        с перекрытием, размер окна НЕ уменьшается
    """

    def __init__(
        self,
        model: str = "google/medgemma-1.5-4b-it",
        tmp_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp"),
        gpu_memory: float = 0.3,
        max_model_len: int = 8192,
    ):
        # Директория для временных файлов (vLLM требует file:// URL)
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

        logger.info(f"Загрузка модели '{model}'...")
        t0 = time.time()

        self.llm = LLM(
            model=model,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory,
            dtype="bfloat16",
            allowed_local_media_path=tmp_dir,
            disable_log_stats=True,
        )

        self.sampling = SamplingParams(
            temperature=0,
            max_tokens=256,
            structured_outputs=StructuredOutputsParams(
                json=CTVerdict.model_json_schema()
            ),
        )

        logger.info(f"Модель загружена за {time.time() - t0:.1f}с")

    @staticmethod
    def hu_to_pil(
        slice_2d: np.ndarray,
        center: float = -600,
        width: float = 1500,
    ) -> Image.Image:
        """
        HU → 8-bit RGB PIL. Если данные уже 0-255 (PNG), пропускаем windowing.

        Стандартные HU-окна для КТ грудной клетки:
          Lung:        center=-600, width=1500  (лёгочная ткань, основные патологии)
          Narrow lung: center=-500, width=1000  (subtle GGO)
          Mediastinal: center=40,   width=350   (мягкие ткани, сосуды, выпот)
          Bone:        center=300,  width=2000  (кости, кальцинаты)
        """
        if slice_2d.max() <= 255 and slice_2d.min() >= 0:
            img = slice_2d.astype(np.uint8)
        else:
            lo, hi = center - width / 2, center + width / 2
            s = np.clip(slice_2d, lo, hi)
            img = ((s - lo) / (hi - lo) * 255).astype(np.uint8)
        return Image.fromarray(img).convert("RGB")

    def _build_windows(self, n: int, window_size: int, n_windows: int):
        """
        Рассчитывает параметры окон с автоподстройкой.

        Возвращает (window_size, n_windows, list of (start, end)).
        """
        ws = window_size
        nw = n_windows

        if n <= ws:
            ws = n
            nw = 1
        elif n < ws * nw:
            nw = -(-n // ws)  # ceil(n / ws)

        zone_size = n // nw
        windows = []
        for w in range(nw):
            center = zone_size * w + zone_size // 2
            start = max(0, center - ws // 2)
            end = min(n, start + ws)
            start = max(0, end - ws)
            windows.append((start, end))

        return ws, nw, windows

    def run(
        self,
        volume: np.ndarray,
        center: float = -600,
        width: float = 1500,
        window_size: int = 20,
        n_windows: int = 5,
        threshold: float = 0.2,
    ) -> dict:
        """
        Запуск инференса на КТ-объёме.

        Args:
            volume: np.ndarray shape (slices, H, W) — КТ-объём.
            center: Уровень окна (HU) для визуализации.
            width: Ширина окна (HU) для визуализации.
            window_size: Базовый размер окна.
            n_windows: Базовое количество окон.
            threshold: Порог для диагноза ABNORMAL.

        Returns:
            dict с ключами:
              verdict, abnormal_ratio, window_verdicts, window_reasonings,
              window_slices, total_slices, n_windows, window_size,
              slices_processed, coverage_pct, inference_time.
        """
        n = volume.shape[0]
        ws, nw, windows = self._build_windows(n, window_size, n_windows)

        all_messages = []
        window_meta = []

        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as tmpdir:
            for w, (start, end) in enumerate(windows):
                image_urls = []
                for i in range(start, end):
                    img = self.hu_to_pil(volume[i], center=center, width=width)
                    path = os.path.join(tmpdir, f"w{w}_s{i}.png")
                    img.save(path)
                    image_urls.append({
                        "type": "image_url",
                        "image_url": {"url": f"file://{path}"},
                    })

                prompt = USER_PROMPT_TEMPLATE.format(
                    n_images=len(image_urls),
                    start=start,
                    end=end - 1,
                    total=n,
                )

                content = image_urls + [{"type": "text", "text": prompt}]
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ]
                all_messages.append(messages)
                window_meta.append((start, end - 1))

            t0 = time.time()
            outputs = self.llm.chat(
                all_messages,
                sampling_params=self.sampling,
                use_tqdm=False,
            )
            elapsed = time.time() - t0

        verdicts = []
        reasonings = []
        for out in outputs:
            try:
                data = json.loads(out.outputs[0].text)
                verdicts.append(data["verdict"])
                reasonings.append(data.get("reasoning", ""))
            except json.JSONDecodeError:
                verdicts.append("NORMAL")
                reasonings.append("JSON parse error")

        abnormal_ratio = verdicts.count("ABNORMAL") / float(len(verdicts)) if len(verdicts) > 0 else 0.0
        final = "ABNORMAL" if abnormal_ratio >= threshold else "NORMAL"

        logger.info(
            f"Result: {final} | "
            f"ABNORMAL windows: {verdicts.count('ABNORMAL')}/{nw} ({abnormal_ratio:.0%}) | "
            f"Slices count: {nw * ws}/{n} ({nw * ws / n * 100:.0f}%) | "
            f"{elapsed:.1f}s"
        )

        return {
            "verdict": final,
            "abnormal_ratio": abnormal_ratio,
            "window_verdicts": verdicts,
            "window_reasonings": reasonings,
            "window_slices": window_meta,
            "total_slices": n,
            "n_windows": nw,
            "window_size": ws,
            "slices_processed": nw * ws,
            "coverage_pct": nw * ws / n * 100,
            "inference_time": elapsed,
        }
