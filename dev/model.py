"""
КТ-скринер: инференс на базе MedGemma + vLLM.

Только инференс — предобработка в preprocessing.py.
"""

import os
import json
import time
from enum import Enum
import logging
import tempfile
from PIL import Image
from typing import Optional
from pydantic import BaseModel, constr, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

logger = logging.getLogger("medscreen.model")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [MODEL] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)


class PathologyType(str, Enum):
    covid = "covid"
    cancer = "cancer"
    abdominal = "abdominal"
    other = "other"

class CTVerdict(BaseModel):
    analysis: constr(max_length=1000)
    is_abnormal: bool
    confidence: int = Field(ge=1, le=3)
    pathology_type: Optional[PathologyType] = None


DEFAULT_PROMPT = """\
Role: Expert CT radiologist.
Task: Screen consecutive CT slices for ANY pathological findings.
Rules: Ignore normal anatomy, age-related changes, and artifacts (noise, motion, hardware).

Output ONLY raw JSON (no markdown):
{
  "analysis": "<Brief step-by-step reasoning>",
  "is_abnormal": <true/false (true ONLY for actual pathology)>,
  "confidence": <integer (1: Low/Artifact, 2: Probable, 3: High/Definite)>
}"""


class CTScreener:
    """
    КТ-скринер на базе MedGemma + vLLM.

    Только инференс: принимает готовые группы PIL-картинок,
    прогоняет через модель, возвращает вердикты.

    Использование:
      from preprocessing import prepare_windows, extract_planes
      from model import CTScreener

      screener = CTScreener()
      groups = prepare_windows(volume, center=-500, width=1000)
      result = screener.run(groups, prompt="...")
    """

    def __init__(
        self,
        model: str = "google/medgemma-1.5-4b-it",
        data_dir: str = "/home/a-lugovoi/Git/medscreen/dev/",
        gpu_memory: float = 0.7,
        max_model_len: int = 8192,
        max_tokens: int = 512,
    ):
        self.data_dir = data_dir

        logger.info(f"Загрузка модели '{model}'...")
        t0 = time.time()

        self.llm = LLM(
            model=model,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory,
            dtype="bfloat16",
            allowed_local_media_path=data_dir,
            disable_log_stats=True,
        )

        self.sampling = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(
                json=CTVerdict.model_json_schema()
            ),
        )

        logger.info(f"Модель загружена за {time.time() - t0:.1f}с")

    def run(
        self,
        image_groups: list[list[Image.Image]],
        prompt: str = None,
    ) -> dict:
        """
        Инференс на готовых группах PIL-картинок.

        Args:
            image_groups: список групп, каждая группа — list[PIL.Image].
            prompt: промпт для модели (если None — DEFAULT_PROMPT).

        Returns:
            dict: abnormal_ratio, verdicts, analyses,
                  reasonings, inference_time.
        """
        nw = len(image_groups)
        all_messages = []

        with tempfile.TemporaryDirectory(dir=self.data_dir) as tmpdir:
            for w, images in enumerate(image_groups):
                image_urls = []
                for i, img in enumerate(images):
                    path = os.path.join(tmpdir, f"w{w}_s{i}.png")
                    img.save(path)
                    image_urls.append({
                        "type": "image_url",
                        "image_url": {"url": f"file://{path}"},
                    })

                content = image_urls + [{"type": "text", "text": prompt or DEFAULT_PROMPT}]
                all_messages.append([{"role": "user", "content": content}])

            t0 = time.time()
            outputs = self.llm.chat(
                all_messages,
                sampling_params=self.sampling,
                use_tqdm=False,
            )
            elapsed = time.time() - t0

        is_abnormal_list = []
        confidences = []
        analyses = []
        pathology_types = []
        for out in outputs:
            try:
                data = json.loads(out.outputs[0].text)
                is_abn = bool(data.get("is_abnormal", False))
                conf = max(1, min(3, int(data.get("confidence", 1))))
                analyses.append(data.get("analysis", ""))
                pathology_types.append(data.get("pathology_type"))
            except (json.JSONDecodeError, ValueError):
                is_abn, conf = False, 1
                analyses.append("JSON parse error")
                pathology_types.append(None)
            is_abnormal_list.append(is_abn)
            confidences.append(conf)

        n_abn = sum(is_abnormal_list)
        abnormal_ratio = n_abn / float(nw) if nw > 0 else 0.0

        logger.info(f"ABNORMAL: {n_abn}/{nw} ({abnormal_ratio:.0%}) | {elapsed:.0f}s")

        return {
            "abnormal_ratio": abnormal_ratio,
            "is_abnormal": is_abnormal_list,
            "confidences": confidences,
            "analyses": analyses,
            "pathology_types": pathology_types,
            "inference_time": elapsed,
        }
