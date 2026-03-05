# MedScreen

Chest CT screening: upload a study, get a verdict — **NORMAL** or **ABNORMAL**.

Powered by [MedGemma 1.5 4B](https://ai.google.dev/gemma/docs/medgemma) + [vLLM](https://github.com/vllm-project/vllm) structured output.

[Online Demo](https://dd62a182fa15b4f0a98aa4497baf2063f.clg07azjl.paperspacegradient.com/) · [Demo Data](https://drive.google.com/drive/folders/1ChmkPR-5OwZB8Ub9h23VuHoOiA2hX-gx?usp=sharing)

## Quick Start

```sh
git clone https://github.com/AlekseiLugovoi/medscreen.git
cd medscreen
echo "HF_TOKEN=hf_your_token_here" > .env
DOCKER_BUILDKIT=1 docker compose up --build
```

> Model weights download on first launch (~5 min). Subsequent starts are instant.

- **Web UI:** http://localhost:8501
- **REST API:** http://localhost:8502  · [Swagger docs](http://localhost:8502/docs)

## Usage

**curl:**
```bash
curl -X POST http://localhost:8502/process \
     -F "files=@normal_dicom_chest.zip" \
     -F "files=@abnormal_nifti_covid_ct4.zip"
```

**Python:**
```python
import requests

resp = requests.post(
    "http://localhost:8502/process",
    files=[
        ("files", ("normal_dicom_chest.zip", open("normal_dicom_chest.zip", "rb"), "application/zip")),
        ("files", ("abnormal_nifti_covid_ct4.zip", open("abnormal_nifti_covid_ct4.zip", "rb"), "application/zip")),
    ],
)

for r in resp.json()["results"]:
    print(f"{r['archive_name']}: {r['verdict']} ({r['abnormal_ratio']:.0%} abnormal)")
```

<details>
<summary>Example response</summary>

```json
{
  "results": [
    {
      "archive_name": "normal_dicom_chest.zip",
      "series_uid": "1.2.840.113704...",
      "source_format": "DICOM Series",
      "modality": "CT",
      "body_part": "CHEST",
      "num_frames": 451,
      "is_valid": true,
      "verdict": "NORMAL",
      "abnormal_ratio": 0.0,
      "window_verdicts": ["NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL"],
      "slices_processed": 100,
      "coverage_pct": 22.2,
      "inference_time": "35.5s"
    },
    {
      "archive_name": "abnormal_nifti_covid_ct4.zip",
      "series_uid": "NIfTI_study_0255.nii.gz",
      "source_format": "NIfTI",
      "modality": "NIFTI",
      "num_frames": 367,
      "is_valid": true,
      "verdict": "ABNORMAL",
      "abnormal_ratio": 0.4,
      "window_verdicts": ["NORMAL", "ABNORMAL", "NORMAL", "ABNORMAL", "NORMAL"],
      "slices_processed": 100,
      "coverage_pct": 27.2,
      "inference_time": "10.1s"
    }
  ]
}
```

</details>

## Supported Formats

Pack your data into a ZIP. Format is detected automatically:

- **DICOM** — `.dcm` files from one series, or a single multi-frame `.dcm`
- **NIfTI** — `.nii` / `.nii.gz` volumes (each file screened independently)
- **Images** — `.png` / `.jpg` files (treated as one series)

## Setup Details

<details>
<summary>System requirements</summary>

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, 8 GB VRAM | RTX 3080/4080+ (12 GB+) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 20 GB |

</details>

<details>
<summary>NVIDIA Container Toolkit (GPU)</summary>

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

</details>

<details>
<summary>Local development (without Docker)</summary>

```sh
conda create -n medscreen python=3.11 --yes
conda activate medscreen
pip install -r requirements.txt
```

Start the API backend (loads the model):
```sh
CUDA_VISIBLE_DEVICES=0 uvicorn app.api:app --port 8502
```

Start the Streamlit frontend (separate terminal):
```sh
python -m streamlit run app/main.py --server.port 8501
```

> The API works standalone — Streamlit is optional.

</details>

<details>
<summary>Architecture</summary>

```
┌─────────────┐       HTTP        ┌──────────────────────┐
│  Streamlit   │ ───────────────→ │  FastAPI + CTScreener │
│  (UI, :8501) │ ←─────────────── │  (model, :8502)      │
└─────────────┘    JSON result    └──────────────────────┘
       ↑                                    ↑
    browser                          curl / pipelines
```

```
medscreen/
├── app/
│   ├── main.py             # Streamlit entry point
│   ├── pages.py            # UI pages (preview, batch, API)
│   ├── api.py              # FastAPI backend
│   ├── ml_inference.py     # CTScreener (MedGemma + vLLM)
│   ├── ml_processing.py    # HTTP client to API
│   ├── file_io.py          # ZIP parsing (DICOM, NIfTI, PNG)
│   ├── data_validation.py  # Series metadata validation
│   └── visualization.py    # CT windowing, GIF animation
├── dev/                    # Research notebooks & experiments
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

</details>
