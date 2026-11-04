# OrthoScan AI — Knee Osteoporosis Detection

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ganapathi-ai/knee_osteoporosis)

> **Research use only.** This tool does not constitute medical advice and must not be used for clinical diagnosis.

## Overview

OrthoScan AI is a minimal, professional web interface for an orthopedic knee radiography pathology classifier. It runs a **DenseNet-121** model (exported to ONNX) that classifies knee X-rays as:

- ✅ **Normal Joint Architecture**
- 🔴 **Osteoporosis / Degenerative Knee Pathology**
- ⚠️ **UNCERTAIN** — cases within ±0.08 of the decision threshold (flagged for radiologist review per [DOI:10.1007/s10278-024-01369-3](https://doi.org/10.1007/s10278-024-01369-3))

The backend uses **FastAPI + ONNX Runtime** (no PyTorch required at inference time). Without a model file, the app runs in **Demo Mode** — returning deterministic hash-based mock predictions so the UI is always functional.

---

## Architecture

| Component | Technology |
|---|---|
| Backend | FastAPI 0.115 + Uvicorn |
| Inference | ONNX Runtime (CPU / CUDA auto-select) |
| Preprocessing | NumPy + Pillow (identical to training pipeline) |
| Frontend | Vanilla HTML/CSS/JS — no build step |
| Fonts | Inter + JetBrains Mono (Google Fonts) |
| Deployment | Render (free tier) |

---

## Model

- **Architecture:** DenseNet-121 (transfer learning from ImageNet)
- **Task:** Binary classification — Normal vs Osteoporosis/Degenerative pathology
- **Input:** 224×224 RGB, ImageNet normalisation
- **Decision threshold:** Youden's J optimised (default 0.50, stored in `model_metadata.json`)
- **Uncertainty zone:** ±0.08 band around threshold → flagged for human review
- **Triage levels:** Routine / Moderate Priority / High Priority — Orthopedic Review Required

---

## Local Setup

```bash
# Clone
git clone https://github.com/ganapathi-ai/knee_osteoporosis.git
cd knee_osteoporosis

# Install dependencies (Python 3.10+)
pip install -r requirements.txt

# Run (demo mode — no model needed)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser.

### With real ONNX model

Place `ortho_knee_model.onnx` (and optionally `model_metadata.json`) in the project root, then start the server. The app auto-detects the model file.

```bash
# Optional: set custom paths
export ONNX_MODEL_PATH=/path/to/model.onnx
export METADATA_PATH=/path/to/model_metadata.json
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Deploy on Render

### One-click deploy

Click the **Deploy to Render** button at the top of this README.

### Manual deploy

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your `ganapathi-ai/knee_osteoporosis` repository
4. Render auto-detects `render.yaml` — click **Deploy**

**Note:** The `.onnx` model file is excluded from git (`.gitignore`) due to file size. After deploying, use **Render Disks** or an environment variable pointing to cloud storage (e.g. Hugging Face Hub, S3) to load the model. The app runs in Demo Mode otherwise.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/predict` | Upload image → JSON prediction |
| `GET` | `/health` | Health check + demo mode status |
| `GET` | `/docs` | Auto-generated OpenAPI docs |

### POST /predict — Response schema

```json
{
  "prediction":               "Normal Joint Architecture",
  "pathology_detected":       false,
  "osteoporosis_probability": 0.1823,
  "confidence":               0.8177,
  "raw_logit":               -1.5021,
  "threshold_used":           0.5000,
  "uncertainty_zone":         false,
  "clinical_action":          "No immediate action. Routine monitoring recommended.",
  "clinical_triage":          "Routine — Normal Architecture",
  "demo_mode":                false
}
```

---

## Project Structure

```
knee_osteoporosis/
├── app.py                    # FastAPI application
├── web_inference_onnx.py     # ONNX inference engine
├── ortho_knee_pipeline.py    # Training pipeline (DenseNet-121)
├── templates/
│   └── index.html            # Web UI (self-contained)
├── requirements.txt
├── render.yaml               # Render deployment config
└── README.md
```

---

## Scientific Methodology Audit & Critical Findings

A rigorous review of the pipeline and notebook identified four critical methodological considerations:

1. **Zero Test-Leakage Thresholding:**
   - The previously reported `0.2585` threshold was selected directly on the test set labels, artificially producing 100% test sensitivity/specificity through test data leakage.
   - When evaluated honestly at the unbiased standard threshold (**0.50**) or via cross-validation thresholds, the out-of-sample test accuracy is **99.44%**.
2. **Expected Calibration Error (ECE) Correction:**
   - An initial report of ECE = `0.7247` was traced to a formula bug that compared class accuracy against $P(\text{osteoporosis})$ rather than model confidence $\max(p, 1-p)$.
   - With the correct Guo et al. (ICML 2017) formulation, the true ECE is **< 0.05**, demonstrating excellent clinical calibration.
3. **Dynamic Explainability vs. Static Captions:**
   - Grad-CAM++ visualizations now compute spatial activation centroids and energy concentration dynamically, including automated alerts if saliency focuses on scanner border margins rather than articular bone.
4. **Cohort Scope & Multi-Center Validation:**
   - The test set consists of **164 unique patient clusters**. Single-center perfection must be interpreted with caution. External validation on public multi-center cohorts (e.g., Osteoarthritis Initiative [OAI], n=4,796) remains essential before clinical translation.

---

## References

| Paper | Venue | Link |
|---|---|---|
| Opportunistic Bone-Loss Screening (STR-Net) | arXiv 2025 | [2604.20268](https://arxiv.org/abs/2604.20268) |
| MC Dropout Uncertainty — Vertebral Metastasis | J. Imaging Inform. Med. 2024 | [10.1007/s10278-024-01369-3](https://doi.org/10.1007/s10278-024-01369-3) |
| Mixup: Beyond Empirical Risk Minimization | ICLR 2018 | [1710.09412](https://arxiv.org/abs/1710.09412) |
| Grad-CAM++ | WACV 2018 | [1710.11063](https://arxiv.org/abs/1710.11063) |

---

## Disclaimer

This software is intended **solely for research and educational purposes**. It has not been validated for clinical use and must not be used to make or inform clinical decisions. Always consult a qualified radiologist.

