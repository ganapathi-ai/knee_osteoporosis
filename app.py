"""
OrthoScan AI — FastAPI Web Application
Orthopedic Knee Radiography Pathology Detector (ONNX Inference Engine)
Model: exports_model/ortho_knee_densenet121.onnx
"""

import os
import io
import json
import random
import hashlib
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orthoscan")

# ---------------------------------------------------------------------------
# Configuration — resolve paths relative to this file
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

ONNX_MODEL_PATH = os.environ.get(
    "ONNX_MODEL_PATH",
    str(BASE_DIR / "exports_model" / "ortho_knee_densenet121.onnx")
)
default_meta = BASE_DIR / "exports_model" / "model_metadata.json"

METADATA_PATH = os.environ.get(
    "METADATA_PATH",
    str(default_meta)
)

DEMO_MODE = not Path(ONNX_MODEL_PATH).exists()

if DEMO_MODE:
    logger.warning(
        "DEMO MODE — model not found at '%s'. Returning mock predictions.", ONNX_MODEL_PATH
    )
else:
    logger.info("Model found at '%s'", ONNX_MODEL_PATH)

# ---------------------------------------------------------------------------
# Load real ONNX predictor
# ---------------------------------------------------------------------------
predictor = None
if not DEMO_MODE:
    try:
        from web_inference_onnx import KneeRadiographONNXPredictor
        predictor = KneeRadiographONNXPredictor(ONNX_MODEL_PATH, METADATA_PATH)
        logger.info("ONNX predictor loaded. Threshold: %.4f", predictor.threshold)
    except Exception as exc:
        logger.error("Failed to load ONNX model: %s — switching to DEMO MODE.", exc)
        DEMO_MODE = True

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OrthoScan AI",
    description="Knee Radiography Pathology Detector — DenseNet-121 ONNX",
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------------------------------------------------------------------------
# Demo-mode mock (deterministic via image hash)
# ---------------------------------------------------------------------------
def _mock_predict(image_bytes: bytes) -> dict:
    digest   = hashlib.md5(image_bytes).hexdigest()
    rng      = random.Random(int(digest[:8], 16))
    prob     = round(rng.uniform(0.05, 0.95), 4)
    threshold        = 0.5000
    uncertainty_band = 0.08
    in_uncertainty   = abs(prob - threshold) <= uncertainty_band
    is_pathology     = prob >= threshold

    if in_uncertainty:
        prediction      = "UNCERTAIN — Refer to Radiologist"
        clinical_action = (
            f"Probability {prob:.3f} is within ±{uncertainty_band:.2f} of "
            f"threshold ({threshold:.3f}). Manual expert review required."
        )
        triage = "Borderline — Expert Review Required"
    elif is_pathology:
        prediction      = "Osteoporosis / Degenerative Knee Pathology"
        clinical_action = "Recommend DXA scan confirmation and clinical assessment."
        triage = "High Priority — Orthopedic Review Required" if prob >= 0.70 else "Moderate Priority — Borderline Degeneration"
    else:
        prediction      = "Normal Joint Architecture"
        clinical_action = "No immediate action. Routine monitoring recommended."
        triage          = "Routine — Normal Architecture"

    confidence = round(prob if is_pathology else (1.0 - prob), 4)
    return {
        "prediction":               prediction,
        "pathology_detected":       bool(is_pathology),
        "osteoporosis_probability": prob,
        "confidence":               confidence,
        "raw_logit":                round(prob / max(1e-9, 1 - prob), 4),
        "threshold_used":           threshold,
        "uncertainty_zone":         bool(in_uncertainty),
        "clinical_action":          clinical_action,
        "clinical_triage":          triage,
        "demo_mode":                True,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "demo_mode": DEMO_MODE},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "demo_mode": DEMO_MODE,
        "model_path": ONNX_MODEL_PATH,
        "model_exists": Path(ONNX_MODEL_PATH).exists(),
        "version": "1.0.0",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(415, detail=f"Unsupported file type: {file.content_type}.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, detail="Uploaded file is empty.")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, detail="File too large. Max 20 MB.")

    try:
        if DEMO_MODE or predictor is None:
            result = _mock_predict(image_bytes)
        else:
            result = predictor.predict(image_bytes)
            result["demo_mode"] = False
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(500, detail=f"Inference error: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
