"""
OrthoScan AI — FastAPI Web Application
Orthopedic Knee Radiography Pathology Detector (ONNX Inference Engine)
Deployable on Render (free tier) with automatic demo-mode fallback.
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
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orthoscan")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "ortho_knee_model.onnx")
METADATA_PATH   = os.environ.get("METADATA_PATH",   "model_metadata.json")
DEMO_MODE       = not Path(ONNX_MODEL_PATH).exists()

if DEMO_MODE:
    logger.warning(
        "DEMO MODE active — ONNX model '%s' not found. "
        "Returning simulated predictions. Upload your model file to enable real inference.",
        ONNX_MODEL_PATH,
    )

# ---------------------------------------------------------------------------
# Load real predictor (only when model is present)
# ---------------------------------------------------------------------------
predictor = None
if not DEMO_MODE:
    try:
        from web_inference_onnx import KneeRadiographONNXPredictor
        predictor = KneeRadiographONNXPredictor(ONNX_MODEL_PATH, METADATA_PATH)
        logger.info("ONNX predictor loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load ONNX model: %s — switching to DEMO MODE.", exc)
        DEMO_MODE = True

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OrthoScan AI",
    description="Knee Radiography Pathology Detector — DenseNet-121 ONNX Inference",
    version="1.0.0",
)

BASE_DIR  = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Static files (only mount if directory exists)
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Demo-mode mock predictor
# ---------------------------------------------------------------------------
def _mock_predict(image_bytes: bytes) -> dict:
    """
    Deterministic mock based on image hash — same image always returns the same result.
    Simulates the full output schema of KneeRadiographONNXPredictor.predict().
    """
    digest   = hashlib.md5(image_bytes).hexdigest()
    seed_val = int(digest[:8], 16)
    rng      = random.Random(seed_val)

    prob      = round(rng.uniform(0.05, 0.95), 4)
    threshold = 0.50
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
        triage = (
            "High Priority — Orthopedic Review Required"
            if prob >= 0.70 else
            "Moderate Priority — Borderline Degeneration"
        )
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
        "raw_logit":                round((prob / (1 - prob + 1e-9)), 4),
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
    return {"status": "ok", "demo_mode": DEMO_MODE, "version": "1.0.0"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate content type
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Upload JPEG or PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > 20 * 1024 * 1024:   # 20 MB hard limit
        raise HTTPException(status_code=413, detail="File too large. Maximum 20 MB.")

    try:
        if DEMO_MODE or predictor is None:
            result = _mock_predict(image_bytes)
        else:
            result = predictor.predict(image_bytes)
            result["demo_mode"] = False
        return JSONResponse(content=result)

    except Exception as exc:
        logger.exception("Inference failed for file '%s': %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")


# ---------------------------------------------------------------------------
# Entry point (local dev)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
