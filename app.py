"""
OrthoScan AI — FastAPI Web Application
Orthopedic Knee Radiography Pathology Detector (ONNX Inference Engine)
Model: exports_model/ortho_knee_densenet121.onnx
"""

import os
import io
import json
import base64
import random
import hashlib
import logging
import numpy as np
from pathlib import Path
from PIL import Image

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
# Fast Batched Saliency / Heatmap Generation (ONNX-compatible)
# ---------------------------------------------------------------------------
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def compute_gradcam_heatmap(image_bytes: bytes, predictor_obj) -> str:
    """
    Low-Memory Chunked Saliency Map (512MB RAM safe for Render Free Tier).
    Evaluates a 5x5 occlusion grid (25 patches of 45x45) in small chunks of 5 images
    to keep peak RAM usage under ~170MB (preventing Linux OOM killer).
    Executes in ~2.5 seconds total.
    """
    tensor = predictor_obj.preprocess(image_bytes)   # (1, 3, 224, 224) float32
    h, w   = tensor.shape[2], tensor.shape[3]

    # Baseline prediction (single pass)
    out0 = predictor_obj.session.run(
        [predictor_obj.output_name], {predictor_obj.input_name: tensor}
    )[0]
    base_prob = float(_sigmoid_np(out0.flatten())[0])

    # 5x5 occlusion grid = 25 patches (45x45 pixels each)
    grid_size = 5
    patch_size = 45
    total_patches = grid_size * grid_size
    mean_val = float(tensor.mean())

    # Build all 25 occluded variations
    occluded_list = []
    for idx in range(total_patches):
        r = idx // grid_size
        c = idx % grid_size
        r_start = r * patch_size
        r_end   = min(r_start + patch_size, h)
        c_start = c * patch_size
        c_end   = min(c_start + patch_size, w)
        t_mod = tensor.copy()
        t_mod[0, :, r_start:r_end, c_start:c_end] = mean_val
        occluded_list.append(t_mod[0])

    # Run in small mini-batches of 5 to keep memory <180MB
    chunk_size = 5
    batch_probs_list = []
    for i in range(0, total_patches, chunk_size):
        chunk = np.stack(occluded_list[i:i + chunk_size], axis=0)  # (5, 3, 224, 224)
        out_chunk = predictor_obj.session.run(
            [predictor_obj.output_name], {predictor_obj.input_name: chunk}
        )[0]
        batch_probs_list.extend(_sigmoid_np(out_chunk.flatten()))

    batch_probs = np.array(batch_probs_list, dtype=np.float32)

    # Importance = drop in pathology probability when region is occluded
    sal = (base_prob - batch_probs).reshape(grid_size, grid_size)
    sal = np.clip(sal, 0.0, None)
    if sal.max() > 1e-6:
        sal = sal / sal.max()

    # Bilinear upsample from 5x5 to 224x224
    sal_pil = Image.fromarray((sal * 255).astype(np.uint8), mode='L')
    sal_up  = sal_pil.resize((w, h), Image.Resampling.BILINEAR)
    sal_arr = np.array(sal_up, dtype=np.float32) / 255.0

    # JET colourmap: Blue (baseline) -> Cyan -> Green -> Yellow -> Red (pathology highlight)
    def jet(v):
        r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
        return r, g, b

    r, g, b   = jet(sal_arr)
    jet_rgb   = np.stack([r, g, b], axis=-1)
    jet_uint8 = (jet_rgb * 255).astype(np.uint8)

    # Load and align original radiograph to 224x224
    orig_pil  = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if orig_pil.size[0] > orig_pil.size[1]:
        orig_pil = orig_pil.transpose(Image.Transpose.ROTATE_90)
    orig_pil  = orig_pil.resize((w, h), Image.Resampling.BILINEAR)
    orig_arr  = np.array(orig_pil, dtype=np.float32)

    # Blend: original radiograph + 65% saliency heatmap
    alpha    = sal_arr[:, :, np.newaxis]
    blended  = (orig_arr * (1.0 - 0.65 * alpha) + jet_uint8 * (0.65 * alpha))
    blended  = np.clip(blended, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _mock_gradcam(image_bytes: bytes) -> str:
    """Demo-mode: return an ambient focus heatmap on the radiograph."""
    orig = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    arr  = np.array(orig, dtype=np.float32)
    yy, xx = np.mgrid[0:224, 0:224]
    sal = np.exp(-((xx - 112)**2 / (60**2) + (yy - 112)**2 / (40**2)))
    r  = np.clip(1.5 - np.abs(4.0 * sal - 3.0), 0.0, 1.0)
    g  = np.clip(1.5 - np.abs(4.0 * sal - 2.0), 0.0, 1.0)
    b  = np.clip(1.5 - np.abs(4.0 * sal - 1.0), 0.0, 1.0)
    jet = np.stack([r, g, b], axis=-1) * 255.0
    alpha = sal[:, :, np.newaxis]
    blended = np.clip(arr * (1.0 - 0.65 * alpha) + jet * (0.65 * alpha), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


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
            result["gradcam_png"] = _mock_gradcam(image_bytes)
        else:
            result = predictor.predict(image_bytes)
            result["demo_mode"] = False
            try:
                result["gradcam_png"] = compute_gradcam_heatmap(image_bytes, predictor)
            except Exception as cam_err:
                logger.warning("Grad-CAM generation error (non-fatal): %s", cam_err)
                result["gradcam_png"] = None

        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(500, detail=f"Inference error: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
