"""
========================================================================================
STANDALONE WEB INFERENCE ENGINE (ZERO PYTORCH / ZERO TRAINING DATA REQUIRED)
Model: Orthopedic Knee Radiography Pathology Detector (ONNX Engine)
Frameworks Supported: FastAPI, Flask, Streamlit, AWS Lambda, Docker, or Desktop GUI
========================================================================================
"""

import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

class KneeRadiographONNXPredictor:
    def __init__(self, onnx_model_path="ortho_knee_model.onnx", metadata_path="model_metadata.json"):
        """
        Initializes the lightweight ONNX runtime inference session.
        Requires NO PyTorch, NO GPU, and NO training data!
        """
        # Resolve ONNX path - cwd-relative first, then script-relative
        if not os.path.exists(onnx_model_path):
            _here = os.path.dirname(os.path.abspath(__file__))
            _found = False
            for _fb in [
                os.path.join("exports_model", "ortho_knee_densenet121.onnx"),
                os.path.join(_here, "exports_model", "ortho_knee_densenet121.onnx"),
            ]:
                if os.path.exists(_fb):
                    onnx_model_path = _fb
                    _found = True
                    break
            if not _found:
                raise FileNotFoundError(
                    "ONNX model not found. Place it at exports_model/ortho_knee_densenet121.onnx"
                )

        # Resolve metadata path - cwd-relative first, then script-relative
        if not os.path.exists(metadata_path):
            _here = os.path.dirname(os.path.abspath(__file__))
            for _fb_m in [
                os.path.join("exports_model", "model_metadata.json"),
                os.path.join(_here, "exports_model", "model_metadata.json"),
            ]:
                if os.path.exists(_fb_m):
                    metadata_path = _fb_m
                    break
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "optimal_threshold": 0.5,
                "input_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "classes": {
                    "0": "Normal Joint Architecture",
                    "1": "Osteoporosis / Degenerative Knee Pathology"
                }
            }
            
        self.threshold = float(self.metadata.get("optimal_threshold", 0.5))
        self.input_size = tuple(self.metadata.get("input_size", [224, 224]))
        self.mean = np.array(self.metadata.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(self.metadata.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.classes = self.metadata.get("classes", {
            "0": "Normal Joint Architecture",
            "1": "Osteoporosis / Degenerative Knee Pathology"
        })
        
        # Initialize ONNX Runtime session
        # Supports CPUExecutionProvider by default; automatically uses CUDAExecutionProvider if GPU runtime is installed
        available_providers = ort.get_available_providers()
        preferred_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_model_path, providers=preferred_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"[ONNX Predictor Initialized] Loaded '{onnx_model_path}' with provider '{self.session.get_providers()[0]}'")
        print(f"  Decision Threshold: {self.threshold:.4f} | Target Input Size: {self.input_size}")

    def preprocess(self, image_input):
        """
        Pure NumPy & PIL preprocessing (Identical to training pipeline).
        Accepts: file path string, byte stream, or PIL Image.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, (bytes, bytearray)):
            import io
            image = Image.open(io.BytesIO(image_input)).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise TypeError("Unsupported image input type. Provide a file path, bytes, or PIL Image.")
            
        # 0. Clinical Alignment: Auto-orient to vertical AP position (Height >= Width)
        if image.size[0] > image.size[1]:
            image = image.transpose(Image.Transpose.ROTATE_90)
            
        # 0b. Knee Joint ROI Cropping: Remove peripheral black border padding
        gray = np.array(image.convert('L'))
        non_pad = (gray > 15)
        if non_pad.sum() >= (gray.size * 0.10):
            y_proj = np.any(non_pad, axis=1)
            x_proj = np.any(non_pad, axis=0)
            y_indices = np.where(y_proj)[0]
            x_indices = np.where(x_proj)[0]
            if len(y_indices) > 0 and len(x_indices) > 0:
                h, w = gray.shape
                y_min = max(0, y_indices[0] - int(0.02 * h))
                y_max = min(h, y_indices[-1] + int(0.02 * h))
                x_min = max(0, x_indices[0] - int(0.02 * w))
                x_max = min(w, x_indices[-1] + int(0.02 * w))
                if (x_max - x_min) >= 30 and (y_max - y_min) >= 30:
                    image = image.crop((x_min, y_min, x_max, y_max))
            
        # 1. Bilinear resize to 224x224
        image = image.resize(self.input_size, Image.Resampling.BILINEAR)
        
        # 2. Convert to float array normalized to [0, 1]
        img_np = np.array(image, dtype=np.float32) / 255.0  # Shape: (H, W, 3)
        
        # 3. Standardize with ImageNet Mean & Std
        img_norm = (img_np - self.mean) / self.std
        
        # 4. Transpose from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        img_chw = np.transpose(img_norm, (2, 0, 1))  # Shape: (3, 224, 224)
        
        # 5. Add Batch Dimension NCHW: (1, 3, 224, 224)
        input_tensor = np.expand_dims(img_chw, axis=0).astype(np.float32)
        return input_tensor

    def predict(self, image_input, custom_threshold=None):
        """
        Executes ONNX forward pass and computes pathology probability.
        Includes uncertainty zone flagging (DOI:10.1007/s10278-024-01369-3):
        If probability is within ±uncertainty_band of threshold, flags as UNCERTAIN.
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        # Read uncertainty band from metadata (default 0.08 per literature recommendation)
        uncertainty_band = float(self.metadata.get('uncertainty_band', 0.08))
        input_tensor = self.preprocess(image_input)
        
        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        raw_logit = float(outputs[0][0])
        
        # Numerically stable Sigmoid: 1 / (1 + exp(-z))
        if raw_logit >= 0:
            prob = 1.0 / (1.0 + np.exp(-raw_logit))
        else:
            prob = np.exp(raw_logit) / (1.0 + np.exp(raw_logit))
            
        is_pathology = (prob >= threshold)
        confidence = prob if is_pathology else (1.0 - prob)

        # Uncertainty zone flag (DOI:10.1007/s10278-024-01369-3)
        # Cases within ±uncertainty_band of threshold have highest human-model disagreement
        in_uncertainty_zone = abs(prob - threshold) <= uncertainty_band
        if in_uncertainty_zone:
            predicted_label  = "UNCERTAIN — Refer to Radiologist"
            clinical_action  = (f"Probability {prob:.3f} is within ±{uncertainty_band:.2f} of "
                                f"threshold ({threshold:.3f}). Manual expert review required.")
        elif is_pathology:
            predicted_label  = self.classes["1"]
            clinical_action  = "Recommend DXA scan confirmation and clinical assessment."
        else:
            predicted_label  = self.classes["0"]
            clinical_action  = "No immediate action. Routine monitoring recommended."

        triage_urgency = "High Priority - Orthopedic Review Required" if prob >= 0.70 else \
                         ("Moderate Priority - Borderline Degeneration" if prob >= threshold else "Routine - Normal Architecture")
                         
        return {
            "prediction"              : predicted_label,
            "pathology_detected"      : bool(is_pathology),
            "osteoporosis_probability": round(float(prob), 4),
            "confidence"              : round(float(confidence), 4),
            "raw_logit"               : round(raw_logit, 4),
            "threshold_used"          : round(threshold, 4),
            "uncertainty_zone"        : bool(in_uncertainty_zone),
            "clinical_action"         : clinical_action,
            "clinical_triage"         : triage_urgency
        }


# ==============================================================================
# Quick Standalone Test Example
# ==============================================================================
if __name__ == "__main__":
    print("Standalone ONNX Inference Engine Ready.")
    print("Usage in FastAPI / Flask:")
    print("  predictor = KneeRadiographONNXPredictor('ortho_knee_model.onnx')")
    print("  result = predictor.predict('patient_knee_xray.png')")
