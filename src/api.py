import cv2
import io
import numpy as np
import os
import re
import torch
import albumentations as A

from contextlib import asynccontextmanager
from omegaconf import OmegaConf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from skp import builder


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH     = os.getenv("CONFIG_PATH",      "configs/mks/mk000.yaml")
# If GCS_CHECKPOINT is set, the model is downloaded from Cloud Storage at startup.
# Otherwise falls back to a local path (useful for local dev / non-GCS deploys).
GCS_CHECKPOINT  = os.getenv("GCS_CHECKPOINT",   "")          # e.g. gs://my-bucket/gender-det/checkpoint.ckpt
# Default local path works both in Docker (/app/src → /app/experiments) and locally
LOCAL_CKPT_PATH = os.getenv("CHECKPOINT_PATH",  "../experiments/mk000/checkpoints/epoch=009-vm=1.8604.ckpt")
CHECKPOINT      = LOCAL_CKPT_PATH                             # resolved at startup

# ── Validation thresholds ─────────────────────────────────────────────────────
# Minimum softmax confidence to return a prediction.
# Below this → "low_confidence" response.
CONFIDENCE_THRESHOLD  = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
# Max allowed colour deviation across RGB channels (0–255 scale).
# Ultrasound images are grayscale; colour photos will exceed this.
GRAYSCALE_THRESHOLD   = float(os.getenv("GRAYSCALE_THRESHOLD",  "18.0"))


def _download_checkpoint_from_gcs(gcs_uri: str, dest: str):
    """Download checkpoint from GCS using the storage client library."""
    from google.cloud import storage as gcs
    # Parse gs://bucket/path/to/file.ckpt
    without_prefix = gcs_uri[len("gs://"):]
    bucket_name, blob_path = without_prefix.split("/", 1)
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading checkpoint from {gcs_uri} → {dest}")
    blob.download_to_filename(dest)
    print("Checkpoint download complete.")
USE_GPU        = os.getenv("USE_GPU", "0") == "1"
IMAGE_SIZE     = 512

LABEL_MAP = {
    0: "boy",
    1: "girl",
    2: "unable_to_assess",
    3: "text_says_boy_girl",
}

# ── Globals (populated at startup) ────────────────────────────────────────────

_model       = None
_preprocessor = None
_resizer      = None


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _preprocessor, _resizer

    # Download checkpoint from GCS if configured
    if GCS_CHECKPOINT:
        _download_checkpoint_from_gcs(GCS_CHECKPOINT, LOCAL_CKPT_PATH)

    ckpt_path = LOCAL_CKPT_PATH

    print(f"Loading config from  : {CONFIG_PATH}")
    print(f"Loading checkpoint   : {ckpt_path}")

    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.model.params.pretrained = False   # skip downloading pretrained weights

    # Build model architecture
    _model = builder.build_model(cfg)

    # Load checkpoint weights
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = {re.sub(r"^model\.", "", k): v for k, v in state["state_dict"].items()}
    _model.load_state_dict(weights)
    _model.eval()

    if USE_GPU and torch.cuda.is_available():
        _model = _model.cuda()
        print("Model running on GPU")
    else:
        print("Model running on CPU")

    # Build the same resize + preprocess transforms as training (no augmentation)
    resize_cfg    = cfg.transform.resize
    preprocess_cfg = cfg.transform.preprocess

    _resizer = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE, p=1),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, p=1,
                      border_mode=cv2.BORDER_CONSTANT, value=0),
    ], p=1)

    _preprocessor = builder.get_transform(cfg.transform, "preprocess")

    print("Model ready — API is up.")
    yield

    _model = None
    _preprocessor = None
    _resizer = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gender Determination API",
    description="Upload a fetal ultrasound image to get gender prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Decode image bytes and apply the same pipeline used during training."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)           # BGR, uint8, (H, W, 3)
    if img is None:
        raise ValueError("Could not decode image. Make sure it is a valid JPG/PNG.")

    # Resize (keep aspect ratio, pad to square)
    img = _resizer(image=img)["image"]                  # (512, 512, 3)

    # Normalize: [0,255] → [0,1] → subtract mean / divide sdev
    img = _preprocessor(img)                            # float32, (512, 512, 3)

    # (H, W, C) → (C, H, W) → add batch dim
    img = img.transpose(2, 0, 1)
    tensor = torch.tensor(img).float().unsqueeze(0)     # (1, 3, 512, 512)

    if USE_GPU and torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


def _is_ultrasound(image_bytes: bytes) -> tuple[bool, str]:
    """
    Reject obviously non-ultrasound images.
    Ultrasound images are grayscale — all 3 RGB channels carry nearly
    identical values. A colour photo (face, selfie, etc.) has large
    per-channel variance and will fail this check.
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return False, "Could not decode image."

    # Convert BGR → float32 for channel comparison
    img_f = img.astype(np.float32)
    rg_diff = float(np.mean(np.abs(img_f[:, :, 2] - img_f[:, :, 1])))  # R-G
    rb_diff = float(np.mean(np.abs(img_f[:, :, 2] - img_f[:, :, 0])))  # R-B

    if rg_diff > GRAYSCALE_THRESHOLD or rb_diff > GRAYSCALE_THRESHOLD:
        return False, (
            f"Image appears to be a colour photo (channel deviation "
            f"R-G={rg_diff:.1f}, R-B={rb_diff:.1f}). "
            "Please upload a fetal ultrasound image of the genitalia region."
        )
    return True, ""


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a fetal ultrasound image (JPG or PNG).
    Returns predicted gender, confidence score, and all class probabilities.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    # ── Validation 1: must look like a grayscale ultrasound ───────────────────
    valid, reason = _is_ultrasound(image_bytes)
    if not valid:
        return JSONResponse(status_code=422, content={
            "predicted_label": "invalid_image",
            "confidence": 0.0,
            "message": reason,
            "filename": file.filename,
        })

    # ── Run model ─────────────────────────────────────────────────────────────
    try:
        tensor = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (4,)

    predicted_class = int(probs.argmax())
    confidence      = float(probs.max())

    probabilities = {
        "boy":                round(float(probs[0]), 4),
        "girl":               round(float(probs[1]), 4),
        "unable_to_assess":   round(float(probs[2]), 4),
        "text_says_boy_girl": round(float(probs[3]), 4),
    }

    # ── Validation 2: confidence must be high enough ──────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        return JSONResponse(status_code=200, content={
            "predicted_label": "low_confidence",
            "confidence":      round(confidence, 4),
            "message": (
                f"Model confidence ({confidence:.0%}) is below threshold "
                f"({CONFIDENCE_THRESHOLD:.0%}). The image may not show the "
                "genitalia region clearly. Please try a different frame."
            ),
            "probabilities": probabilities,
            "filename": file.filename,
        })

    return JSONResponse({
        "predicted_label": LABEL_MAP[predicted_class],
        "confidence":      round(confidence, 4),
        "probabilities":   probabilities,
        "filename":        file.filename,
    })
