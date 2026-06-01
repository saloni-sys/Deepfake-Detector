from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Backend.model import lifespan, predict
from Backend.detector import preprocess_image, is_valid_image

# ── App setup ──────────────────────────────────────────────────────────────────
# lifespan loads ML model once at startup — kept from new version
# Everything else is your original structure, just fixed and extended

app = FastAPI(lifespan=lifespan)

# ── CORS ───────────────────────────────────────────────────────────────────────
# FIXED: moved outside any function — was inside detect_image before (never ran)
# This is your original CORS config, just in the right place now

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # your original setting
    allow_credentials=True,     # your original setting
    allow_methods=["*"],        # your original setting
    allow_headers=["*"],        # your original setting
)

# ── Response model ─────────────────────────────────────────────────────────────
# Your original DetectResponse + face_detected field added
# face_detected tells frontend whether a face was found in the image

class DetectResponse(BaseModel):
    label: str            # "fake", "real", or "unknown"
    confidence: float     # 0.0 to 1.0
    face_detected: bool   # NEW: was a face found?

# ── Constants ──────────────────────────────────────────────────────────────────
# Your original allowed types + webp added
# Your original 2MB limit kept — change to 10 if you want larger files

ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"]
MAX_SIZE_BYTES = 2 * 1024 * 1024   # 2MB — your original limit

# ── Root route ─────────────────────────────────────────────────────────────────
# Your original route kept exactly as-is

@app.get("/")
def read_root():
    return {"message": "Server is running"}

# ── Health check ───────────────────────────────────────────────────────────────
# Your original route kept exactly as-is

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ── Detect endpoint ────────────────────────────────────────────────────────────
# Your original structure kept — validation logic same as yours
# Added: corruption check, face detection, real model inference

@app.post("/detect", response_model=DetectResponse)
async def detect_image(file: UploadFile = File(...)):

    # Your original file type check — kept exactly
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Your original size check — kept exactly
    contents = await file.read()
    if len(contents) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large")

    # NEW: check image is not corrupted
    if not is_valid_image(contents):
        raise HTTPException(status_code=400, detail="Image is corrupted or unreadable")

    # NEW: preprocess + face detection
    try:
        tensor, face_detected = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    # NEW: no face found — return early, don't run model
    if not face_detected:
        return DetectResponse(
            label="unknown",
            confidence=0.0,
            face_detected=False
        )

    # NEW: run real model inference
    try:
        result = predict(tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # Return real result — same structure as your original dummy return
    return DetectResponse(
        label=result["label"],
        confidence=result["confidence"],
        face_detected=True
    )

# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)