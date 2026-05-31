import torch
import torch.nn as nn
from torchvision import models
from contextlib import asynccontextmanager
from fastapi import FastAPI

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_PATH  = "../weights/efficientnet_b4_best.pth"  # path to your weights file
THRESHOLD   = 0.5                                   # above this = fake
NUM_CLASSES = 2                                     # real, fake

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global model variable ──────────────────────────────────────────────────────
# We store the model here so it stays in memory for the lifetime of the server.
# This means we load it ONCE at startup, not on every request.

model = None


# ── Step 1: Build model architecture ──────────────────────────────────────────

def build_model() -> nn.Module:
    """
    Loads EfficientNet-B4 architecture and replaces the final
    classification layer to output 2 classes (real vs fake).

    Why replace the final layer?
    EfficientNet was originally trained on 1000 ImageNet classes.
    We only need 2 (real/fake), so we swap the head.
    """
    # Load EfficientNet-B4 architecture (no pretrained weights yet)
    net = models.efficientnet_b4(weights=None)

    # Replace classifier head: original outputs 1000 classes → we need 2
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return net


# ── Step 2: Load weights ───────────────────────────────────────────────────────

def load_model() -> nn.Module:
    """
    Builds the model and loads pretrained deepfake detection weights.
    Called once at server startup via the lifespan function below.

    Where to get weights:
    Option A — Hugging Face:
        search 'deepfake detection efficientnet' on huggingface.co/models
    Option B — FaceForensics++ trained checkpoint:
        github.com/ondyari/FaceForensics
    Option C — Train your own (advanced, not needed for now)
    """
    global model

def load_model() -> nn.Module:
    global model

    net = build_model()

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    load_result = net.load_state_dict(state_dict, strict=False)
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)

    net = net.to(device)
    net.eval()

    model = net
    print(f"Model loaded successfully on {device}")
    return net


# ── Step 3: Run inference ──────────────────────────────────────────────────────

def predict(tensor: torch.Tensor) -> dict:
    """
    Takes a preprocessed tensor from detector.py
    and returns the prediction result.

    tensor shape expected: (1, 3, 224, 224)

    Returns:
        {
            "label": "fake" or "real",
            "confidence": float between 0.0 and 1.0
        }
    """
    # torch.no_grad() — turns off gradient tracking during inference
    # Makes it faster and uses less memory (gradients are only needed for training)
    with torch.no_grad():

        # Forward pass — send tensor through the model
        # Output shape: (1, 2) — one row, two class scores (real, fake)
        logits = model(tensor)

        # Softmax converts raw scores to probabilities that sum to 1.0
        # dim=1 means apply softmax across the class dimension
        # e.g. [-1.2, 3.4] → [0.02, 0.98]
        probabilities = torch.softmax(logits, dim=1)

        # Get fake probability (index 1 = fake class)
        fake_prob = probabilities[0][1].item()  # .item() converts tensor → Python float

    # Apply threshold to decide label
    label = "fake" if fake_prob >= THRESHOLD else "real"

    # Confidence = how sure the model is of its prediction
    # If fake: confidence is fake_prob
    # If real: confidence is 1 - fake_prob
    confidence = fake_prob if label == "fake" else 1.0 - fake_prob

    return {
        "label": label,
        "confidence": round(confidence, 4),
    }


# ── Step 4: FastAPI lifespan — load model at startup ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan runs code at server startup and shutdown.

    On startup  → load model into memory (runs once)
    On shutdown → cleanup (optional)

    Usage in main.py:
        from model import lifespan
        app = FastAPI(lifespan=lifespan)
    """
    # Startup
    print("Starting server — loading model...")
    load_model()
    print("Server ready.")

    yield  # server runs here, handling requests

    # Shutdown (runs when server stops)
    print("Server shutting down.")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this to test inference with a dummy tensor before connecting to FastAPI.
    Note: you need a real weights file at MODEL_PATH for this to work.
    """
    print(f"Device: {device}")

    # Simulate a preprocessed tensor (random values, shape matches real input)
    dummy_tensor = torch.randn(1, 3, 224, 224).to(device)

    print("Loading model...")
    load_model()

    print("Running inference on dummy tensor...")
    result = predict(dummy_tensor)

    print(f"Label:      {result['label']}")
    print(f"Confidence: {result['confidence']}")