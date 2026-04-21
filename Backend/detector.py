from PIL import Image
import io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

# ── Constants ──────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

# ── Load MTCNN face detector ───────────────────────────────────────────────────
# keep_all=False  → only detect the largest/most confident face
# post_process=False → return raw pixel values (we handle normalization ourselves)
# device          → use GPU if available, otherwise CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(
    keep_all=False,
    post_process=False,
    device=device
)

# ── Preprocessing pipeline ─────────────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Step 1: Validate image ─────────────────────────────────────────────────────

def is_valid_image(image_bytes: bytes) -> bool:
    """
    Confirms the uploaded file is a real, uncorrupted image.
    Call this first in main.py before doing anything else.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return True
    except Exception:
        return False


# ── Step 2: Detect and crop face ──────────────────────────────────────────────

def detect_face(image: Image.Image):
    """
    Runs MTCNN on the PIL image.
    Returns a cropped PIL face image if a face is found.
    Returns None if no face is detected.

    How it works:
    - MTCNN scans the image for faces
    - Returns a tensor of the cropped face region
    - We convert it back to a PIL image for the preprocessing pipeline
    """

    # MTCNN returns a tensor of shape (3, H, W) with pixel values 0-255
    # or None if no face is found
    face_tensor = mtcnn(image)

    if face_tensor is None:
        return None  # caller handles this — no face found

    # Convert tensor back to PIL image for the preprocessing pipeline
    # face_tensor values are 0-255 floats, need to be uint8 for PIL
    face_array = face_tensor.permute(1, 2, 0).byte().numpy()  # (H, W, 3)
    face_image = Image.fromarray(face_array)

    return face_image


# ── Step 3: Preprocess face for model ─────────────────────────────────────────

def preprocess_image(image_bytes: bytes):
    """
    Full pipeline: bytes → face detection → tensor

    Returns:
        tensor        → torch.Tensor of shape (1, 3, 224, 224) — pass to model
        face_detected → bool — tells main.py if a face was found

    Usage in main.py:
        tensor, face_detected = preprocess_image(image_bytes)
        if not face_detected:
            raise HTTPException(status_code=400, detail="No face detected in image")
    """

    # Open image from raw bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Try to detect and crop face
    face_image = detect_face(image)

    if face_image is None:
        return None, False

    # Run preprocessing pipeline on the cropped face
    tensor = preprocess(face_image)         # shape: (3, 224, 224)
    tensor = tensor.unsqueeze(0)            # shape: (1, 3, 224, 224)
    tensor = tensor.to(device)             # move to GPU if available

    return tensor, True


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Using device: {device}")
    print("Testing with a blank image (no face expected)...")

    dummy = Image.new("RGB", (300, 400), color=(120, 80, 200))
    buffer = io.BytesIO()
    dummy.save(buffer, format="JPEG")
    image_bytes = open("test.jpg", "rb").read()

    tensor, face_detected = preprocess_image(image_bytes)

    if not face_detected:
        print("No face detected — correct! (blank image has no face)")
    else:
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print("Face detected and preprocessed successfully!")

    print("\nTo test with a real face image:")
    print("  1. Put a photo called 'test.jpg' in this folder")
    print("  2. Replace dummy image code with: image_bytes = open('test.jpg','rb').read()")