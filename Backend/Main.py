from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

app = FastAPI()

# Response model (optional but good practice)
class DetectResponse(BaseModel):
    label: str
    confidence: float


@app.get("/")
def read_root():
    return {"message": "Server is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


#  Detect endpoint
@app.post("/detect", response_model=DetectResponse)
async def detect_image(file: UploadFile = File(...)):

        # 🔸 File type check
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
     # 🔸 File size check (max 2MB)
    contents = await file.read()
    if len(contents) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    return {
        "label": "real",
        "confidence": 0.9
    }

    #CORS Code
    app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sab allow (dev purpose)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)