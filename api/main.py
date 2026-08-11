import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
import os

# Add ml/src to Python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "src")))
from rx_recognition.pipeline import run_image_pipeline

app = FastAPI(
    title="Medi-Script-AI API", 
    description="Offline Medical Prescription Recognition", 
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_prescription(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        # Run pipeline with the standard 'yolo' mode which now includes 
        # TrOCR decoding + EasyOCR reconciliation.
        record = run_image_pipeline(image, mode="yolo")
        return {"status": "success", "result": record}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
