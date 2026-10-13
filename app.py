import os
import sys
import uuid
import cv2
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ml", "src")))

from rx_recognition_heuristic.pipeline import run_heuristic_pipeline
from rx_llm_advice.other_classifier import classify_other_fields
from rx_llm_advice.dosage_advice import generate_dosage_advice

app = FastAPI(title="Medi-Script AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("web/static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    # Save file
    file_ext = os.path.splitext(file.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join("uploads", unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(file_path)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    # Run pipeline
    base_name = os.path.splitext(unique_name)[0]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", base_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        record = run_heuristic_pipeline(image, output_dir=output_dir)
        record = classify_other_fields(record)

        medicines = record.get("Medical Text", [])
        if isinstance(medicines, str):
            medicines = medicines.split(" | ")

        if medicines:
            advice = generate_dosage_advice(medicines)
        else:
            advice = "No medicines detected in the prescription."

        record["Dosage Advice"] = advice

        return JSONResponse(content=record)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


app.mount("/", StaticFiles(directory="web/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
