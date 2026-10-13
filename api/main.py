"""FastAPI service exposing the Medi-Script-AI pipeline and the web dashboard.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000   (then open http://localhost:8000)

Endpoints
  POST /api/upload   full pipeline + LLM post-processing (used by the dashboard)
  POST /analyze      pipeline only; ``?advice=true`` adds the LLM stages
  GET  /health
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from mediscript.config import OUTPUTS_DIR, PROJECT_ROOT
from mediscript.pipeline import run_heuristic_pipeline

load_dotenv()
log = logging.getLogger("mediscript.api")

UPLOADS_DIR = PROJECT_ROOT / "uploads"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Medi-Script-AI API",
    description="Handwritten medical prescription digitisation (YOLOv8 + Gemini/TrOCR + EasyOCR).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _with_llm(record: dict) -> dict:
    """Apply the optional LLM stages (needs GEMINI_API_KEY)."""
    from mediscript.llm.dosage_advice import generate_dosage_advice
    from mediscript.llm.other_classifier import classify_other_fields

    record = classify_other_fields(record)
    medicines = record.get("Medical Text", [])
    if isinstance(medicines, str):
        medicines = medicines.split(" | ")
    record["Dosage Advice"] = (
        generate_dosage_advice(medicines) if medicines else "No medicines detected in the prescription."
    )
    return record


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    """Dashboard endpoint: saves the upload, runs pipeline + LLM stages, returns the record."""
    UPLOADS_DIR.mkdir(exist_ok=True)
    suffix = Path(file.filename or "").suffix or ".jpg"
    name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOADS_DIR / name
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(str(path))
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    out_dir = OUTPUTS_DIR / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        record = _with_llm(run_heuristic_pipeline(image, output_dir=str(out_dir)))
        return JSONResponse(content=record)
    except Exception as exc:  # pragma: no cover - surfaced to the client
        log.exception("pipeline failure")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/analyze")
async def analyze_prescription(file: UploadFile = File(...), advice: bool = False) -> dict:
    """Programmatic endpoint: pipeline only, LLM stages optional."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    try:
        record = run_heuristic_pipeline(image)
        if advice:
            record = _with_llm(record)
        return {"status": "success", "result": record}
    except Exception as exc:  # pragma: no cover
        log.exception("pipeline failure")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc


# Static dashboard (must be mounted last so it does not shadow the API routes)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
