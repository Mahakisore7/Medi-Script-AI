"""FastAPI service exposing the Medi-Script-AI pipeline.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from mediscript.pipeline import run_heuristic_pipeline

log = logging.getLogger("mediscript.api")

app = FastAPI(
    title="Medi-Script-AI API",
    description="Handwritten medical prescription digitisation (YOLOv8 + TrOCR/Gemini + EasyOCR).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_prescription(file: UploadFile = File(...), advice: bool = False) -> dict:
    """Digitise one prescription image.

    ``advice=true`` additionally calls the LLM modules (needs ``GEMINI_API_KEY``).
    """
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        record = run_heuristic_pipeline(image)
        if advice:
            from mediscript.llm.dosage_advice import generate_dosage_advice
            from mediscript.llm.other_classifier import classify_other_fields

            record = classify_other_fields(record)
            meds = record.get("Medical Text", [])
            record["Dosage Advice"] = generate_dosage_advice(meds) if meds else ""
        return {"status": "success", "result": record}
    except Exception as exc:  # pragma: no cover - surfaced to the client
        log.exception("pipeline failure")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc
