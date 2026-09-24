"""
gemini_ocr.py
─────────────
Uses Gemini 2.0 Flash Vision to read handwritten text from prescription image crops.
Replaces TrOCR as the OCR backend in the YOLO pipeline.
"""

import logging
import os

import numpy as np
from PIL import Image

# ── Singleton: reuse the same client ───────────────────────────────────────────
_gemini_client = None


def _get_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise OSError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )

    try:
        from google import genai

        _gemini_client = genai.Client(api_key=api_key)
        logging.info("✨ Gemini 2.0 Flash OCR client initialized.")
        return _gemini_client
    except ImportError:
        raise ImportError("google-genai not installed. Run: uv add google-genai")


# ── Prompt ─────────────────────────────────────────────────────────────────────
_MEDICINE_PROMPT = (
    "You are a pharmacist reading a handwritten prescription crop. "
    "The image shows ONE medicine line (name + dosage). "
    "Read and output ONLY the medicine name and dosage exactly as written "
    "(e.g. 'Ecosprin 75mg' or 'Clopidogrel 75 mg'). "
    "Do NOT add explanation. "
    "If unclear or not a medicine, output exactly: UNREADABLE"
)


def recognize_with_gemini(image: np.ndarray, field_type: str = "medicine") -> tuple[str, float]:
    """
    Send a crop image to Gemini Vision and return (text, confidence).
    """
    try:
        from google.genai import types

        client = _get_client()

        # Convert BGR numpy → PIL RGB → bytes
        pil_img = Image.fromarray(image[:, :, ::-1])  # BGR → RGB

        import io
        import time

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # Sleep for 4 seconds to avoid hitting Gemini Free Tier 15 RPM limits
        time.sleep(4)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                _MEDICINE_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        )

        raw = response.text.strip() if response.text else ""

        if not raw or raw.upper() in ("UNREADABLE", "N/A", "UNCLEAR", ""):
            logging.info("[GeminiOCR] Crop marked UNREADABLE.")
            return "", 0.0

        logging.info(f"[GeminiOCR] Read: '{raw}'")
        return raw, 85.0

    except Exception as e:
        logging.warning(f"[GeminiOCR] Failed: {e}")
        return "", 0.0
