"""Tier-1 recogniser: a Gemini vision model reads one handwritten crop.

Returns ``("", 0.0)`` on ``UNREADABLE`` or on any API error so that the caller
can fall back to the local TrOCR tier.
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np
from PIL import Image

from mediscript.config import GEMINI_MODEL, LLM_DELAY_S, gemini_api_key

# Singleton: reuse the same client across calls
_gemini_client = None

_MEDICINE_PROMPT = (
    "You are an expert pharmacist reading a handwritten prescription crop. "
    "Read and output the text exactly as written. "
    "The text may contain a medicine name, a dosage instruction, or both. "
    "Do NOT add any explanation, markdown formatting, or extra words. "
    "If the image is completely blank or completely illegible noise, output exactly: UNREADABLE"
)

# Fixed pseudo-confidence reported for a successful Gemini read.
GEMINI_CONFIDENCE = 85.0


def _get_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = gemini_api_key()
    if not api_key:
        raise OSError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    try:
        from google import genai

        _gemini_client = genai.Client(api_key=api_key)
        logging.info("Gemini OCR client initialised (%s).", GEMINI_MODEL)
        return _gemini_client
    except ImportError as exc:
        raise ImportError("google-genai not installed. Run: pip install google-genai") from exc


def recognize_with_gemini(image: np.ndarray, field_type: str = "medicine") -> tuple[str, float]:
    """Send a BGR crop to Gemini Vision and return ``(text, confidence)``."""
    try:
        from google.genai import types

        client = _get_client()

        pil_img = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        response = None
        for attempt in range(2):
            try:
                time.sleep(LLM_DELAY_S)  # free-tier rate limit; set MEDISCRIPT_LLM_DELAY=0 to disable
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        _MEDICINE_PROMPT,
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    ],
                )
                break
            except Exception as exc:  # retry once on overload / rate limit
                if ("503" in str(exc) or "429" in str(exc)) and attempt == 0:
                    logging.info("[GeminiOCR] API error %s. Retrying in 10s...", exc)
                    time.sleep(10)
                    continue
                raise

        if response is None:
            return "", 0.0

        raw = response.text.strip() if response.text else ""
        if not raw or raw.upper() in ("UNREADABLE", "N/A", "UNCLEAR"):
            logging.info("[GeminiOCR] Crop marked UNREADABLE.")
            return "", 0.0

        logging.info("[GeminiOCR] Read: '%s'", raw)
        return raw, GEMINI_CONFIDENCE

    except Exception as exc:
        logging.warning("[GeminiOCR] Failed: %s", exc)
        return "", 0.0
