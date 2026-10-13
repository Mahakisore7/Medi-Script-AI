import logging
import os

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
        logging.info("✨ Gemini 2.5 Flash client initialized for Dosage Advice.")
        return _gemini_client
    except ImportError:
        raise ImportError("google-genai not installed. Run: uv add google-genai")


_ADVICE_PROMPT = """You are a helpful and knowledgeable pharmacist assistant.
You will be provided with a list of extracted medicine strings from a doctor's prescription.
Your job is to translate these raw strings into simple, plain-English patient instructions.

For example, if the input is: ["Cap. Rozat 10D AC 7AM", "T. Ambulax 1 BD"]
Your output should be:
- Rozat 10D (Capsule): Take 1 before meals (AC) at 7 AM.
- Ambulax (Tablet): Take 1 twice a day (BD).

Do NOT diagnose the patient or offer medical advice beyond translating the abbreviations.
Output ONLY the translated bulleted list.

Here are the medicines:
"""


def generate_dosage_advice(medicines: list[str]) -> str:
    """
    Takes a list of extracted medical strings (medicine name + dosage)
    and uses Gemini to generate a plain-English dosage advice summary.
    """
    if not medicines:
        return "No medicines detected to provide advice for."

    try:
        import time

        client = _get_client()
        prompt = _ADVICE_PROMPT + "\n- " + "\n- ".join(medicines)

        time.sleep(4)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        advice = response.text.strip() if response.text else ""
        return advice

    except Exception as e:
        logging.warning(f"[DosageAdvice] Failed to generate advice: {e}")
        return "Unable to generate dosage advice at this time (LLM Error)."
