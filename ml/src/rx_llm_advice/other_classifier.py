import json
import logging
import os

_gemini_client = None


def _get_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise OSError("GEMINI_API_KEY environment variable not set.")

    try:
        from google import genai

        _gemini_client = genai.Client(api_key=api_key)
        logging.info("✨ Gemini 2.5 Flash client initialized for Other Classifier.")
        return _gemini_client
    except ImportError:
        raise ImportError("google-genai not installed.")


_CLASSIFIER_PROMPT = """You are a helpful and knowledgeable pharmacist assistant.
You will be provided with a raw string extracted from a prescription which we couldn't classify (separated by '|').
Your job is to identify if any parts of this string belong to the following fields:
- PatientName (Any person's name, unless it has a doctor title)
- Age
- Sex
- DoctorName (A name MUST explicitly start with 'Dr.', 'Dr ', 'Doctor', etc. to be classified as DoctorName. Otherwise, it is PatientName)
- Date
- HospitalClinicInfo

Any text that you CANNOT confidently classify into the above fields MUST be collected and returned in the 'RemainingOther' field, separated by ' | '.

Output your response strictly as a JSON object matching this schema:
{
  "PatientName": "",
  "Age": "",
  "Sex": "",
  "DoctorName": "",
  "Date": "",
  "HospitalClinicInfo": "",
  "RemainingOther": ""
}

Return ONLY valid JSON and nothing else.

Here is the raw text to classify:
"""


def classify_other_fields(record: dict) -> dict:
    """
    Takes the existing record, sends the 'Other' string to Gemini to classify,
    and fills in any EMPTY fields in the record.
    """
    other_text = record.get("Other", "")
    if not other_text or not isinstance(other_text, str):
        return record

    try:
        import time

        client = _get_client()
        prompt = _CLASSIFIER_PROMPT + other_text

        time.sleep(4)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        parsed = json.loads(raw_text.strip())

        # Update fields ONLY if they are currently empty
        if not record.get("DoctorName") and parsed.get("DoctorName"):
            record["DoctorName"] = parsed["DoctorName"]

        if not record.get("Date") and parsed.get("Date"):
            record["Date"] = parsed["Date"]

        if parsed.get("HospitalClinicInfo"):
            # If clinic info exists, append it, or replace if empty
            if not record.get("HospitalClinicInfo"):
                record["HospitalClinicInfo"] = parsed["HospitalClinicInfo"]
            else:
                record["HospitalClinicInfo"] += " | " + parsed["HospitalClinicInfo"]

        patient_dict = record.get("Patient", {})
        if not patient_dict.get("Name") and parsed.get("PatientName"):
            patient_dict["Name"] = parsed["PatientName"]

        if not patient_dict.get("Age") and parsed.get("Age"):
            patient_dict["Age"] = str(parsed["Age"])

        if not patient_dict.get("Sex") and parsed.get("Sex"):
            patient_dict["Sex"] = parsed["Sex"]

        record["Patient"] = patient_dict

        # Replace 'Other' with the unclassified leftovers
        record["Other"] = parsed.get("RemainingOther", "")

    except Exception as e:
        logging.warning(f"[OtherClassifier] Failed to classify other fields: {e}")

    return record
