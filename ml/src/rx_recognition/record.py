def build_record(field_texts: dict[str, str]) -> dict:
    """
    Takes the dictionary of raw recognized texts and formats them into a final JSON record.
    """
    # Combine multiple instances of the same field (e.g. medicine, medicine_1, medicine_2)
    # Also accept "Prescription" as a class name (used by the Bannari Amman Roboflow dataset)
    _MEDICINE_KEYS = ("medicine", "prescription")
    symptoms = "\n".join([v for k, v in field_texts.items() if k.lower().startswith("symptoms") and v])
    medicines = "\n".join([v for k, v in field_texts.items() if k.lower().startswith(_MEDICINE_KEYS) and v])

    return {
        "patient": {
            "name": field_texts.get("name", ""),
            "age": field_texts.get("age", ""),
            "gender": field_texts.get("gender", ""),
        },
        "clinical": {
            "symptoms": symptoms,
            "prescription": medicines,
        },
    }
