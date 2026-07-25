def build_record(field_texts: dict[str, str]) -> dict:
    """
    Takes the dictionary of raw recognized texts and formats them into a final JSON record.
    """
    return {
        "patient": {
            "name": field_texts.get("name", ""),
            "age": field_texts.get("age", ""),
            "gender": field_texts.get("gender", ""),
        },
        "clinical": {
            "symptoms": field_texts.get("symptoms", ""),
            "prescription": field_texts.get("medicine", ""),
        },
    }
