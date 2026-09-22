def build_record(field_texts: dict[str, str]) -> dict:
    """
    Takes the dictionary of raw recognized texts and formats them into a final JSON record.
    """
    # Combine multiple instances of the same field (e.g. medicine, medicine_1, medicine_2)
    _MEDICINE_KEYS = ("medicine", "prescription", "medical text")
    _SYMPTOM_KEYS = ("symptoms", "diagnosis")
    _DOCTOR_KEYS = ("doctername",)
    _PATIENT_KEYS = ("patientinfo",)

    symptoms = "\n".join(
        [v for k, v in field_texts.items() if k.lower().startswith(_SYMPTOM_KEYS) and v]
    )
    medicines = "\n".join(
        [v for k, v in field_texts.items() if k.lower().startswith(_MEDICINE_KEYS) and v]
    )
    doctor_name = "\n".join(
        [v for k, v in field_texts.items() if k.lower().startswith(_DOCTOR_KEYS) and v]
    )
    patient_info = "\n".join(
        [v for k, v in field_texts.items() if k.lower().startswith(_PATIENT_KEYS) and v]
    )

    return {
        "patient": {
            "raw_info": patient_info,
            "name": field_texts.get("name", ""),
            "age": field_texts.get("age", ""),
            "gender": field_texts.get("gender", ""),
        },
        "doctor": {"name": doctor_name},
        "clinical": {
            "symptoms": symptoms,
            "prescription": medicines,
        },
    }
