from rx_recognition.field_mapping import map_text_regions_to_fields


def test_map_text_regions_to_fields_assigns_basic_prescription_lines() -> None:
    regions = [
        {"clean_text": "Linda Moore", "raw_text": "Linda Moore", "bbox": (10, 10, 100, 30)},
        {"clean_text": "40", "raw_text": "40", "bbox": (10, 60, 40, 80)},
        {"clean_text": "Female", "raw_text": "Female", "bbox": (10, 110, 90, 130)},
        {"clean_text": "Hypertension", "raw_text": "Hypertension", "bbox": (10, 160, 150, 190)},
        {"clean_text": "Amlodipine", "raw_text": "Amlodipine", "bbox": (10, 210, 140, 240)},
    ]

    mapped = map_text_regions_to_fields(regions)

    assert mapped == {
        "name": "Linda Moore",
        "age": "40",
        "gender": "Female",
        "symptoms": "Hypertension",
        "medicine": "Amlodipine",
    }
