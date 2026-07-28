import re

FIELD_NAMES = ["name", "age", "gender", "symptoms", "medicine"]
GENDER_TERMS = {"male", "female", "m", "f"}


def map_text_regions_to_fields(regions: list[dict[str, object]]) -> dict[str, str]:
    """
    Heuristic field mapper for text-detection-first pipelines.
    Expects OCR'd text regions with bounding boxes sorted top-to-bottom.
    """
    cleaned_regions = []
    for region in sorted(regions, key=lambda item: item["bbox"][1]):
        text = str(region.get("clean_text") or region.get("raw_text") or "").strip()
        if text:
            cleaned_regions.append(
                {
                    "text": text,
                    "bbox": region["bbox"],
                }
            )

    mapped = {field: "" for field in FIELD_NAMES}
    remaining: list[dict[str, object]] = []

    for region in cleaned_regions:
        text = region["text"]
        lowered = text.lower()
        if not mapped["age"] and re.fullmatch(r"\d{1,3}", lowered):
            mapped["age"] = text
        elif not mapped["gender"] and lowered in GENDER_TERMS:
            mapped["gender"] = text
        else:
            remaining.append(region)

    if remaining:
        mapped["name"] = remaining[0]["text"]
    if len(remaining) >= 2:
        mapped["medicine"] = remaining[-1]["text"]
    if len(remaining) >= 3:
        symptom_lines = [region["text"] for region in remaining[1:-1]]
        mapped["symptoms"] = " ".join(symptom_lines)
    elif len(remaining) == 2:
        mapped["symptoms"] = remaining[1]["text"]

    return mapped
