import re


def classify_text_heuristic(
    text: str, y_center_normalized: float, x_center_normalized: float
) -> str:
    """
    Classifies a raw text string based on simple heuristics and regex rules.
    y_center_normalized is the center Y coordinate of the box / image height.
    0.0 is the top of the image, 1.0 is the bottom.
    """
    lower_text = text.lower().strip()

    # 1. Doctor Name Rules
    # Looks for 'Dr', 'Dr.', 'MD', 'MBBS', etc.
    if re.search(r"\b(dr\.?|md\b|mbbs\b|prof\.?)", lower_text):
        return "DoctorName"

    # 2. Patient Info Rules
    # Looks for Name:, Age:, Gender:, Date:, etc.
    # We remove the mandatory colon so it matches "Name Karuna" or "Namc"
    if re.search(r"\b(name|namc|age|ag|gender|sex|patient)\b\s*:?", lower_text):
        return "PatientInfo"

    # Age/Gender heuristic (e.g., 24/M, 35 / F, 19yrs)
    if re.search(r"\b\d{1,3}\s*[/\-]\s*[m|f|male|female]\b", lower_text) or re.search(
        r"\b\d{1,3}\s*yrs?\b", lower_text
    ):
        return "PatientInfo"

    # 3. Date Rules
    if re.search(r"\bdate\b\s*:?", lower_text) or re.search(
        r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", lower_text
    ):
        return "Date"
    # Fallback for OCR missing slashes in top-right corner (e.g., 2241212- for 22/12/22)
    if (
        re.match(r"^[\d\-\.\/]{6,10}$", text.strip())
        and y_center_normalized < 0.25
        and x_center_normalized > 0.6
    ):
        return "Date"

    # 4. Hospital / Clinic Name (Spatial Rule)
    # If it's at the top/bottom of the page OR on the left margin,
    # it's likely a hospital, clinic, or doctor info.
    if y_center_normalized < 0.25 or y_center_normalized > 0.8 or x_center_normalized < 0.25:
        if re.search(r"\b(dr\.?|doctor|m\.b\.b\.s|s\.m\.o\.?)\b", lower_text):
            return "DoctorName"
        elif re.search(
            r"\b(hospital|clinic|care|health|medical|ph\.?|mob\.?|reg|chc|phc|institute|centre)\b",
            lower_text,
        ):
            return "HospitalClinicInfo"
        elif len(text.split()) >= 2:  # At least two words (e.g. "City Hospital")
            return "HospitalClinicInfo"

    # Default fallback
    return "Other"


def parse_patient_info(text: str) -> dict:
    """Extracts Name, Age, Sex from a messy PatientInfo string."""
    info = {"Name": "", "Age": "", "Sex": ""}
    lower_text = text.lower()

    # Extract Age
    age_match = re.search(r"(?:age|ag|years?|yrs?)\s*:?\s*(\d{1,3})", lower_text)
    if age_match:
        info["Age"] = age_match.group(1)

    # Extract Sex
    sex_match = re.search(r"(?:sex|gender|gen)\s*:?\s*([m|f|male|female])\b", lower_text)
    if sex_match:
        val = sex_match.group(1).upper()
        info["Sex"] = "M" if val.startswith("M") else "F"

    # Extract Name (Heuristic: take the words after Name/Namc until Sex/Age)
    name_match = re.search(
        r"(?:name|namc|patient)\s*:?\s*(.+?)(?:sex|gender|age|ag|date|$)", lower_text
    )
    if name_match:
        name = name_match.group(1).strip()
        # Clean up common OCR artifacts at the end of the name
        name = re.sub(r"[_,\-\.]+$", "", name).strip()
        info["Name"] = name.title()

    return info


def parse_date(text: str) -> str:
    """Extracts date from a messy Date string."""
    match = re.search(r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", text)
    if match:
        return match.group(1)
    return text.strip()
