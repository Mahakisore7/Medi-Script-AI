import pytest

from mediscript.parsing.record import build_record
from mediscript.parsing.regex_classifier import (
    classify_text_heuristic,
    parse_date,
    parse_patient_info,
)


@pytest.mark.parametrize(
    "text, y, x, expected",
    [
        ("Dr. Sanjeev Saxena MBBS", 0.1, 0.5, "DoctorName"),
        ("Name: Karuna", 0.3, 0.3, "PatientInfo"),
        ("28/M", 0.3, 0.3, "PatientInfo"),
        ("17/7/24", 0.5, 0.5, "Date"),
        ("City Hospital Kota", 0.05, 0.5, "HospitalClinicInfo"),
        ("random scribble", 0.5, 0.5, "Other"),
    ],
)
def test_classify_text_heuristic(text, y, x, expected):
    assert classify_text_heuristic(text, y, x) == expected


def test_parse_patient_info():
    info = parse_patient_info("Name Kamuna Chouhan Age: 36 Sex: F")
    assert info["Age"] == "36"
    assert info["Sex"] == "F"
    assert "Kamuna" in info["Name"]


def test_parse_date():
    assert parse_date("Date 17/7/24") == "17/7/24"


def test_build_record_merges_medicine_keys():
    rec = build_record({"name": "A", "medicine": "Ozad", "medicine_1": "Rozat", "symptoms": "fever"})
    assert rec["patient"]["name"] == "A"
    assert rec["clinical"]["prescription"].split("\n") == ["Ozad", "Rozat"]
    assert rec["clinical"]["symptoms"] == "fever"
