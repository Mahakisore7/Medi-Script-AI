import csv
import os


def process_kaggle_medicines(kaggle_csv_path: str) -> list[str]:
    """
    Reads the Kaggle A-Z Indian Medicine dataset and extracts the brand names.
    """
    if not os.path.exists(kaggle_csv_path):
        print(f"File not found: {kaggle_csv_path}")
        return []

    drugs = set()
    try:
        with open(kaggle_csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "").strip()
                if name:
                    # Extract just the first word (brand name) to improve fuzzy matching
                    # e.g., "Augmentin 625 Duo Tablet" -> "Augmentin"
                    brand = name.split()[0].title()
                    drugs.add(brand)
                    # Also add the full name just in case
                    drugs.add(name.title())
    except Exception as e:
        print(f"Error parsing Kaggle dataset: {e}")

    print(f"Extracted {len(drugs)} unique Indian drug names/brands.")
    return sorted(list(drugs))


def get_common_symptoms() -> list[str]:
    """Returns a hardcoded list of common medical symptoms."""
    return sorted(
        [
            "Fever",
            "Cough",
            "Headache",
            "Nausea",
            "Vomiting",
            "Fatigue",
            "Dizziness",
            "Shortness of breath",
            "Chest pain",
            "Abdominal pain",
            "Diarrhea",
            "Constipation",
            "Chills",
            "Sweating",
            "Muscle ache",
            "Joint pain",
            "Rash",
            "Itching",
            "Sore throat",
            "Runny nose",
            "Congestion",
            "Sneezing",
            "Wheezing",
            "Palpitations",
            "Weight loss",
            "Weight gain",
            "Insomnia",
            "Anxiety",
            "Depression",
            "Confusion",
            "Weakness",
            "Numbness",
            "Tingling",
            "Vision changes",
            "Hearing loss",
        ]
    )


def save_to_csv(data: list[str], filepath: str):
    """Saves a list of strings to a single-column CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for item in data:
            writer.writerow([item])
    print(f"Saved {len(data)} entries to {filepath}")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    kaggle_csv = os.path.join(project_root, "data", "lexicon", "A_Z_medicines_dataset_of_India.csv")
    drugs = process_kaggle_medicines(kaggle_csv)

    if drugs:
        save_to_csv(drugs, os.path.join(project_root, "data", "lexicon", "medicines.csv"))

    symptoms = get_common_symptoms()
    save_to_csv(symptoms, os.path.join(project_root, "data", "lexicon", "symptoms.csv"))
