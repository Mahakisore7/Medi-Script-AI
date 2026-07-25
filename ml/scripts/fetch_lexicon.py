import csv
import json
import urllib.request
import os

def fetch_fda_drugs(limit=1000) -> list[str]:
    """
    Fetches drug names (brand and generic) from the openFDA API.
    """
    url = f"https://api.fda.gov/drug/label.json?search=_exists_:openfda.brand_name&limit={limit}"
    
    print(f"Fetching up to {limit} drug labels from openFDA...")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to fetch from openFDA: {e}")
        return []

    drugs = set()
    for result in data.get("results", []):
        openfda = result.get("openfda", {})
        
        # Add brand names
        for brand in openfda.get("brand_name", []):
            drugs.add(brand.strip().title())
            
        # Add generic names
        for generic in openfda.get("generic_name", []):
            drugs.add(generic.strip().title())
            
    print(f"Successfully extracted {len(drugs)} unique drug names.")
    return sorted(list(drugs))

def get_common_symptoms() -> list[str]:
    """Returns a hardcoded list of common medical symptoms."""
    return sorted([
        "Fever", "Cough", "Headache", "Nausea", "Vomiting", "Fatigue", 
        "Dizziness", "Shortness of breath", "Chest pain", "Abdominal pain", 
        "Diarrhea", "Constipation", "Chills", "Sweating", "Muscle ache", 
        "Joint pain", "Rash", "Itching", "Sore throat", "Runny nose", 
        "Congestion", "Sneezing", "Wheezing", "Palpitations", "Weight loss", 
        "Weight gain", "Insomnia", "Anxiety", "Depression", "Confusion", 
        "Weakness", "Numbness", "Tingling", "Vision changes", "Hearing loss"
    ])

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
    
    drugs = fetch_fda_drugs(limit=1000)
    save_to_csv(drugs, os.path.join(project_root, "data", "lexicon", "medicines.csv"))
    
    symptoms = get_common_symptoms()
    save_to_csv(symptoms, os.path.join(project_root, "data", "lexicon", "symptoms.csv"))
