import json
import re
import pandas as pd

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20
}

def parse_stat(value):
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0

def clean_oracle_text(text):
    if not text:
        return ""
    return re.sub(r'\(.*?\)', '', text).strip()

def extract_max_number(text):
    if not text:
        return 0.0
    numbers = [float(n) for n in re.findall(r'\b\d+\b', text)]
    for word, val in WORD_TO_NUM.items():
        if re.search(rf'\b{word}\b', text, re.IGNORECASE):
            numbers.append(float(val))
    valid_numbers = [n for n in numbers if n <= 20]
    return max(valid_numbers) if valid_numbers else 0.0

def load_processed_dataframe(file_path="data/all-cards.jsonl"):
    print(f"Load data '{file_path}'...")
    raw_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    cards = []
    for card in raw_data:
        layout = card.get("layout", "")
        cmc = card.get("cmc")
        type_line = card.get("type_line", "")

        if layout in ["normal", "adventure", "saga", "split"] and cmc is not None:
            if "Land" not in type_line:
                raw_text = card.get("oracle_text", "")
                if not raw_text and "card_faces" in card:
                    raw_text = " ".join([face.get("oracle_text", "") for face in card["card_faces"] if face.get("oracle_text")])
                text = clean_oracle_text(raw_text)
                
                cards.append({
                    "name": card.get("name"),
                    "set": card.get("set", "").lower(),
                    "cmc": float(cmc),
                    "power": parse_stat(card.get("power")),
                    "toughness": parse_stat(card.get("toughness")),
                    "oracle_text": text,
                    "colors_count": len(card.get("colors", [])),
                    "type_line": type_line,
                    "max_number_in_text": extract_max_number(text),
                    "text_length": float(len(text)),
                    "is_creature": 1.0 if "Creature" in type_line else 0.0,
                    "is_spell": 1.0 if "Instant" in type_line or "Sorcery" in type_line else 0.0,
                    "is_artifact": 1.0 if "Artifact" in type_line else 0.0,
                    "is_enchantment": 1.0 if "Enchantment" in type_line else 0.0,
                })

    df = pd.DataFrame(cards).drop_duplicates(subset=["name"])
    print(f"Processed cards: {len(df)}")
    return df