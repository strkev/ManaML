import os, json, re, joblib, numpy as np, pandas as pd, streamlit as st
from src.data_loader import parse_stat, clean_oracle_text, extract_max_number

st.set_page_config(page_title="ManaML", layout="wide")

@st.cache_data
def load_cards_database(file_path="data/all-cards.jsonl"):
    if not os.path.exists(file_path):
        return None
    
    cards_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            card = json.loads(line)
            layout = card.get("layout", "")
            cmc = card.get("cmc")
            type_line = card.get("type_line", "")

            if layout in ["normal", "adventure", "saga", "split"] and cmc is not None:
                if "Land" not in type_line:
                    raw_text = card.get("oracle_text", "")
                    if not raw_text and "card_faces" in card:
                        raw_text = " ".join([face.get("oracle_text", "") for face in card["card_faces"] if face.get("oracle_text")])

                    text = clean_oracle_text(raw_text)

                    image_url = None
                    if "image_uris" in card and "normal" in card["image_uris"]:
                        image_url = card["image_uris"]["normal"]
                    elif "card_faces" in card and card["card_faces"]:
                        first_face = card["card_faces"][0]
                        if "image_uris" in first_face and "normal" in first_face["image_uris"]:
                            image_url = first_face["image_uris"]["normal"]

                    cards_list.append({
                        "name": card.get("name"),
                        "set": card.get("set", "").upper(),
                        "cmc": float(cmc),
                        "power": parse_stat(card.get("power")),
                        "toughness": parse_stat(card.get("toughness")),
                        "oracle_text": text,
                        "raw_oracle_text": raw_text,
                        "colors_count": len(card.get("colors", [])),
                        "type_line": type_line,
                        "max_number_in_text": extract_max_number(text),
                        "text_length": float(len(text)),
                        "is_creature": 1.0 if "Creature" in type_line else 0.0,
                        "is_spell": 1.0 if "Instant" in type_line or "Sorcery" in type_line else 0.0,
                        "is_artifact": 1.0 if "Artifact" in type_line else 0.0,
                        "is_enchantment": 1.0 if "Enchantment" in type_line else 0.0,
                        "image_url": image_url
                    })

    df = pd.DataFrame(cards_list).drop_duplicates(subset=["name"])
    return df

def get_available_models(models_dir="models"):
    if not os.path.exists(models_dir):
        return []
    models = []
    for entry in os.listdir(models_dir):
        folder = os.path.join(models_dir, entry)
        if os.path.isdir(folder):
            if os.path.exists(os.path.join(folder, "mtg_cmc_model.pkl")) and os.path.exists(os.path.join(folder, "mtg_vectorizer.pkl")):
                models.append(entry)
    return models

@st.cache_resource
def load_model_and_vectorizer(model_name, models_dir="models"):
    model_path = os.path.join(models_dir, model_name, "mtg_cmc_model.pkl")
    vec_path = os.path.join(models_dir, model_name, "mtg_vectorizer.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

st.title("ManaML: Card Analysis & Prediction")

df_cards = load_cards_database()
if df_cards is None:
    st.error("Database file data/all-cards.jsonl not found.")
    st.stop()

available_models = get_available_models()
if not available_models:
    st.error("No trained models found in models/ folder.")
    st.stop()

st.sidebar.header("Settings")
selected_model_name = st.sidebar.selectbox("Select Model", available_models)

selected_card_name = st.selectbox(
    "Search Card",
    options=df_cards["name"].tolist(),
    index=0
)

card_data = df_cards[df_cards["name"] == selected_card_name].iloc[0]

model, vectorizer = load_model_and_vectorizer(selected_model_name)

feature_cols = [
    "colors_count", "power", "toughness",
    "max_number_in_text", "text_length",
    "is_creature", "is_spell", "is_artifact", "is_enchantment"
]

X_text = vectorizer.transform([card_data["oracle_text"]]).toarray()
X_num = np.array([[card_data[col] for col in feature_cols]])
X_input = np.hstack((X_text, X_num))

pred_log = model.predict(X_input)[0]
predicted_cmc = float(np.expm1(pred_log))
real_cmc = float(card_data["cmc"])
diff = predicted_cmc - real_cmc

col_img, col_info = st.columns([1, 2])

with col_img:
    if card_data["image_url"]:
        st.image(card_data["image_url"], use_container_width=True)
    else:
        st.info("No card image available in dataset.")

with col_info:
    st.subheader(card_data["name"])
    st.text(f"Set: {card_data['set']} | Type: {card_data['type_line']}")
    st.text(f"Oracle Text:\n{card_data['raw_oracle_text']}")
    
    st.markdown("---")
    st.subheader("Model Prediction Evaluation")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Real CMC", f"{real_cmc:.1f}")
    col_m2.metric("Predicted CMC", f"{predicted_cmc:.2f}")
    col_m3.metric("Difference", f"{diff:+.2f}")

    if abs(diff) <= 0.4:
        evaluation = "Fairly Priced (Model agrees with actual CMC)"
        st.info(f"Evaluation: {evaluation}")
    elif diff > 0.4:
        evaluation = "Underpriced / Weak for its cost (Model suggests HIGHER cost)"
        st.warning(f"Evaluation: {evaluation}")
    else:
        evaluation = "Overpriced / Strong for its cost (Model suggests LOWER cost)"
        st.success(f"Evaluation: {evaluation}")