import os
import joblib
import numpy as np
import pandas as pd
from src.features import FEATURE_COLS
from src.data_loader import load_processed_dataframe
from src.plots import plot_feature_importance

def analyze_feature_importance(model, vectorizer, top_n=20, model_name="hist_gradient_boosting"):
    if not hasattr(model, "feature_importances_"):
        print(f"\nModel '{type(model).__name__}'  does not support feature_importances_.")
        return None

    text_feature_names = list(vectorizer.get_feature_names_out())
    all_feature_names = text_feature_names + FEATURE_COLS

    importances = model.feature_importances_

    df_importances = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\n" + "=" * 55)
    print(f"Top {top_n} most important features ({type(model).__name__})")
    print("=" * 55)
    
    for idx, row in df_importances.head(top_n).reset_index(drop=True).iterrows():
        pct = row['importance'] * 100
        print(f"{idx+1:2d}. {row['feature']:25s}")
        
    plot_feature_importance(model, vectorizer, top_n=top_n, model_name=model_name)
    return df_importances


def analyze_top_errors(model, vectorizer, df, test_sets=["hob", "hoc"], top_n=5):
    df_test = df[df["set"].isin(test_sets)].copy()

    if len(df_test) == 0:
        print(f"\nNo test cards {test_sets} found.")
        return

    X_text = vectorizer.transform(df_test["oracle_text"]).toarray()
    X_num = df_test[FEATURE_COLS].values
    X_test = np.hstack((X_text, X_num))

    y_pred_log = model.predict(X_test)
    y_pred_cmc = np.expm1(y_pred_log)

    df_test["pred_cmc"] = np.round(y_pred_cmc, 2)
    df_test["diff"] = np.round(np.abs(df_test["cmc"] - df_test["pred_cmc"]), 2)

    top_errors = df_test.sort_values(by="diff", ascending=False).head(top_n)

    print(f"TOP {top_n} worst predictions of ({type(model).__name__})")

    for idx, row in top_errors.reset_index(drop=True).iterrows():
        print(f"\n[{idx+1}] {row['name']}  ({row['type_line']})")
        print(f"Real CMC: {row['cmc']}  |  Predicted: {row['pred_cmc']}  |  Diff: {row['diff']} Mana")
        if row['oracle_text']:
            text_fmt = row['oracle_text'].replace('\n', '\n    ')
            print(f"Text: \"{text_fmt}\"\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze saved MTG models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="random_forest", 
        help="Name of the model directory inside models/ (e.g., random_forest)"
    )
    args = parser.parse_args()

    model_dir = os.path.join("models", args.model)
    model_path = os.path.join(model_dir, "mtg_cmc_model.pkl")
    vectorizer_path = os.path.join(model_dir, "mtg_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print(f"[Error] Model or vectorizer not found in '{model_dir}'!")
        print("Please run main.py first to train and save the models.")
    else:
        print(f"Loading model and vectorizer from '{model_dir}'...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        df = load_processed_dataframe("data/all-cards.jsonl")

        analyze_feature_importance(model, vectorizer, top_n=15)
        analyze_top_errors(model, vectorizer, df, test_sets=["hob", "hoc"], top_n=5)