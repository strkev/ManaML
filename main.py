import argparse
import os

from src.data_loader import load_processed_dataframe
from src.evaluator import evaluate_and_save
from src.features import prepare_features, save_processed_features, load_processed_features
from src.analyzer import analyze_feature_importance, analyze_top_errors

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

def main():
    parser = argparse.ArgumentParser(description="MTG CMC ML")
    parser.add_argument(
        "--reuse-features", 
        action="store_true", 
        help="Reuse cached TF-IDF features and numpy arrays from data/processed/"
    )
    args = parser.parse_args()

    cache_dir = "data/processed"
    
    if args.reuse_features and os.path.exists(os.path.join(cache_dir, "X_train.npy")):
        X_train, X_test, y_train, y_test, vectorizer, df = load_processed_features(cache_dir)
    else:
        df = load_processed_dataframe("data/all-cards.jsonl")
        X_train, X_test, y_train, y_test, vectorizer = prepare_features(df, max_features=2000)
        save_processed_features(df, X_train, X_test, y_train, y_test, vectorizer, cache_dir)

    models_to_test = {
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1, max_samples=0.8
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, max_depth=10, l2_regularization=1.0, random_state=42
        )

    }

    results = {}
    for name, model in models_to_test.items():
        print(f"\nStart training for '{name}'...")
        model.fit(X_train, y_train)
        scores = evaluate_and_save(model, vectorizer, X_test, y_test, model_name=name)
        results[name] = scores

        analyze_feature_importance(model, vectorizer, top_n=15)
        analyze_top_errors(model, vectorizer, df, test_sets=["hob", "hoc"], top_n=5)

    print("\nComparison")
    for name, scores in results.items():
        print(f"{name:25s} | R²: {scores['r2']:.3f} | MAE: {scores['mae']:.3f}")

if __name__ == "__main__":
    main()