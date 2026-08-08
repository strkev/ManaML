import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.plots import plot_actual_vs_predicted, plot_residual_distribution

def evaluate_and_save(model, vectorizer, X_test, y_test_log, model_name="model", save_dir="models"):
    y_pred_log = model.predict(X_test)

    y_pred = np.expm1(y_pred_log)
    y_test = np.expm1(y_test_log)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nEVALUATION: {model_name.upper()}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    output_path = os.path.join(save_dir, model_name)
    os.makedirs(output_path, exist_ok=True)
    
    joblib.dump(model, os.path.join(output_path, "mtg_cmc_model.pkl"))
    joblib.dump(vectorizer, os.path.join(output_path, "mtg_vectorizer.pkl"))
    print(f"Artefacts saved under '{output_path}/'")

    plot_actual_vs_predicted(y_test, y_pred, model_name=model_name.replace("_", " ").title())
    plot_residual_distribution(y_test, y_pred, model_name=model_name.replace("_", " ").title())

    return {"mae": mae, "rmse": rmse, "r2": r2}