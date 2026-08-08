import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import FEATURE_COLS

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def plot_feature_importance(model, vectorizer, top_n=15, save_dir="models/plots", model_name="hist_gradient_boosting"):
    if not hasattr(model, "feature_importances_"):
        print(f"Skipping feature importance plot: '{type(model).__name__}' does not support feature_importances_.")
        return None

    os.makedirs(save_dir, exist_ok=True)
    text_feature_names = list(vectorizer.get_feature_names_out())
    all_feature_names = text_feature_names + FEATURE_COLS
    importances = model.feature_importances_

    df_imp = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_imp, x="importance", y="feature", palette="Blues_r", ax=ax)
    
    ax.set_title(f"Top {top_n} Feature Importances ({model_name.replace('_', ' ').title()})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Relative Importance Score", fontsize=12)
    ax.set_ylabel("Feature / Token", fontsize=12)
    
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.3f}",
                    (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center',
                    xytext=(5, 0), textcoords='offset points', fontsize=10)

    save_path = os.path.join(save_dir, f"{model_name}_feature_importance.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: '{save_path}'")
    return save_path


def plot_actual_vs_predicted(y_true, y_pred, model_name="Hist Gradient Boosting", save_dir="models/plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_true, y_pred, alpha=0.5, color="#1f77b4", edgecolors="none", s=40, label="Test Cards")

    max_val = max(max(y_true), max(y_pred)) + 1
    ax.plot([0, max_val], [0, max_val], color="#d62728", linestyle="--", linewidth=2, label="Ideal Prediction (y = x)")
    
    ax.set_title(f"Actual vs. Predicted Mana Value ({model_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Mana Value (CMC)", fontsize=12)
    ax.set_ylabel("Predicted Mana Value (CMC)", fontsize=12)
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])
    ax.legend(loc="upper left", frameon=True)
    
    file_name = f"{model_name.lower().replace(' ', '_')}_actual_vs_predicted.png"
    save_path = os.path.join(save_dir, file_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: '{save_path}'")
    return save_path


def plot_residual_distribution(y_true, y_pred, model_name="Hist Gradient Boosting", save_dir="models/plots"):
    os.makedirs(save_dir, exist_ok=True)
    residuals = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#2ca02c", bins=25, ax=ax, stat="density")
    
    ax.axvline(0, color="#d62728", linestyle="--", linewidth=2, label="Zero Error")
    
    ax.set_title(f"Prediction Error Distribution ({model_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Residual Error (Predicted - Actual CMC)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    
    file_name = f"{model_name.lower().replace(' ', '_')}_residuals.png"
    save_path = os.path.join(save_dir, file_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: '{save_path}'")
    return save_path


def plot_model_comparison(results, save_dir="models/plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    records = []
    for model_name, metrics in results.items():
        clean_name = model_name.replace("_", " ").title()
        for metric_name, value in metrics.items():
            records.append({"Model": clean_name, "Metric": metric_name.upper(), "Value": value})
            
    df_results = pd.DataFrame(records)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df_results, x="Metric", y="Value", hue="Model", palette="Set2", ax=ax)
    
    ax.set_title("Model Performance Comparison (Test Set)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Evaluation Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(title="Model", frameon=True)
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points', fontsize=9)

    save_path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: '{save_path}'")
    return save_path
