import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, top_k_accuracy_score, classification_report, ConfusionMatrixDisplay
)


def evaluate_models(models, X_test, y_test, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_name = None
    best_model = None
    best_f1 = -1

    for name, model in models.items():
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="macro", zero_division=0)
        recall = recall_score(y_test, predictions, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, predictions)

        # Top-K accuracy if probability predictions are supported
        top2_str = "-"
        top3_str = "-"
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            classes = model.classes_
            top2 = top_k_accuracy_score(y_test, probs, k=2, labels=classes)
            top3 = top_k_accuracy_score(y_test, probs, k=3, labels=classes)
            top2_str = f"{top2 * 100:.2f}%"
            top3_str = f"{top3 * 100:.2f}%"

        rows.append({
            "Model": name,
            "Accuracy (Top-1)": accuracy,
            "Top-2 Accuracy": top2_str,
            "Top-3 Accuracy": top3_str,
            "Balanced Accuracy": balanced_acc,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": macro_f1,
            "Cohen Kappa": kappa
        })

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_model = model

    results = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    print("\nMODEL COMPARISON")
    print(results.to_string(index=False))

    print(f"\nBest model: {best_name}")
    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test, best_model.predict(X_test), zero_division=0
    ))

    # Save comparison table
    results.to_csv(output_dir / "model_comparison.csv", index=False)

    # Confusion matrix for the selected model
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test,
        xticks_rotation=90,
        cmap="Blues",
        ax=ax,
        colorbar=False
    )
    ax.set_title(f"Confusion Matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    return best_name, best_model, results
