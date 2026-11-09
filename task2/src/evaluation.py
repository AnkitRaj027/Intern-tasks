
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


def evaluate_models(
    trained_models,
    X_test,
    y_test
):

    print("\n========== MODEL EVALUATION ==========")

    results = []

    best_model = None
    best_model_name = None
    best_f1 = -1

    for name, model in trained_models.items():

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="macro", zero_division=0)
        recall = recall_score(y_test, predictions, average="macro", zero_division=0)
        f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, predictions)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1,
            "Cohen Kappa": kappa
        })

        print(f"\n{name}")
        print("Accuracy:         ", round(accuracy, 4))
        print("Balanced Accuracy:", round(balanced_acc, 4))
        print("Precision (Macro):", round(precision, 4))
        print("Recall (Macro):   ", round(recall, 4))
        print("F1 Score (Macro): ", round(f1, 4))
        print("Cohen's Kappa:    ", round(kappa, 4))

        # Select model using macro F1
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)

    print("\n========== MODEL COMPARISON ==========")
    print(
        results_df.sort_values(
            by="Macro F1",
            ascending=False
        ).to_string(index=False)
    )

    # Save CSV comparison
    try:
        from pathlib import Path
        out_dir = Path(__file__).resolve().parent.parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        results_df.to_csv(out_dir / "model_comparison.csv", index=False)
    except Exception as e:
        pass

    print("\nBest Model:", best_model_name)

    # Evaluate the selected model
    best_predictions = best_model.predict(X_test)

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            best_predictions,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=["Movie", "TV Show"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Movie", "TV Show"]
    )

    display.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()

    try:
        from pathlib import Path
        save_dir = Path(__file__).resolve().parent.parent / "models"
        save_dir.mkdir(exist_ok=True)
        plot_path = save_dir / "confusion_matrix.png"
        plt.savefig(plot_path)
        print(f"\nConfusion matrix saved to: {plot_path}")
    except Exception as e:
        print(f"Note: Could not save plot: {e}")

    plt.close()

    return best_model, best_model_name, results_df
