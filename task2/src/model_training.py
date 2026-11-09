
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

from src.preprocessing import prepare_features


def train_models(X, y):

    print("\n========== MODEL TRAINING ==========")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        # Create independent preprocessing pipeline
        pipeline = Pipeline([
            ("preprocessor", prepare_features(X_train)),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

        print(f"{name} training completed!")

    return trained_models, X_test, y_test
