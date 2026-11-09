from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .preprocessing import build_preprocessor


def train_and_tune_models(X_train, y_train, random_state=42):
    models = {
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
            min_samples_leaf=2
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2
        )
    }

    trained = {}
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ])
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        print(f"Trained: {name}")

    # Tune a Random Forest using cross-validation on the training split only.
    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    parameter_space = {
        "classifier__n_estimators": [200, 300, 400],
        "classifier__max_depth": [None, 15, 25, 35],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", 0.5]
    }

    min_class_count = y_train.value_counts().min()
    n_splits = min(3, int(min_class_count))

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            rf_pipeline,
            param_distributions=parameter_space,
            n_iter=8,
            scoring="f1_macro",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            refit=True
        )
        search.fit(X_train, y_train)
        trained["Tuned Random Forest"] = search.best_estimator_
        print("\nBest Random Forest parameters:")
        print(search.best_params_)
        print(f"Best CV macro F1: {search.best_score_:.4f}")
    else:
        print("Skipping hyperparameter tuning: not enough samples per class.")

    return trained
