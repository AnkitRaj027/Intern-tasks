import pandas as pd

from .preprocessing import prepare_features


def predict_from_metadata(model):
    """Predict rating for a new title when the user supplies its attributes."""
    print("\nEnter content details. Leave unknown values blank.")

    content_type = input("Type (Movie/TV Show): ").strip() or None
    director = input("Director: ").strip() or None
    country = input("Country: ").strip() or None
    release_year = input("Release year: ").strip() or None
    genres = input("Genres (comma-separated): ").strip() or None
    duration = input("Duration (e.g. 90 min, 2 Seasons): ").strip() or None

    record = pd.DataFrame([{
        "type": content_type,
        "director": director,
        "country": country,
        "release_year": release_year,
        "listed_in": genres,
        "duration": duration
    }])

    features = prepare_features(record)
    prediction = model.predict(features)[0]

    print("\n" + "=" * 45)
    print("PREDICTED AUDIENCE RATING:", prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = model.named_steps["classifier"].classes_
        best_index = probabilities.argmax()
        print(f"Model probability for prediction: {probabilities[best_index] * 100:.2f}%")
        print("Note: This is a model probability, not a guarantee of correctness.")
    else:
        print("Probability estimate is not available for this model.")
    print("=" * 45)


def predict_for_existing_title(model, df):
    """Look up an existing title and predict using its content attributes."""
    while True:
        title = input("\nEnter Netflix title (or 'back' to return): ").strip()
        if title.lower() == "back":
            return

        matches = df[df["title"].astype(str).str.strip().str.casefold() == title.casefold()]
        if matches.empty:
            print("Title not found in the dataset.")
            print("You can use the new-title metadata prediction option instead.")
            continue

        if len(matches) > 1:
            print("\nMultiple matching entries:")
            for idx, row in matches.iterrows():
                print(f"{idx}: {row['title']} ({row.get('release_year', 'Unknown')})")
            choice = input("Enter the row number to use, or press Enter for the first: ").strip()
            if choice.isdigit() and int(choice) in matches.index:
                selected = matches.loc[int(choice)]
            else:
                selected = matches.iloc[0]
        else:
            selected = matches.iloc[0]

        row = pd.DataFrame([selected.to_dict()])
        features = prepare_features(row)
        prediction = model.predict(features)[0]

        print("\nTitle:", selected["title"])
        print("Predicted audience rating:", prediction)
        print("Dataset rating (for comparison):", selected["rating"])
        print("Note: Dataset rating is shown only as a reference; the model prediction is separate.")


def prediction_menu(model, df):
    while True:
        print("\n" + "=" * 50)
        print("NETFLIX AUDIENCE RATING PREDICTOR")
        print("1. Predict for a title in the dataset")
        print("2. Predict a new title using its metadata")
        print("3. Exit")

        try:
            choice = input("Choose an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting predictor.")
            break

        if choice == "1":
            predict_for_existing_title(model, df)
        elif choice == "2":
            predict_from_metadata(model)
        elif choice == "3":
            print("Exiting predictor.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
