import json
import os
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Root paths
ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "data" / "raw" / "Dataset.csv"

# Global state
DF = None
PROCESSED_DF = None
WEIGHTED_MATRIX = None
TASK2_MODEL = None
TASK3_MODEL = None
TASK4_MODEL = None


def load_task_model(task_dir, model_rel_path):
    orig_path = list(sys.path)
    for mod in list(sys.modules.keys()):
        if mod == "src" or mod.startswith("src."):
            del sys.modules[mod]
    sys.path.insert(0, str((ROOT_DIR / task_dir).resolve()))
    try:
        model = joblib.load(ROOT_DIR / model_rel_path)
    finally:
        sys.path = orig_path
    return model


def init_ml_models():
    global DF, PROCESSED_DF, WEIGHTED_MATRIX, TASK2_MODEL, TASK3_MODEL, TASK4_MODEL

    print("Initializing Machine Learning models...")

    # Load Dataset
    if DATASET_PATH.exists():
        raw_df = pd.read_csv(DATASET_PATH)
        # Deduplicate
        dup_cols = ["type", "title", "director", "country", "date_added", "release_year", "rating", "duration", "listed_in"]
        DF = raw_df.drop_duplicates(subset=dup_cols).reset_index(drop=True)
        print(f"Loaded dataset: {len(DF)} records.")

        # Init Task 1 (Recommendation engine)
        try:
            sys_path_backup = list(sys.path)
            for m in list(sys.modules.keys()):
                if m == "src" or m.startswith("src."):
                    del sys.modules[m]
            sys.path.insert(0, str((ROOT_DIR / "task1").resolve()))

            from src.preprocessing import preprocess_data
            from src.feature_engineering import create_combined_features, create_weighted_tfidf

            p_df = preprocess_data(DF)
            PROCESSED_DF = create_combined_features(p_df)
            WEIGHTED_MATRIX = create_weighted_tfidf(PROCESSED_DF)
            print("Task 1 (Recommendation Engine) initialized.")
            sys.path = sys_path_backup
        except Exception as e:
            print(f"Error initializing Task 1: {e}")

    # Load Task 2 Model
    try:
        TASK2_MODEL = load_task_model("task2", "task2/models/best_model.joblib")
        print("Task 2 (Content Type Classifier) loaded.")
    except Exception as e:
        print(f"Error loading Task 2 model: {e}")

    # Load Task 3 Model
    try:
        TASK3_MODEL = load_task_model("task3", "task3/models/best_rating_model.joblib")
        print("Task 3 (Audience Rating Model) loaded.")
    except Exception as e:
        print(f"Error loading Task 3 model: {e}")

    # Load Task 4 Model
    try:
        t4_dict = load_task_model("task4", "task4/models/netflix_kmeans_model.joblib")
        TASK4_MODEL = t4_dict
        print("Task 4 (Content Segmentation KMeans) loaded.")
    except Exception as e:
        print(f"Error loading Task 4 model: {e}")


class AuspifyRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for local testing if needed
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def send_json_response(self, data, status_code=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/status":
            self.send_json_response({
                "status": "online",
                "total_titles": len(DF) if DF is not None else 0,
                "task1_ready": WEIGHTED_MATRIX is not None,
                "task2_ready": TASK2_MODEL is not None,
                "task3_ready": TASK3_MODEL is not None,
                "task4_ready": TASK4_MODEL is not None,
            })
            return

        elif path == "/api/titles":
            search_query = query.get("q", [""])[0].strip().lower()
            limit = int(query.get("limit", [15])[0])

            if DF is None:
                self.send_json_response([], 200)
                return

            if not search_query:
                # Popular suggestions
                popular = ["Breaking Bad", "Stranger Things", "Narcos", "The Irishman", "The Queen's Gambit", "Ozark", "Dark", "Peaky Blinders"]
                results = DF[DF["title"].isin(popular)][["title", "type", "release_year", "listed_in"]].to_dict(orient="records")
            else:
                matches = DF[DF["title"].astype(str).str.lower().str.contains(search_query, regex=False)]
                results = matches.head(limit)[["title", "type", "release_year", "listed_in"]].to_dict(orient="records")

            self.send_json_response(results)
            return

        elif path == "/api/title-details":
            title = query.get("title", [""])[0].strip()
            if DF is not None and title:
                matches = DF[DF["title"].astype(str).str.casefold() == title.casefold()]
                if not matches.empty:
                    self.send_json_response(matches.iloc[0].to_dict())
                    return
            self.send_json_response({"error": "Title not found"}, 404)
            return

        elif path == "/api/task4/clusters":
            csv_path = ROOT_DIR / "task4" / "outputs" / "cluster_summary.csv"
            examples_path = ROOT_DIR / "task4" / "outputs" / "cluster_examples.csv"

            if csv_path.exists():
                summary_df = pd.read_csv(csv_path)
                examples_df = pd.read_csv(examples_path) if examples_path.exists() else pd.DataFrame()
                combined = summary_df.merge(examples_df, on="cluster", how="left")
                self.send_json_response(combined.to_dict(orient="records"))
                return
            self.send_json_response({"error": "Cluster data not found"}, 404)
            return

        elif path == "/api/evaluations":
            eval_data = {
                "task1": {
                    "task_id": "Task 1",
                    "title": "Recommendation Engine Architecture Benchmark",
                    "subtitle": "Evaluated across 50 diverse catalog titles (500 recommendations evaluated)",
                    "sample_info": "50 Diverse Catalog Titles • Top-10 Ranking • 500 Recommendation Slots Audited",
                    "columns": ["Model Architecture", "Precision@10", "Format Match", "Avg Genre Overlap (Jaccard)", "NDCG@10", "Catalog Coverage"],
                    "rows": [
                        {"model": "Weighted Multi-Feature TF-IDF", "col1": "100.00%", "col2": "100.00%", "col3": "96.57%", "col4": "0.9974", "col5": "5.45%", "rank": 1, "is_best": True},
                        {"model": "Standard Combined TF-IDF", "col1": "79.40%", "col2": "88.80%", "col3": "56.07%", "col4": "0.8707", "col5": "5.41%", "rank": 2, "is_best": False}
                    ],
                    "artifacts": [
                        {"caption": "Task 1: Recommendation Engine Benchmark (50 Diverse Catalog Titles)", "src": "/task1/outputs/model_comparison.png"}
                    ],
                    "verdict": "Evaluating across 50 diverse catalog titles reveals that feature weighting achieves a +40.5% boost in Jaccard genre overlap (96.57% vs 56.07%) and near-perfect NDCG@10 (0.9974), ranking the most relevant content first."
                },
                "task2": {
                    "task_id": "Task 2",
                    "title": "Content Format Binary Classification (Movie vs. TV Show)",
                    "subtitle": "Unseen holdout test split (1,758 samples) with zero duration leakage and zero genre label leakage",
                    "sample_info": "80/20 Stratified Split (7,029 train / 1,758 test) • 5-Fold Stratified CV Verified",
                    "columns": ["Model", "Accuracy", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Macro F1 Score", "Cohen Kappa"],
                    "rows": [
                        {"model": "Linear SVM", "col1": "98.75%", "col2": "98.52%", "col3": "98.52%", "col4": "98.52%", "col5": "0.9852", "col6": "0.9704", "rank": 1, "is_best": True},
                        {"model": "Decision Tree", "col1": "98.69%", "col2": "98.48%", "col3": "98.43%", "col4": "98.48%", "col5": "0.9845", "col6": "0.9691", "rank": 2, "is_best": False},
                        {"model": "Logistic Regression", "col1": "98.41%", "col2": "98.22%", "col3": "98.02%", "col4": "98.22%", "col5": "0.9812", "col6": "0.9624", "rank": 3, "is_best": False},
                        {"model": "Random Forest", "col1": "97.78%", "col2": "97.56%", "col3": "97.21%", "col4": "97.56%", "col5": "0.9738", "col6": "0.9477", "rank": 4, "is_best": False}
                    ],
                    "artifacts": [
                        {"caption": "Task 2: Linear SVM Format Binary Confusion Matrix (Holdout Test)", "src": "/task2/models/confusion_matrix.png"}
                    ],
                    "verdict": "After rigorously purging all format token leakage from genre tags and duration, Linear SVM achieves 98.75% accuracy and 98.52% balanced accuracy with an outstanding Cohen's Kappa of 0.9704."
                },
                "task3": {
                    "task_id": "Task 3",
                    "title": "Audience Maturity Rating Multi-Class Classification",
                    "subtitle": "Predicting 12 audience rating categories with probability confidence and top-k accuracy",
                    "sample_info": "80/20 Stratified Split (7,032 train / 1,758 test) • 12 Target Classes • 3-Fold Stratified CV",
                    "columns": ["Model", "Top-1 Accuracy", "Top-2 Accuracy", "Top-3 Accuracy", "Balanced Accuracy", "Macro F1 Score", "Cohen Kappa"],
                    "rows": [
                        {"model": "Tuned Random Forest", "col1": "50.63%", "col2": "76.79%", "col3": "89.48%", "col4": "41.72%", "col5": "0.3942", "col6": "0.3691", "rank": 1, "is_best": True},
                        {"model": "Random Forest (Baseline)", "col1": "42.72%", "col2": "65.42%", "col3": "77.59%", "col4": "40.64%", "col5": "0.3260", "col6": "0.3130", "rank": 2, "is_best": False},
                        {"model": "Extra Trees Classifier", "col1": "41.52%", "col2": "64.33%", "col3": "76.73%", "col4": "38.08%", "col5": "0.3149", "col6": "0.3026", "rank": 3, "is_best": False},
                        {"model": "Decision Tree Classifier", "col1": "37.88%", "col2": "60.13%", "col3": "63.71%", "col4": "35.31%", "col5": "0.3085", "col6": "0.2474", "rank": 4, "is_best": False}
                    ],
                    "artifacts": [
                        {"caption": "Task 3: Netflix Audience Rating Category Distribution", "src": "/task3/outputs/rating_distribution.png"},
                        {"caption": "Task 3: Tuned Random Forest Multi-Class Confusion Matrix", "src": "/task3/outputs/confusion_matrix.png"}
                    ],
                    "verdict": "For 12 imbalanced content ratings, hyperparameter tuning boosts Macro F1 by +20.9%. Critically, Top-2 Accuracy reaches 76.79% and Top-3 reaches 89.48%, demonstrating reliable boundary estimation between adjacent age brackets."
                },
                "task4": {
                    "task_id": "Task 4",
                    "title": "Catalog Segmentation Multi-Index Optimization",
                    "subtitle": "K-Means cluster cohesion, separation, and variance evaluation across K = 2 to 8",
                    "sample_info": "8,787 Netflix titles clustered over preprocessed metadata • Multi-Index Validation",
                    "columns": ["Clusters (K)", "Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz", "Inertia (SSE)", "Partition Quality"],
                    "rows": [
                        {"model": "K = 3", "col1": "0.3015", "col2": "1.294 (Optimal Min)", "col3": "2945.4", "col4": "28864.9", "col5": "Global Optimal Cohesion", "rank": 1, "is_best": True},
                        {"model": "K = 2", "col1": "0.2984", "col2": "1.355", "col3": "3606.8", "col4": "34186.9", "col5": "Sub-optimal separation", "rank": 2, "is_best": False},
                        {"model": "K = 5", "col1": "0.1707", "col2": "2.075", "col3": "1901.5", "col4": "25841.5", "col5": "Over-fragmented subgenres", "rank": 3, "is_best": False},
                        {"model": "K = 4", "col1": "0.1570", "col2": "2.032", "col3": "2269.8", "col4": "27163.2", "col5": "Ambiguous boundaries", "rank": 4, "is_best": False},
                        {"model": "K = 6", "col1": "0.1303", "col2": "2.271", "col3": "1682.4", "col4": "24629.1", "col5": "Fragmented micro-clusters", "rank": 5, "is_best": False},
                        {"model": "K = 8", "col1": "0.1199", "col2": "2.293", "col3": "1379.7", "col4": "22962.0", "col5": "High intra-cluster noise", "rank": 6, "is_best": False},
                        {"model": "K = 7", "col1": "0.1194", "col2": "2.325", "col3": "1476.9", "col4": "23999.7", "col5": "Worst separation index", "rank": 7, "is_best": False}
                    ],
                    "artifacts": [
                        {"caption": "Task 4: Truncated SVD 2D Projection of Content Clusters", "src": "/task4/outputs/cluster_visualization.png"},
                        {"caption": "Task 4: Format Composition Across Segments", "src": "/task4/outputs/cluster_content_types.png"},
                        {"caption": "Task 4: Cluster Population Distribution", "src": "/task4/outputs/cluster_sizes.png"}
                    ],
                    "verdict": "K = 3 is mathematically confirmed by simultaneously attaining the global minimum Davies-Bouldin index (1.294, where lower is better) and peak multi-cluster Silhouette score (0.3015), creating distinct Feature Film, Episodic Series, and Vintage Archive segments."
                }
            }
            self.send_json_response(eval_data)
            return

        # Fallback to standard static file serving
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)
        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            payload = {}

        if path == "/api/task1/recommend":
            title = payload.get("title", "").strip()
            top_n = int(payload.get("top_n", 10))

            if DF is None or PROCESSED_DF is None or WEIGHTED_MATRIX is None:
                self.send_json_response({"error": "Recommendation engine not initialized"}, 500)
                return

            # Lookup title in processed_df
            matches = PROCESSED_DF[PROCESSED_DF["title"].astype(str).str.casefold() == title.casefold()]
            if matches.empty:
                self.send_json_response({"error": f"Title '{title}' not found in catalog."}, 404)
                return

            orig_row = matches.iloc[0]
            pos_idx = PROCESSED_DF.index.get_loc(matches.index[0])

            # Cosine similarity calculation
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity(WEIGHTED_MATRIX[pos_idx], WEIGHTED_MATRIX).flatten()
            sorted_indices = scores.argsort()[::-1]
            sorted_indices = [i for i in sorted_indices if i != pos_idx][:top_n]

            recs = DF.iloc[sorted_indices][["title", "type", "director", "country", "release_year", "rating", "duration", "listed_in"]].copy()
            recs["similarity_score"] = [round(float(scores[i]), 4) for i in sorted_indices]

            self.send_json_response({
                "query_title": orig_row["title"],
                "query_type": orig_row["type"],
                "query_genres": orig_row["listed_in"],
                "recommendations": recs.to_dict(orient="records")
            })
            return

        elif path == "/api/task2/predict-type":
            if TASK2_MODEL is None:
                self.send_json_response({"error": "Task 2 model not available"}, 500)
                return

            director = str(payload.get("director", "unknown")).strip().lower()
            country = str(payload.get("country", "united states")).strip().lower()
            release_year = int(payload.get("release_year", 2023))
            rating = str(payload.get("rating", "tv-ma")).strip().lower()
            listed_in = str(payload.get("listed_in", "")).strip().lower()

            input_df = pd.DataFrame([{
                "director": director,
                "country": country,
                "release_year": release_year,
                "rating": rating,
                "listed_in": listed_in
            }])

            try:
                pred = TASK2_MODEL.predict(input_df)[0]
                confidence = None
                if hasattr(TASK2_MODEL, "predict_proba"):
                    probs = TASK2_MODEL.predict_proba(input_df)[0]
                    confidence = round(float(probs.max()) * 100, 1)

                self.send_json_response({
                    "predicted_type": pred,
                    "confidence": confidence,
                    "inputs": payload
                })
            except Exception as e:
                self.send_json_response({"error": str(e)}, 500)
            return

        elif path == "/api/task3/predict-rating":
            if TASK3_MODEL is None:
                self.send_json_response({"error": "Task 3 model not available"}, 500)
                return

            content_type = str(payload.get("type", "Movie")).strip()
            director = str(payload.get("director", "unknown")).strip()
            country = str(payload.get("country", "United States")).strip()
            release_year = int(payload.get("release_year", 2023))
            listed_in = str(payload.get("listed_in", "")).strip()
            duration = str(payload.get("duration", "90 min")).strip()

            # Prepare feature frame for Task 3
            sys_path_backup = list(sys.path)
            for m in list(sys.modules.keys()):
                if m == "src" or m.startswith("src."):
                    del sys.modules[m]
            sys.path.insert(0, str((ROOT_DIR / "task3").resolve()))
            from src.preprocessing import prepare_features

            input_df = pd.DataFrame([{
                "type": content_type,
                "director": director,
                "country": country,
                "release_year": release_year,
                "listed_in": listed_in,
                "duration": duration
            }])

            feats = prepare_features(input_df)
            sys.path = sys_path_backup

            try:
                pred = TASK3_MODEL.predict(feats)[0]
                confidence = None
                class_probs = []
                if hasattr(TASK3_MODEL, "predict_proba"):
                    probs = TASK3_MODEL.predict_proba(feats)[0]
                    classes = TASK3_MODEL.named_steps["classifier"].classes_
                    confidence = round(float(probs.max()) * 100, 1)
                    # Top 3 probabilities
                    top_idx = probs.argsort()[::-1][:4]
                    class_probs = [{"rating": str(classes[i]), "probability": round(float(probs[i]) * 100, 1)} for i in top_idx]

                self.send_json_response({
                    "predicted_rating": pred,
                    "confidence": confidence,
                    "top_probabilities": class_probs,
                    "inputs": payload
                })
            except Exception as e:
                self.send_json_response({"error": str(e)}, 500)
            return

        elif path == "/api/task4/assign-cluster":
            if TASK4_MODEL is None:
                self.send_json_response({"error": "Task 4 model not available"}, 500)
                return

            content_type = str(payload.get("type", "Movie")).strip()
            country = str(payload.get("country", "United States")).strip()
            rating = str(payload.get("rating", "TV-MA")).strip()
            listed_in = str(payload.get("listed_in", "")).strip()
            release_year = int(payload.get("release_year", 2023))
            duration = str(payload.get("duration", "90 min")).strip()

            # Prepare feature frame for Task 4
            sys_path_backup = list(sys.path)
            for m in list(sys.modules.keys()):
                if m == "src" or m.startswith("src."):
                    del sys.modules[m]
            sys.path.insert(0, str((ROOT_DIR / "task4").resolve()))
            from src.preprocessing import prepare_feature_frame

            input_df = pd.DataFrame([{
                "type": content_type,
                "country": country,
                "rating": rating,
                "listed_in": listed_in,
                "release_year": release_year,
                "duration": duration
            }])

            feats = prepare_feature_frame(input_df)
            sys.path = sys_path_backup

            try:
                preprocessor = TASK4_MODEL["preprocessor"]
                kmeans = TASK4_MODEL["kmeans"]
                X_trans = preprocessor.transform(feats)
                cluster_id = int(kmeans.predict(X_trans)[0])

                cluster_names = {
                    0: "Cluster 0: Modern Feature Films",
                    1: "Cluster 1: Binge-Worthy TV & Episodic Content",
                    2: "Cluster 2: Vintage Cinema & Classic Vault"
                }

                self.send_json_response({
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
                    "inputs": payload
                })
            except Exception as e:
                self.send_json_response({"error": str(e)}, 500)
            return

        self.send_json_response({"error": "Endpoint not found"}, 404)


def run(port=8080):
    init_ml_models()
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, AuspifyRequestHandler)
    print(f"\n=======================================================")
    print(f" Auspify Netflix ML Studio Web Server is Live!")
    print(f" Access URL: http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run(port)
