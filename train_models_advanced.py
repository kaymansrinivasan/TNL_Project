import os
import joblib
import pandas as pd
import numpy as np

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

SEED = 42

POSITIVE_EMOJIS = set([
    "😀", "😃", "😄", "😁", "😊", "😍", "🥰", "😘", "😎",
    "👍", "👌", "👏", "🙌", "❤️", "♥️", "💖", "💯", "🔥",
    "✅", "✨", "⭐", "🌟"
])

NEGATIVE_EMOJIS = set([
    "😡", "😠", "🤬", "😭", "😢", "😞", "😔", "😤", "😩",
    "👎", "❌", "💔", "😒", "🙄", "😕", "😟", "😫"
])

NEUTRAL_EMOJIS = set([
    "😐", "😶", "🤔", "😑", "🙂"
])


def emoji_features(emoji_text):
    emoji_text = str(emoji_text)

    pos_count = sum(1 for e in emoji_text if e in POSITIVE_EMOJIS)
    neg_count = sum(1 for e in emoji_text if e in NEGATIVE_EMOJIS)
    neu_count = sum(1 for e in emoji_text if e in NEUTRAL_EMOJIS)

    score = pos_count - neg_count
    has_emoji = 1 if len(emoji_text.strip()) > 0 else 0

    return pd.Series({
        "emoji_positive_count": pos_count,
        "emoji_negative_count": neg_count,
        "emoji_neutral_count": neu_count,
        "emoji_score": score,
        "has_emoji": has_emoji
    })


def load_data():
    df = pd.read_csv("dataset_preprocessed.csv")

    df["processed_text"] = df["processed_text"].fillna("")
    df["emojis"] = df["emojis"].fillna("")

    label_mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    df["label_num"] = df["label"].map(label_mapping)

    if df["label_num"].isna().any():
        raise ValueError("Some labels are not mapped correctly. Check label names.")

    emoji_df = df["emojis"].apply(emoji_features)
    df = pd.concat([df, emoji_df], axis=1)

    pd.DataFrame({
        "label": ["negative", "neutral", "positive"],
        "encoded_value": [0, 1, 2]
    }).to_csv("results/label_mapping.csv", index=False)

    return df


def evaluate_models(X_train, X_test, y_train, y_test, experiment_name):
    model_configs = {
        "Naive Bayes": {
            "model": MultinomialNB(),
            "params": {
                "alpha": [0.01, 0.1, 0.5, 1.0, 2.0]
            }
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=SEED),
            "params": {
                "C": [0.1, 0.5, 1.0, 5.0, 10.0],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "Linear SVM": {
            "model": LinearSVC(random_state=SEED),
            "params": {
                "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=SEED),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        }
    }

    results = []
    best_f1 = -1
    best_model = None
    best_name = None
    best_pred = None

    for name, config in model_configs.items():
        print(f"\nTraining {experiment_name}: {name}")

        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring="f1_macro",
            cv=5,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        results.append({
            "Experiment": experiment_name,
            "Model": name,
            "Best Parameters": str(grid.best_params_),
            "CV F1-Macro": grid.best_score_,
            "Test Accuracy": acc,
            "Test Precision-Macro": p_macro,
            "Test Recall-Macro": r_macro,
            "Test F1-Macro": f1_macro,
            "Test F1-Weighted": f1_weighted
        })

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model = model
            best_name = name
            best_pred = y_pred

    report = classification_report(
        y_test,
        best_pred,
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    )

    with open(f"results/{experiment_name}_best_model_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Best Model: {best_name}\n")
        f.write(str(best_model.get_params()))
        f.write("\n\n")
        f.write(report)

    joblib.dump(best_model, f"models/{experiment_name}_best_model.joblib")

    return results


def main():
    df = load_data()

    X_text = df["processed_text"]
    y = df["label_num"]

    emoji_cols = [
        "emoji_positive_count",
        "emoji_negative_count",
        "emoji_neutral_count",
        "has_emoji"
]

    X_emoji = df[emoji_cols]

    X_train_text, X_test_text, X_train_emoji, X_test_emoji, y_train, y_test = train_test_split(
        X_text,
        X_emoji,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    split_summary = pd.DataFrame({
        "Set": ["Train", "Test"],
        "Total Rows": [len(y_train), len(y_test)],
        "Negative": [(y_train == 0).sum(), (y_test == 0).sum()],
        "Neutral": [(y_train == 1).sum(), (y_test == 1).sum()],
        "Positive": [(y_train == 2).sum(), (y_test == 2).sum()]
    })
    split_summary.to_csv("results/train_test_split_summary.csv", index=False)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    X_train_emoji_sparse = csr_matrix(X_train_emoji.values)
    X_test_emoji_sparse = csr_matrix(X_test_emoji.values)

    X_train_advanced = hstack([X_train_tfidf, X_train_emoji_sparse])
    X_test_advanced = hstack([X_test_tfidf, X_test_emoji_sparse])

    feature_summary = pd.DataFrame({
        "Feature Type": ["TF-IDF text features", "Emoji sentiment features", "Total advanced features"],
        "Number of Features": [
            X_train_tfidf.shape[1],
            len(emoji_cols),
            X_train_advanced.shape[1]
        ]
    })
    feature_summary.to_csv("results/feature_encoding_summary.csv", index=False)

    all_results = []

    baseline_results = evaluate_models(
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        "baseline_tfidf_only"
    )
    all_results.extend(baseline_results)

    advanced_results = evaluate_models(
        X_train_advanced,
        X_test_advanced,
        y_train,
        y_test,
        "advanced_tfidf_plus_emoji"
    )
    all_results.extend(advanced_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/model_comparison_baseline_vs_advanced.csv", index=False)

    print("\nSaved:")
    print("results/train_test_split_summary.csv")
    print("results/label_mapping.csv")
    print("results/feature_encoding_summary.csv")
    print("results/model_comparison_baseline_vs_advanced.csv")


if __name__ == "__main__":
    main()