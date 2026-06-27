import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
from xgboost import XGBClassifier

# Create output directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def load_data():
    print("--- Loading and Preparing Data ---")
    if not os.path.exists("dataset_preprocessed.csv"):
        raise FileNotFoundError("dataset_preprocessed.csv not found! Please run Step 2 (Preprocess.py) first.")
    
    df = pd.read_csv("dataset_preprocessed.csv")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check for NaN values in processed_text
    nan_count = df["processed_text"].isna().sum()
    if nan_count > 0:
        print(f"Found {nan_count} missing values in 'processed_text'. Filling them with empty strings.")
        df["processed_text"] = df["processed_text"].fillna("")
    
    # Class distribution
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())
    
    # Map labels to integers
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df["label_num"] = df["label"].map(label_mapping)
    
    return df

def main():
    # Set seed for reproducibility
    seed = 42
    
    # Load dataset
    df = load_data()
    
    X = df["processed_text"]
    y = df["label_num"]
    
    # Train-test split (stratified to maintain label balances)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"\nTrain size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")
    
    # Feature Extraction: TF-IDF Vectorizer
    print("\n--- Extracting Features (TF-IDF) ---")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # unigrams and bigrams
        sublinear_tf=True,   # log-scaling TF values
        min_df=2             # drop terms that appear in less than 2 reviews
    )
    
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # Save the TF-IDF vectorizer
    vectorizer_path = "models/tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved fitted TfidfVectorizer -> {vectorizer_path}")
    print(f"Vocabulary size: {X_train.shape[1]} features")
    
    # Define models and their parameter grids for GridSearchCV
    model_configs = {
        "Naive Bayes": {
            "model": MultinomialNB(),
            "params": {
                "alpha": [0.01, 0.1, 0.5, 1.0, 2.0]
            }
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=seed),
            "params": {
                "C": [0.1, 0.5, 1.0, 5.0, 10.0, 20.0],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "Linear SVM": {
            "model": LinearSVC(random_state=seed, dual="auto"),
            "params": {
                "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=seed),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric="mlogloss"),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        }
    }
    
    results = []
    trained_models = {}
    
    best_overall_f1 = -1.0
    best_overall_model_name = None
    best_overall_model = None
    
    print("\n--- Starting Model Training and Hyperparameter Tuning ---")
    for name, config in model_configs.items():
        print(f"\nTuning {name}...")
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring="f1_macro",
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_
        
        print(f"Best CV F1-Macro: {best_cv_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        
        # Calculate test metrics
        test_acc = accuracy_score(y_test, y_pred)
        
        # Precision, Recall, F1 (Macro & Weighted)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
        
        print(f"Test Accuracy: {test_acc:.4f} | Test F1-Macro: {f1_macro:.4f}")
        
        # Store results
        results.append({
            "Model": name,
            "Best Parameters": str(best_params),
            "CV F1-Macro": best_cv_score,
            "Test Accuracy": test_acc,
            "Test F1-Macro": f1_macro,
            "Test F1-Weighted": f1_weighted,
            "Test Precision-Macro": p_macro,
            "Test Recall-Macro": r_macro
        })
        
        trained_models[name] = {
            "model": best_model,
            "predictions": y_pred,
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        
        # Check if it is the best overall model (prioritizing Test F1-Macro)
        if f1_macro > best_overall_f1:
            best_overall_f1 = f1_macro
            best_overall_model_name = name
            best_overall_model = best_model
            
    # Save results to df
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/model_comparison.csv", index=False)
    print("\nSaved comparison report -> results/model_comparison.csv")
    
    # Save the best model
    best_model_path = "models/best_sentiment_model.joblib"
    joblib.dump(best_overall_model, best_model_path)
    print(f"\nSaved Best Model ({best_overall_model_name}) -> {best_model_path}")
    
    # Output the table to console
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY (Sorted by Test F1-Macro)")
    print("="*80)
    print(results_df.sort_values(by="Test F1-Macro", ascending=False).to_string(index=False))
    print("="*80)
    
    # --- Visualization Section ---
    print("\n--- Generating Visual Reports ---")
    
    # 1. Plot Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    plot_df = results_df.melt(id_vars="Model", value_vars=["Test Accuracy", "Test F1-Macro", "Test F1-Weighted"],
                             var_name="Metric", value_name="Score")
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Model Sentiment Analysis Performance Comparison")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    comparison_chart_path = "results/model_comparison_bar.png"
    plt.savefig(comparison_chart_path)
    plt.close()
    print(f"Saved comparative bar chart -> {comparison_chart_path}")
    
    # 2. Plot Confusion Matrix Grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    labels = ["Negative", "Neutral", "Positive"]
    
    for idx, (name, details) in enumerate(trained_models.items()):
        cm = details["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[idx])
        axes[idx].set_title(f"{name} Confusion Matrix")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
        
    # Disable unused subplot (since we have 5 models but a 2x3 grid = 6 subplots)
    axes[5].axis("off")
    plt.tight_layout()
    cm_grid_path = "results/confusion_matrices_grid.png"
    plt.savefig(cm_grid_path)
    plt.close()
    print(f"Saved confusion matrix grid -> {cm_grid_path}")
    
    # 3. Output classification report for the best model
    best_y_pred = trained_models[best_overall_model_name]["predictions"]
    print(f"\nDetailed Classification Report for the Best Model ({best_overall_model_name}):")
    report = classification_report(y_test, best_y_pred, target_names=["Negative", "Neutral", "Positive"])
    print(report)
    
    with open("results/best_model_classification_report.txt", "w") as f:
        f.write(f"Best Model: {best_overall_model_name}\n")
        f.write(f"Parameters: {best_overall_model.get_params()}\n\n")
        f.write(report)
    print("Saved detailed report -> results/best_model_classification_report.txt")

if __name__ == "__main__":
    main()
