# TNL6323 — Sentiment Analysis Deployment
## Step 5: Web Application (Flask)

### What this is
A web application that lets anyone type or paste a Malaysian app review and instantly see whether it is **Positive**, **Neutral**, or **Negative**, along with confidence scores and detected emojis.

---

### Folder structure
```
deployment/
├── app.py                  ← Flask backend (the server)
├── requirements.txt        ← Python packages needed
├── README.md               ← This file
├── models/
│   ├── best_sentiment_model.joblib   ← Trained Naive Bayes model
│   └── tfidf_vectorizer.joblib       ← Fitted TF-IDF vectorizer
└── templates/
    └── index.html          ← Web interface (frontend)
```

---

### How to run

**Step 1 — Install Python packages**
```bash
pip install -r requirements.txt
```

**Step 2 — Download NLTK data (one-time, auto-done on first run)**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Step 3 — Start the server**
```bash
python app.py
```

**Step 4 — Open your browser**
```
http://localhost:5000
```

---

### How to use the app
1. Select the **language** of the review (English or Bahasa Malaysia).
2. **Type or paste** an app review in the text box.
3. Click **Analyse Sentiment** (or press `Ctrl+Enter`).
4. The result shows:
   - **Sentiment label** — Positive / Neutral / Negative
   - **Confidence bars** — probability score for each class
   - **Processed tokens** — what the model actually sees after preprocessing
   - **Emojis detected** — emojis extracted from the original text

---

### API endpoint (for developers)
`POST /predict`

Request body (JSON):
```json
{
  "text": "This app is amazing!",
  "lang": "en"
}
```

Response (JSON):
```json
{
  "sentiment": "Positive",
  "confidence": { "Positive": 51.2, "Neutral": 27.4, "Negative": 21.4 },
  "processed": "app amaz",
  "emojis": [],
  "model": "MultinomialNB"
}
```

---

### Model info
| Model        | Best Parameters          | Test Accuracy |
|--------------|--------------------------|---------------|
| Naive Bayes  | alpha=2.0 *(best)*       | 52.2%         |
| Log. Regress.| C=0.1                    | 50.0%         |
| Linear SVM   | C=0.01                   | 50.0%         |
| Random Forest| n_estimators=200         | 52.2%         |
| XGBoost      | lr=0.01, max_depth=5     | 42.2%         |

The best model (**Naive Bayes**) was automatically saved during training.
