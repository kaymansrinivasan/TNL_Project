"""
TNL6323 - Malaysia Sentiment Analysis System
Step 5: Deployment (Flask Web Application)

Run: python app.py
Then open: http://localhost:5000
"""

import re
import string
import os
import joblib
import emoji
import nltk
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# ── one-time NLTK downloads ──────────────────────────────────────────────────
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(pkg, quiet=True)

# ── try to import Sastrawi (Malay stemmer); graceful fallback ───────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    ms_stemmer  = StemmerFactory().create_stemmer()
    MS_STOP     = set(StopWordRemoverFactory().get_stop_words())
    SASTRAWI_OK = True
except ImportError:
    ms_stemmer  = None
    MS_STOP     = set()
    SASTRAWI_OK = False

EN_STOP        = set(stopwords.words("english"))
MANGLISH_STOP  = {"lah","lor","meh","wor","ah","ke","kan","je","tu","ni"}
ALL_STOP       = EN_STOP | MS_STOP | MANGLISH_STOP
en_stemmer     = SnowballStemmer("english")

# ── label maps ───────────────────────────────────────────────────────────────
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
APP_LABELS = {
    "tng":  "Touch 'n Go eWallet",
    "grab": "Grab",
    "mys":  "MySejahtera",
}

# ── load model & vectorizer ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
model     = joblib.load(os.path.join(BASE, "models", "best_sentiment_model.joblib"))
vectorizer = joblib.load(os.path.join(BASE, "models", "tfidf_vectorizer.joblib"))

# ── preprocessing (mirrors Preprocess.py) ───────────────────────────────────
def extract_emojis(text: str) -> list:
    return emoji.distinct_emoji_list(str(text))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_text(text: str, lang: str = "en") -> str:
    cleaned = clean_text(text)
    tokens  = word_tokenize(cleaned)
    tokens  = [t for t in tokens if t.isalpha() and t not in ALL_STOP and len(t) > 1]
    if lang == "ms" and SASTRAWI_OK:
        stemmed = ms_stemmer.stem(" ".join(tokens)).split()
    else:
        stemmed = [en_stemmer.stem(t) for t in tokens]
    return " ".join(stemmed)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")   # "en" or "ms"

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    emojis_found  = extract_emojis(text)
    processed     = process_text(text, lang)
    features      = vectorizer.transform([processed])
    pred_num      = model.predict(features)[0]

    # probability scores (Naive Bayes supports predict_proba)
    try:
        proba = model.predict_proba(features)[0].tolist()
        confidence = {LABEL_MAP[i]: round(p * 100, 1) for i, p in enumerate(proba)}
    except Exception:
        confidence = {LABEL_MAP[pred_num]: 100.0}

    return jsonify({
        "sentiment":   LABEL_MAP[pred_num],
        "confidence":  confidence,
        "processed":   processed,
        "emojis":      emojis_found,
        "model":       type(model).__name__,
    })

if __name__ == "__main__":
    print("=" * 55)
    print("  TNL6323 Sentiment Analysis System")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
