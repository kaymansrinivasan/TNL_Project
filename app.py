"""
TNL6323 - Malaysia-Focused Sentiment Analysis
Step: Deployment  (web app)

A Streamlit app that loads the trained model + both TF-IDF vectorizers and
predicts the sentiment of a Malaysian app review (English or Malay).

IMPORTANT: it reuses preprocess.py and emoji_features.py so a new review is
processed EXACTLY the way the training data was (clean -> stem -> word TF-IDF
+ char TF-IDF + emoji features, in that order). If you change preprocessing,
the app updates automatically.

SETUP:  pip install streamlit
RUN:    streamlit run app.py
"""

import joblib
import emoji as emoji_lib
import streamlit as st
from scipy.sparse import hstack, csr_matrix

from Preprocess import process_row          # same cleaning/stemming as training
from emoji_features import build_emoji_matrix, emoji_features

# model predicts 0/1/2  (mapping from train_models.py)
LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}
EMOJIS = {0: "😞", 1: "😐", 2: "😊"}

MS_HINTS = {"tak", "tidak", "boleh", "saya", "nak", "dan", "yang", "ini", "tapi",
            "guna", "susah", "bagus", "apps", "sangat", "dah", "kenapa", "buat"}


@st.cache_resource
def load_artifacts():
    word_vec = joblib.load("models/tfidf_word.joblib")
    char_vec = joblib.load("models/tfidf_char.joblib")
    model = joblib.load("models/best_sentiment_model.joblib")
    return word_vec, char_vec, model


def guess_lang(text: str) -> str:
    toks = set(text.lower().split())
    return "ms" if len(toks & MS_HINTS) >= 2 else "en"


def predict(text, word_vec, char_vec, model):
    lang = guess_lang(text)
    processed = process_row(text, lang)                 # clean + stem
    w = word_vec.transform([processed])                 # word TF-IDF
    c = char_vec.transform([processed])                 # char TF-IDF
    e = csr_matrix(build_emoji_matrix([text]))          # emoji features
    X = hstack([w, c, e]).tocsr()                       # SAME order as training
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    return pred, proba, processed, lang


# ----------------------------------------------------------------------
st.set_page_config(page_title="Malaysian Review Sentiment", page_icon="📱")

st.title("📱 Malaysian E-Services Sentiment Analyzer")
st.caption("TNL6323 · Touch 'n Go eWallet · MySejahtera · Grab · English + Bahasa Melayu")

word_vec, char_vec, model = load_artifacts()

st.write("Type a Malaysian app review and get its predicted sentiment.")
text = st.text_area(
    "Review text",
    height=120,
    placeholder="e.g. app ni sangat bagus dan senang guna 👍  /  cannot login, keep crashing 😡",
)

col1, col2 = st.columns(2)
examples = {
    "😊 Positive (BM)": "app ni memang terbaik, senang guna dan cepat 👍",
    "😡 Negative (EN)": "worst update ever, cannot login and keeps crashing",
}
for (label, ex), col in zip(examples.items(), (col1, col2)):
    if col.button(label):
        text = ex
        st.session_state["review"] = ex

if st.button("Analyze sentiment", type="primary") and text.strip():
    pred, proba, processed, lang = predict(text, word_vec, char_vec, model)

    st.markdown(
        f"<h2 style='color:{COLORS[pred]}'>{EMOJIS[pred]} {LABELS[pred]}</h2>",
        unsafe_allow_html=True,
    )
    if proba is not None:
        st.write(f"**Confidence: {proba[pred] * 100:.1f}%**")
        st.bar_chart({LABELS[i]: [float(proba[i])] for i in range(3)})

    with st.expander("How it was processed"):
        st.write(f"**Detected language:** {'Malay' if lang == 'ms' else 'English'}")
        st.write(f"**Cleaned / stemmed text:** `{processed or '(empty)'}`")
        found = "".join(emoji_lib.distinct_emoji_list(text))
        st.write(f"**Emojis detected:** {found or '(none)'}  "
                 f"→ features {emoji_features(text)}")

elif text == "":
    st.info("Enter a review above, or try one of the example buttons.")

st.divider()
st.caption("Model: TF-IDF (word + char) + emoji features → best classifier. "
           "Trained on 360+ hand-verified Malaysian reviews.")