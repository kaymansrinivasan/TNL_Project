"""
TNL6323 - Malaysia-Focused Sentiment Analysis
Step 2: Data Pre-processing   (Member 1)

Input : dataset_360.csv   (from Step 1 - columns: text, label, app, language, ...)
Output: dataset_preprocessed.csv

Pipeline per review:
    1. Extract emojis into their own column  <-- for Member 4's emoji feature
       (done FIRST, before we clean them away)
    2. Clean text   (lowercase, strip URLs / @mentions / digits / punctuation,
                     squash repeated chars: "baguuuss" -> "bagus")
    3. (optional) Translate Malay -> English   [TRANSLATE = False by default]
    4. Tokenize
    5. Remove stopwords   (English + Malay)
    6. Stem              (English: Snowball, Malay: Sastrawi)
    7. Join back -> processed_text   (the column Member 2 feeds to TF-IDF)

WHY bilingual: the dataset mixes English + Malay + Manglish. We route each
row by its 'language' tag so Malay rows get Malay stemming/stopwords. A
TF-IDF + ML model learns from tokens, so translation is NOT required - it's
left as a toggle you can flip on if you want to show it as a step.

SETUP:  pip install pandas nltk PySastrawi emoji
RUN:    python preprocess.py
"""

import re
import string
import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ---- one-time NLTK data ----
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(pkg, quiet=True)

TRANSLATE = False   # set True to translate Malay -> English (needs deep-translator)

# ---- resources ----
EN_STOP = set(stopwords.words("english"))
MS_STOP = set(StopWordRemoverFactory().get_stop_words())
# a few Manglish particles worth dropping regardless of language
MANGLISH_STOP = {"lah", "lor", "meh", "wor", "ah", "ke", "kan", "je", "tu", "ni"}
ALL_STOP = EN_STOP | MS_STOP | MANGLISH_STOP

en_stemmer = SnowballStemmer("english")
ms_stemmer = StemmerFactory().create_stemmer()


# ----------------------------------------------------------------------
def extract_emojis(text: str) -> str:
    """Return a string of all emojis found (for Member 4's feature)."""
    return "".join(emoji.distinct_emoji_list(str(text)))


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace="")        # remove emojis (kept in own col)
    text = re.sub(r"http\S+|www\.\S+", " ", text)        # URLs
    text = re.sub(r"@\w+", " ", text)                    # @mentions
    text = re.sub(r"\d+", " ", text)                     # digits
    text = re.sub(r"(.)\1{2,}", r"\1", text)             # baguuuss -> bagus
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()             # extra whitespace
    return text


def translate_to_en(text: str) -> str:
    from deep_translator import GoogleTranslator   # pip install deep-translator
    try:
        return GoogleTranslator(source="auto", target="en").translate(text) or text
    except Exception:
        return text


def process_row(text: str, lang: str) -> str:
    cleaned = clean_text(text)
    if TRANSLATE and lang == "ms":
        cleaned = clean_text(translate_to_en(cleaned))
        lang = "en"

    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t.isalpha() and t not in ALL_STOP and len(t) > 1]

    stemmer = ms_stemmer if lang == "ms" else en_stemmer
    if lang == "ms":
        # Sastrawi stems a whole string most reliably
        stemmed = ms_stemmer.stem(" ".join(tokens)).split()
    else:
        stemmed = [en_stemmer.stem(t) for t in tokens]

    return " ".join(stemmed)


# ----------------------------------------------------------------------
def main():
    df = pd.read_csv("dataset_360.csv")

    df["emojis"] = df["text"].apply(extract_emojis)
    df["clean_text"] = df["text"].apply(clean_text)
    df["processed_text"] = [
        process_row(t, l) for t, l in zip(df["text"], df.get("language", "en"))
    ]

    # drop rows that became empty after processing (safety)
    empty = (df["processed_text"].str.strip() == "").sum()
    if empty:
        print(f"Note: {empty} row(s) empty after processing - review these.")

    cols = ["text", "clean_text", "processed_text", "emojis",
            "label", "app", "language"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv("dataset_preprocessed.csv", index=False, encoding="utf-8-sig")

    print(f"Saved -> dataset_preprocessed.csv ({len(df)} rows)")
    print("\nLabel balance:")
    print(df["label"].value_counts().to_string())
    print("\nSample:")
    for _, r in df.head(3).iterrows():
        print(f"\n[{r['label']}|{r.get('language','?')}] {r['text'][:60]}")
        print(f"   clean    : {r['clean_text'][:60]}")
        print(f"   processed: {r['processed_text'][:60]}")
        print(f"   emojis   : {r['emojis']}")


if __name__ == "__main__":
    main()