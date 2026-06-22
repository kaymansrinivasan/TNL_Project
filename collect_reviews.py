"""
TNL6323 - Malaysia-Focused Sentiment Analysis
Step 1: Data Collection

Scrapes Google Play reviews for 3 Malaysian apps (Touch 'n Go eWallet,
MySejahtera, Grab) and saves them to a CSV with a provisional sentiment
label derived from the star rating.

WHY BY-RATING: app reviews skew heavily positive/negative, so neutral is
hard to find. We pull each star level (1-5) separately so the neutral
(3-star) bucket is actually filled. You then OVER-collect and hand-curate
down to a clean, balanced 120 / 120 / 120.

------------------------------------------------------------------
SETUP (run once in your terminal):
    pip install google-play-scraper pandas
RUN:
    python collect_reviews.py
OUTPUT:
    raw_reviews.csv   (raw, over-collected, provisional labels)
------------------------------------------------------------------
"""

import time
import pandas as pd
from google_play_scraper import reviews, Sort

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
APPS = {
    "Touch n Go eWallet": "my.com.tngdigital.ewallet",
    "MySejahtera":        "my.gov.onegovappstore.mysejahtera",
    "Grab":               "com.grabtaxi.passenger",
}

# Pull both English and Malay reviews -> captures Manglish / BM,
# which strengthens the "Malaysia-focused" angle in your report.
LANGS = ["en", "ms"]

COUNTRY = "my"              # Malaysia store
REVIEWS_PER_SCORE = 80     # per app, per star, per language (generous buffer)

# Star rating -> provisional sentiment label
def rating_to_label(score: int) -> str:
    if score >= 4:
        return "positive"
    if score == 3:
        return "neutral"
    return "negative"        # 1 or 2 stars


# ----------------------------------------------------------------------
# SCRAPE
# ----------------------------------------------------------------------
def collect():
    rows = []
    for app_name, app_id in APPS.items():
        for lang in LANGS:
            for score in [1, 2, 3, 4, 5]:
                try:
                    result, _ = reviews(
                        app_id,
                        lang=lang,
                        country=COUNTRY,
                        sort=Sort.NEWEST,
                        count=REVIEWS_PER_SCORE,
                        filter_score_with=score,   # <-- pull this star only
                    )
                except Exception as e:
                    print(f"  ! {app_name} [{lang}|{score}*] failed: {e}")
                    continue

                for r in result:
                    text = (r.get("content") or "").strip()
                    if not text:
                        continue
                    rows.append({
                        "review_id":  r.get("reviewId"),
                        "app":        app_name,
                        "text":       text,
                        "rating":     r.get("score"),
                        "label":      rating_to_label(r.get("score")),
                        "language":   lang,
                        "thumbs_up":  r.get("thumbsUpCount"),
                        "date":       r.get("at"),
                        "source":     "Google Play",
                    })

                print(f"  {app_name:20s} [{lang}|{score}*] -> {len(result)} reviews")
                time.sleep(1)   # be polite, avoid hammering

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# CLEAN + SAVE
# ----------------------------------------------------------------------
def main():
    print("Collecting reviews...\n")
    df = collect()

    # de-duplicate (same review can appear across en/ms pulls)
    before = len(df)
    df = df.drop_duplicates(subset="review_id")
    df = df.drop_duplicates(subset="text")
    # drop ultra-short noise like "ok", "good", emojis only -> keep >= 3 words
    df = df[df["text"].str.split().str.len() >= 3].reset_index(drop=True)
    print(f"\nDe-duplicated / filtered: {before} -> {len(df)} reviews")

    df.to_csv("raw_reviews.csv", index=False, encoding="utf-8-sig")
    print("Saved -> raw_reviews.csv")

    # ------------------------------------------------------------------
    # Summary so you can see if each class has enough to curate from
    # ------------------------------------------------------------------
    print("\n=== Provisional label counts (target: 120 each after curation) ===")
    print(df["label"].value_counts().to_string())
    print("\n=== Per app x label ===")
    print(pd.crosstab(df["app"], df["label"]).to_string())


if __name__ == "__main__":
    main()