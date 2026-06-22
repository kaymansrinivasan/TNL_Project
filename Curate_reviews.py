"""
TNL6323 - Malaysia-Focused Sentiment Analysis
Step 1: Data Collection (curation stage)

Takes raw_reviews.csv and prepares a BALANCED set for manual verification.
Workflow:
    1. Run this with MODE = "sample"  -> creates to_curate.csv
    2. Open to_curate.csv in Excel. For each row:
         - check 'verified_label' is correct (fix wrong ones)
         - set 'keep' to 0 to drop a row (junk, duplicate-ish, unclear)
       Aim to leave >= 120 good rows per label.
    3. Run this with MODE = "finalise" -> creates dataset_360.csv (exactly 120/120/120)

WHY: star ratings are only a proxy. A 3-star review is often actually
positive or negative in tone. Verifying by hand = the labeling discipline
that earns the Data Collection marks (and a defensible dataset for training).
"""

import pandas as pd

MODE = "sample"        # "sample" first, then change to "finalise"

PER_LABEL_BUFFER = 150  # sample a bit over 120 so you can drop bad rows
TARGET_PER_LABEL = 120  # final count per class
SEED = 42


# ----------------------------------------------------------------------
def sample():
    df = pd.read_csv("raw_reviews.csv")

    picked = []
    for label in ["positive", "neutral", "negative"]:
        sub = df[df["label"] == label]

        # balance across the 3 apps as evenly as the data allows
        apps = sub["app"].unique()
        per_app = PER_LABEL_BUFFER // len(apps)
        chunks = []
        for app in apps:
            app_rows = sub[sub["app"] == app]
            chunks.append(app_rows.sample(min(per_app, len(app_rows)),
                                          random_state=SEED))
        chosen = pd.concat(chunks)

        # top up if any app was short, so we still reach the buffer
        if len(chosen) < PER_LABEL_BUFFER:
            remaining = sub.drop(chosen.index)
            extra = remaining.sample(
                min(PER_LABEL_BUFFER - len(chosen), len(remaining)),
                random_state=SEED)
            chosen = pd.concat([chosen, extra])

        picked.append(chosen)

    out = pd.concat(picked).reset_index(drop=True)
    out["verified_label"] = out["label"]   # you correct this column by hand
    out["keep"] = 1                        # set to 0 to drop a row

    cols = ["keep", "verified_label", "label", "rating", "app",
            "language", "text", "review_id", "thumbs_up", "date", "source"]
    out = out[cols]
    out.to_csv("to_curate.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote to_curate.csv  ({len(out)} rows)")
    print(out["verified_label"].value_counts().to_string())
    print("\nNow open to_curate.csv in Excel, fix labels, set keep=0 to drop,")
    print("then run this again with MODE = 'finalise'.")


# ----------------------------------------------------------------------
def finalise():
    df = pd.read_csv("to_curate.csv")
    df = df[df["keep"] == 1]

    final = []
    for label in ["positive", "neutral", "negative"]:
        sub = df[df["verified_label"] == label]
        if len(sub) < TARGET_PER_LABEL:
            print(f"!! Only {len(sub)} '{label}' rows after curation "
                  f"(need {TARGET_PER_LABEL}). Verify more rows / re-sample.")
            final.append(sub)
        else:
            final.append(sub.sample(TARGET_PER_LABEL, random_state=SEED))

    out = pd.concat(final).sample(frac=1, random_state=SEED)  # shuffle
    keep_cols = ["text", "verified_label", "app", "language",
                 "rating", "review_id", "source"]
    out = out[keep_cols].rename(columns={"verified_label": "label"})
    out = out.reset_index(drop=True)

    out.to_csv("dataset_360.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote dataset_360.csv  ({len(out)} rows)")
    print(out["label"].value_counts().to_string())


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if MODE == "sample":
        sample()
    elif MODE == "finalise":
        finalise()
    else:
        print("Set MODE to 'sample' or 'finalise'.")