"""
TNL6323 - Apply corrected labels

After you edit 'corrected_label' in relabel_worksheet.csv, this merges your
corrections back into the dataset and rebalances to an equal count per class.

RUN:   python apply_relabel.py
IN:    dataset_preprocessed.csv  +  relabel_worksheet.csv
OUT:   dataset_preprocessed.csv  (overwritten, corrected + rebalanced)
       dataset_preprocessed_backup.csv  (safety copy of the original)
"""

import pandas as pd

def main():
    df = pd.read_csv("dataset_preprocessed.csv")
    ws = pd.read_csv("relabel_worksheet.csv")

    # safety backup
    df.to_csv("dataset_preprocessed_backup.csv", index=False, encoding="utf-8-sig")

    # match corrections back to df by exact text
    fix = dict(zip(ws["text"], ws["corrected_label"]))
    df["label"] = [fix.get(t, l) for t, l in zip(df["text"], df["label"])]

    valid = {"positive", "neutral", "negative"}
    bad = ~df["label"].isin(valid)
    if bad.any():
        print(f"WARNING: {bad.sum()} rows have an invalid label - fix these:")
        print(df.loc[bad, "label"].value_counts().to_string())

    print("\nLabel counts AFTER correction:")
    print(df["label"].value_counts().to_string())

    # rebalance to equal count per class (so training stays balanced)
    n = df["label"].value_counts().min()
    balanced = (df.groupby("label", group_keys=False)
                  .apply(lambda g: g.sample(n=n, random_state=42)))
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    balanced.to_csv("dataset_preprocessed.csv", index=False, encoding="utf-8-sig")
    print(f"\nBalanced to {n} per class -> {len(balanced)} rows total.")
    print("Saved -> dataset_preprocessed.csv  (backup in _backup.csv)")
    print("Next: rerun train_models.py")

if __name__ == "__main__":
    main()