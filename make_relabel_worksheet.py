"""
TNL6323 - Relabeling helper (star-vs-text mismatch fixer)

The labels currently follow the STAR RATING, but sentiment analysis must
follow the TEXT. Many rows disagree (5-star review that says "took forever
to load"). This script SCORES each row for how likely its label is wrong,
using positive/negative cue words in English + Malay, and sorts the most
suspicious rows to the top so you fix the worst offenders first.

It does NOT auto-relabel. You decide. It just tells you where to look.

RUN:   python make_relabel_worksheet.py
IN:    dataset_preprocessed.csv
OUT:   relabel_worksheet.csv   (open in Excel, edit 'corrected_label')

WORKFLOW:
  1. Open relabel_worksheet.csv. Rows are sorted MOST-SUSPICIOUS first.
  2. Read 'text'. Fix 'corrected_label' where it's wrong (positive/neutral/negative).
     Rule: judge the WORDS, not the star. Complaint/problem -> negative.
     Praise/satisfaction -> positive. Factual / mixed / question -> neutral.
  3. Once you reach a long run of rows that are already correct, you can stop.
  4. Run apply_relabel.py (printed at the end) to write the corrected dataset.
"""

import re
import pandas as pd

# --- bilingual cue lexicons (lowercase, stemmed-ish roots) ---
NEG_CUES = {
    # english
    "not","no","cant","cannot","can't","never","fail","failed","error","bug",
    "crash","crashed","slow","lag","laggy","worst","bad","terrible","horrible",
    "useless","hate","annoying","annoy","problem","issue","stuck","unable",
    "wait","waiting","forever","ages","expensive","overpriced","poor","broken",
    "difficult","hard","fix","refund","scam","stupid","mess","sucks","suck",
    # malay / manglish
    "tak","tidak","xboleh","xleh","xde","xda","teruk","susah","payah","rosak",
    "hilang","gagal","lambat","mahal","bodoh","sampah","menyusahkan","susahkan",
    "lupa","tunggu","jangan","bukan","masalah","kecewa","sedih","geram","maki",
    "gila","paksa","potong","rompak","curi",
}
POS_CUES = {
    # english
    "good","great","easy","love","best","nice","awesome","excellent","perfect",
    "helpful","convenient","smooth","fast","reliable","recommend","wonderful",
    "amazing","useful","satisfied","happy","thanks","thank",
    # malay
    "terbaik","baik","bagus","senang","mudah","suka","cepat","berbaloi","mantap",
    "puas","gembira","membantu","bantu","cemerlang","selesa","jimat",
}

def tokens(text):
    return set(re.findall(r"[a-z']+", str(text).lower()))

def cue_counts(text):
    t = tokens(text)
    return len(t & POS_CUES), len(t & NEG_CUES)

def suspicion(label, text):
    """Higher = more likely the label is wrong."""
    pos, neg = cue_counts(text)
    if label == "positive":
        # positive label but the text reads negative
        return neg - pos
    if label == "negative":
        # negative label but the text reads positive
        return pos - neg
    # neutral label but text has strong one-sided polarity
    return abs(pos - neg) - 1

def main():
    df = pd.read_csv("dataset_preprocessed.csv")

    df["susp_score"] = [suspicion(l, t) for l, t in zip(df["label"], df["text"])]
    df["corrected_label"] = df["label"]     # you edit this column

    cols = ["susp_score", "label", "corrected_label", "text", "app", "language"]
    out = df[cols].sort_values("susp_score", ascending=False).reset_index(drop=True)
    out = out.rename(columns={"label": "current_label"})

    out.to_csv("relabel_worksheet.csv", index=False, encoding="utf-8-sig")

    flagged = (out["susp_score"] > 0).sum()
    print(f"Wrote relabel_worksheet.csv ({len(out)} rows)")
    print(f"Rows flagged as likely-wrong (susp_score > 0): {flagged}")
    print("Open it, fix 'corrected_label' top-down, stop when rows look clean.")
    print("\n--- Top 8 most-suspicious rows preview ---")
    for _, r in out.head(8).iterrows():
        print(f"[{r['current_label']:8}] (score {int(r['susp_score'])}) {r['text'][:70]}")

if __name__ == "__main__":
    main()