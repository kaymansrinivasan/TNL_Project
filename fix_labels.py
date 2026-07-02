"""
TNL6323 - Apply text-based label corrections (done by reading the reviews)

Rewrites star-derived labels to match what each review actually SAYS:
  complaint/problem -> negative | praise -> positive | question/mixed -> neutral

SAFE against the pandas 3.x groupby.apply bug that dropped the label column.

RUN:  python fix_labels.py
IN:   dataset_preprocessed.csv   (MUST contain a 'label' column)
OUT:  dataset_preprocessed.csv        (corrected + rebalanced)
      dataset_preprocessed_prefix.csv (backup taken before this step)
"""

import sys
import pandas as pd
import re

REBALANCE = True
SEED = 42
MIN_SAFE = 100      # refuse to rebalance if any class falls below this (safety)

CORRECTIONS = {
    "nfc reload doesn't work with old phone": "negative",
    "the latest update has problem to book appointments": "negative",
    "just set allow for mysejahtera on any ad blocker": "negative",
    "i cant open my app. its stuck at secure with pin": "negative",
    "nak buat temujanji klinik gigi semuanya penuh": "negative",
    "so , nak download di handphone baru , tak dapat pula": "neutral",
    "knp acc lama saya tak blh nk buka sy tukar fon": "negative",
    "since last two so difficult to use tng": "negative",
    "cant refresh my profile, the date remain": "negative",
    "kenapa sistem madani sekarang menyusahkan rakyat": "negative",
    "kalau no fon yg dftar da tak aktif": "negative",
    "totally cannot use the function inside app": "negative",
    "tng tusah betul nak pindah kan dari hp": "negative",
    "email tak boleh ubah kenapa ya": "negative",
    "saya nak tukar no fon baru ..tapi sulit": "negative",
    "terok gila tak boleh buat tempahan janji temu": "negative",
    "lepas update version baru, apps tak boleh buka.. blank": "negative",
    "less 1 star because grabmap error": "negative",
    "can't see tng app": "negative",
    "not straight forward enough": "negative",
    "after i got the app back i don't know where to upload": "neutral",
    "boleh tak tambahkan maklumat suntikan typhoid": "neutral",
    "sangat kecewa susah dapat driver": "negative",
    "why quick payment and auto reload is unavailable": "negative",
    "cuba dulu kalau bagus baru bagi 5 star": "neutral",
    "susah gila nk dpt driver dhla student": "negative",
    "very slow login and need to wait more time to scan": "negative",
    "over all the service was fine until it came into a refund": "negative",
    "acc sy tak boleh log in kenape yer": "negative",
    "terbaik senang bayaran bil elektrik": "positive",
    "good customer service from the company, but my main issue": "negative",
    "makin update new version makin teruk": "negative",
    "mohon bantuan, kenapa saya suruh hantar": "negative",
    "kenapa tng jdi lmbt slow msuk adoi": "negative",
    "the aba payment thing sucks": "negative",
    "after my last complain about there is no available slot": "neutral",
    "pin lokasi tidak tepat jln dtk": "negative",
    "tarikh untuk temujanji tidak dapat diplilih": "negative",
    "idk if its just me but the latest update is kinda buggy": "negative",
    "after the previous update, the app took forever to load": "negative",
    "susah sangat nk tgk rekod kesihatan": "negative",
    "kejap bole bukak pastu tetiba je apk xleh buka": "negative",
    "sangat bagus dan membantu": "positive",
    "janan susahkan guna biometric atau passcode": "negative",
    "dah tekan lupa kata laluan . tapi sampai sekarang dari bulan 8": "negative",
    "tolong di review lagi app nya": "negative",
    "i hate this because i wait so long": "negative",
    "look, can you not make a nearest drop point": "negative",
    "senang membuat pembayaran": "positive",
    "unable to update latest apps version for model samsung z fold": "negative",
    "hrini saya ada appoiment dkt kk": "negative",
    "tak boleh nak book appointment. tak keluar pun nearest": "negative",
    "baru update apps ni, tpi makin pening": "negative",
    "it's great for booking an appointment but the qr code scanner": "neutral",
    "with this new updated app, i can't find my covid 19 vaccination": "neutral",
    "after so long not open this app since pkp": "neutral",
    "kenapa makin lama makin slow server dia": "negative",
    "tak boleh nk book oppiment": "negative",
    "i hope they would add a feature to cancel your order": "neutral",
    "grab app is good but i have issue with sale debit": "neutral",
    "apa ke benda ni ???@ nak buat temujanji": "negative",
    # --- second pass: residual positive-labelled complaints / requests ---
    "x boleh bayar bil elektrik dan banyak pulak iklan": "negative",
    "kenapa setiap kali saya nak login semula": "negative",
    "kenapa selepas kemas kini sijil jadi hitam putih": "negative",
    "my sejahtera saya bhgian profil xdpt buka cma kosong": "negative",
    "saya penah guna ewallet tapi dah lama x buka bila nak buka balik nombor telifon": "negative",
    "latest version tiada detail di blood donor card": "negative",
    "recently after updated, the fixed notification for mysj trace always appeared": "negative",
    "please make dark mode version": "neutral",
    "it would be nice for the app to have dark mode": "neutral",
    "please bring back the option to pay and split evenly": "neutral",
    "can be improve.": "neutral",
}


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def main():
    df = pd.read_csv("dataset_preprocessed.csv")

    # --- GUARD 1: the label column must exist ---
    if "label" not in df.columns:
        sys.exit(
            "ERROR: dataset_preprocessed.csv has no 'label' column "
            f"(columns: {list(df.columns)}).\n"
            "The file is corrupted. Recover by running:  python preprocess.py\n"
            "then run this script again."
        )

    df.to_csv("dataset_preprocessed_prefix.csv", index=False, encoding="utf-8-sig")

    norm_text = df["text"].map(norm)
    changed = miss = multi = 0
    for snippet, new_label in CORRECTIONS.items():
        mask = norm_text.str.contains(re.escape(norm(snippet)), regex=True)
        n = int(mask.sum())
        if n == 0:
            print(f"  ! no match: {snippet[:50]}"); miss += 1
        elif n > 1:
            print(f"  ! {n} matches (skipped): {snippet[:50]}"); multi += 1
        else:
            df.loc[mask, "label"] = new_label; changed += 1

    print(f"\nApplied {changed} corrections ({miss} not found, {multi} ambiguous).")
    counts = df["label"].value_counts()
    print("\nLabel counts AFTER correction:")
    print(counts.to_string())

    # --- GUARD 2: safe rebalance (concat, not groupby.apply) ---
    if REBALANCE:
        n = int(counts.min())
        if n < MIN_SAFE:
            print(f"\n!! Smallest class has only {n} rows (< {MIN_SAFE}). "
                  "Skipping rebalance to avoid destroying the dataset. "
                  "Keeping all corrected rows (imbalanced).")
        else:
            parts = [g.sample(n=n, random_state=SEED)
                     for _, g in df.groupby("label")]
            df = (pd.concat(parts)
                    .sample(frac=1, random_state=SEED)
                    .reset_index(drop=True))
            print(f"\nRebalanced to {n} per class -> {len(df)} rows.")

    # --- GUARD 3: never save a broken file ---
    assert "label" in df.columns, "label column lost - aborting save"
    assert len(df) >= 300, f"only {len(df)} rows - aborting save"

    df.to_csv("dataset_preprocessed.csv", index=False, encoding="utf-8-sig")
    print("Saved -> dataset_preprocessed.csv (backup: _prefix.csv)")
    print("Next: rerun train_models.py")


if __name__ == "__main__":
    main()