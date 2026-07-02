"""
TNL6323 - Malaysia-Focused Sentiment Analysis
Advanced Feature: Emoji Sentiment Integration  (shared module)

Turns the `emojis` column (produced by preprocess.py) into numeric features
that get concatenated onto the TF-IDF matrix, so the model actually LEARNS
from emojis instead of ignoring them.

IMPORTANT: import this in BOTH train_models.py AND the deployment app, so
emojis are scored the SAME way at training time and at prediction time.

Features per row (all NON-NEGATIVE so MultinomialNB still works):
    [ n_emoji , n_pos , n_neg , n_neu ]

Emoji polarity = curated list first (high precision on common ones),
then a fallback that reads the emoji's CLDR name (e.g. ":pouting_face:")
so we still classify emojis not in the list.
"""

import numpy as np
import emoji as emoji_lib

FEATURE_NAMES = ["emoji_total", "emoji_pos", "emoji_neg", "emoji_neu"]

# --- curated overrides (exact emoji -> polarity) ---
_POS = {"😍","😊","😁","😄","😃","🥰","❤️","❤","💕","💖","👍","🙏","🔥","💯",
        "👏","🥳","😂","🤩","😘","😻","✅","⭐","🌟","💪","🙌","😎"}
_NEG = {"😡","😠","🤬","😤","👎","💔","😢","😭","😩","😞","😔","🙄","😒","🤮",
        "🤢","💩","😨","😰","😥","⚠️","❌","😣","😖","😫","😑"}
_NEU = {"😐","🤔","😶","🤷","😕","🫤","😬"}

# --- name-based fallback (substring match on demojized name) ---
_NEG_NAME = ("angry","rage","pout","cry","sob","disappoint","thumbs_down",
             "broken_heart","weary","tired","fearful","confounded","unamused",
             "vomit","nauseat","pile_of_poo","frown","worried","anxious","anguish")
_POS_NAME = ("smil","grin","joy","heart","love","kiss","thumbs_up","clap",
             "star_struck","hug","blush","hundred","ok_hand","raising_hands",
             "party","fire","sparkl")
_NEU_NAME = ("neutral_face","expressionless","thinking","shrug","confused")


def emoji_polarity(e: str) -> int:
    """+1 positive, -1 negative, 0 neutral/unknown."""
    if e in _POS:
        return 1
    if e in _NEG:
        return -1
    if e in _NEU:
        return 0
    name = emoji_lib.demojize(e)
    if any(k in name for k in _NEG_NAME):   # check neg first (broken_heart)
        return -1
    if any(k in name for k in _POS_NAME):
        return 1
    return 0


def emoji_features(emoji_str) -> list:
    """One row -> [n_emoji, n_pos, n_neg, n_neu]."""
    s = "" if emoji_str is None else str(emoji_str)
    found = [d["emoji"] for d in emoji_lib.emoji_list(s)]
    n_pos = sum(1 for e in found if emoji_polarity(e) == 1)
    n_neg = sum(1 for e in found if emoji_polarity(e) == -1)
    n_neu = len(found) - n_pos - n_neg
    return [len(found), n_pos, n_neg, n_neu]


def build_emoji_matrix(emoji_series) -> np.ndarray:
    """A pandas Series of emoji strings -> (n_rows, 4) float matrix."""
    return np.array([emoji_features(x) for x in emoji_series], dtype=float)


if __name__ == "__main__":
    for s in ["😍👍", "😡😭", "😐", "", "🔥🔥best"]:
        print(f"{s!r:12} -> {dict(zip(FEATURE_NAMES, emoji_features(s)))}")