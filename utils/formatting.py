# utils/formatting.py
# dataframe helpers and simple validation

from __future__ import annotations

import pandas as pd
from typing import List, Dict, Tuple


def cards_to_dataframe(cards: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(cards)
    if df.empty:
        return pd.DataFrame(columns=["question", "answer", "source_chunk"])
    # enforce column order
    df = df[["question", "answer", "source_chunk"]]
    return df


def validate_cards(df: pd.DataFrame) -> Tuple[bool, str]:
    if df.empty:
        return False, "no cards to show yet"
    # basic length checks to avoid super-long sides
    long_q = df["question"].fillna("").str.len().max()
    long_a = df["answer"].fillna("").str.len().max()
    if long_q > 600:
        return False, "some questions are too long (>600 chars). try editing or regenerating."
    if long_a > 1500:
        return False, "some answers are very long (>1500 chars). consider trimming."
    if (df[["question", "answer"]].fillna("") == "").any().any():
        return False, "found empty question/answer cells. fill or delete them before exporting."
    return True, "ok"