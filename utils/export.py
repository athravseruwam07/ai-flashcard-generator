# utils/export.py
# export helpers for csv and anki text

from __future__ import annotations

import io
import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    out = df[["front", "back"]] if set(["front", "back"]).issubset(df.columns) else df[["question", "answer"]].rename(columns={"question": "front", "answer": "back"})
    out.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def to_anki_txt_bytes(df: pd.DataFrame) -> bytes:
    out = df[["question", "answer"]]
    lines = [f"{q}\t{a}" for q, a in out.itertuples(index=False)]
    txt = "\n".join(lines)
    return txt.encode("utf-8")