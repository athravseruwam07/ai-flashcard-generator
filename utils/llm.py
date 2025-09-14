# utils/llm.py
# turbo single-call generation with a robust parser that accepts:
# - tsv: "question<TAB>answer"
# - q:/a: on one line or two lines
# - numbered/bulleted lists like "1) Q ... - A ..."
# - json list of {question, answer}
# comments are lowercase + friendly

from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Optional
from openai import OpenAI

# small, strict prompt that tends to work well with mistral
SYSTEM_TURBO = (
    "you create high-quality study flashcards from user notes. "
    "ask clear, testable questions and give short, precise answers only from the notes. "
    "prefer why/how/compare when useful. if not in notes, write 'not found in notes'. "
    "return exactly N flashcards."
)

# we show the model multiple acceptable formats but ask strongly for tsv
USER_TURBO = (
    "notes (compressed excerpts):\n---\n{corpus}\n---\n\n"
    "create exactly {n_total} flashcards about the core ideas. keep answers 1–2 sentences.\n"
    "preferred output: one line per card, tab-separated -> question\tanswer\n"
    "acceptable fallback formats if needed: 'Q: ... A: ...' on one line OR two lines.\n"
    "do not add numbering, bullets, headers, or extra commentary."
)

# --------------------------
# parsing helpers
# --------------------------

_Q_PREFIX = re.compile(r'^\s*(q(uestion)?[:\-]\s*)', re.IGNORECASE)
_A_PREFIX = re.compile(r'^\s*(a(nswer)?[:\-]\s*)', re.IGNORECASE)
_NUM_PREFIX = re.compile(r'^\s*[\-\u2022\*]?\s*(\d+[\)\.\-:]|\-|\u2022|\*)\s*', re.IGNORECASE)

def _clean_piece(s: str) -> str:
    # strip bullets like "1) ", "• ", "- "
    s = _NUM_PREFIX.sub("", s.strip())
    # remove leading Q:/A:
    s = _Q_PREFIX.sub("", s)
    s = _A_PREFIX.sub("", s)
    # collapse spaces
    return re.sub(r'\s+', ' ', s).strip()

def _parse_tsv_lines(text: str) -> List[Dict]:
    out: List[Dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "\t" not in line:
            continue
        q, a = line.split("\t", 1)
        q, a = _clean_piece(q), _clean_piece(a)
        if q and a:
            out.append({"question": q, "answer": a, "source_chunk": 0})
    return out

def _parse_q_a_one_line(text: str) -> List[Dict]:
    # pattern: "Q: ... A: ..." on the same line
    out: List[Dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if " A:" in line and line.lower().startswith("q:"):
            try:
                q = line.split(":", 1)[1].split(" A:", 1)[0]
                a = line.split(" A:", 1)[1]
                q, a = _clean_piece(q), _clean_piece(a)
                if q and a:
                    out.append({"question": q, "answer": a, "source_chunk": 0})
            except Exception:
                continue
        elif " Answer:" in line and line.lower().startswith(("q:", "question:")):
            # support "Q: ... Answer: ..."
            try:
                q = line.split(":", 1)[1].split(" Answer:", 1)[0]
                a = line.split(" Answer:", 1)[1]
                q, a = _clean_piece(q), _clean_piece(a)
                if q and a:
                    out.append({"question": q, "answer": a, "source_chunk": 0})
            except Exception:
                continue
    return out

def _parse_q_a_two_lines(text: str) -> List[Dict]:
    # pattern across two consecutive lines:
    # Q: ...
    # A: ...
    out: List[Dict] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    while i < len(lines) - 1:
        l1, l2 = lines[i], lines[i + 1]
        if _Q_PREFIX.match(l1) and _A_PREFIX.match(l2):
            q = _clean_piece(_Q_PREFIX.sub("", l1))
            a = _clean_piece(_A_PREFIX.sub("", l2))
            if q and a:
                out.append({"question": q, "answer": a, "source_chunk": 0})
            i += 2
        else:
            i += 1
    return out

def _parse_numbered_pairs(text: str) -> List[Dict]:
    # try to match common "1) Question - Answer" or "1. Question: Answer" styles
    out: List[Dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # remove leading numbers/bullets
        line = _NUM_PREFIX.sub("", line)
        # common separators between q and a
        for sep in [" - ", " — ", " : ", " – "]:
            if sep in line:
                q, a = line.split(sep, 1)
                q, a = _clean_piece(q), _clean_piece(a)
                if q.lower().startswith(("q:", "question:")):
                    q = _clean_piece(q)
                if a.lower().startswith(("a:", "answer:")):
                    a = _clean_piece(a)
                if q and a:
                    out.append({"question": q, "answer": a, "source_chunk": 0})
                break
    return out

def _parse_json_list(text: str) -> List[Dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except Exception:
                return []
        else:
            return []
    if not isinstance(data, list):
        return []
    out: List[Dict] = []
    for item in data:
        if isinstance(item, dict):
            q = _clean_piece(str(item.get("question", "")))
            a = _clean_piece(str(item.get("answer", "")))
            if q and a:
                out.append({"question": q, "answer": a, "source_chunk": 0})
    return out

def _robust_parse_any(text: str) -> List[Dict]:
    # try strict → permissive fallbacks
    parsers = [
        _parse_tsv_lines,
        _parse_q_a_one_line,
        _parse_q_a_two_lines,
        _parse_numbered_pairs,
        _parse_json_list,
    ]
    for p in parsers:
        cards = p(text)
        if cards:
            return cards
    return []

# --------------------------
# corpus compression
# --------------------------

def _compress_corpus(text: str, max_chars: int = 12000, slices: int = 8) -> str:
    # evenly sample slices across the doc to keep request small + fast
    if len(text) <= max_chars:
        return text
    slice_len = max_chars // slices
    segs = []
    for i in range(slices):
        start = round(i * (len(text) - slice_len) / max(1, slices - 1))
        segs.append(text[start:start + slice_len])
    return "\n...\n".join(segs)

# --------------------------
# public api
# --------------------------

def generate_flashcards_turbo(
    full_text: str,
    n_total: int,
    api_key: str,
    temperature: float,
    topics: Optional[List[str]] = None,
) -> List[Dict]:
    """single fast call that asks for all cards at once; robustly parses output."""
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("MODEL_NAME", "mistral:7b-instruct")
    client = OpenAI(api_key=api_key, base_url=base_url)

    corpus = _compress_corpus(full_text, max_chars=12000, slices=8)
    if topics:
        corpus += "\nFocus on: " + ", ".join(topics)

    user = USER_TURBO.format(corpus=corpus, n_total=int(n_total))
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": SYSTEM_TURBO},
                  {"role": "user", "content": user}],
        temperature=float(temperature),
        top_p=1.0,
        max_tokens=min(1200, 45 * max(1, n_total)),  # give mistral room but keep it tight for speed
    )
    text = resp.choices[0].message.content or ""

    cards = _robust_parse_any(text)
    if len(cards) >= max(1, n_total // 2):
        return cards[:n_total]

    # second try: very strict reminder to return TSV only
    resp2 = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": SYSTEM_TURBO},
                  {"role": "user", "content": user + "\nRespond TSV only: question\\tanswer per line; no extra text."}],
        temperature=float(temperature),
        top_p=1.0,
        max_tokens=min(1200, 45 * max(1, n_total)),
    )
    text2 = resp2.choices[0].message.content or ""
    cards2 = _robust_parse_any(text2)
    return cards2[:n_total]