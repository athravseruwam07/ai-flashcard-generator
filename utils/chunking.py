# utils/chunking.py
# token estimation and chunking logic

from __future__ import annotations

import re
from typing import List

import tiktoken


def _get_encoder():
    # cl100k_base matches gpt-4/3.5 families reasonably well
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return tiktoken.get_encoding("p50k_base")


def estimate_tokens(text: str) -> int:
    enc = _get_encoder()
    return len(enc.encode(text or ""))


def _split_candidates(text: str) -> List[str]:
    # split by markdown-ish headings or all-caps lines first
    parts = re.split(r"(?m)^(?:#{1,6} .*|[A-Z0-9][A-Z0-9 \-]{6,})\n", text)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]
    # then by paragraphs
    paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if len(paras) > 1:
        return paras
    # fallback: sentences via simple regex
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def chunk_text(text: str, target_tokens: int = 1400, overlap_tokens: int = 150) -> List[str]:
    """greedily pack chunks up to ~target tokens, with overlap between chunks.
    we prefer to add whole paragraphs/sections when possible.
    """
    enc = _get_encoder()
    units = _split_candidates(text)

    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0

    for unit in units:
        utoks = len(enc.encode(unit))
        if cur_tok + utoks <= target_tokens:
            cur.append(unit)
            cur_tok += utoks
        else:
            if cur:
                chunks.append("\n\n".join(cur))
            # start new chunk, but try to add overlap from end of previous chunk
            if chunks:
                last = chunks[-1]
                last_tokens = enc.encode(last)
                # take the last overlap_tokens of previous chunk as overlap
                overlap_slice = enc.decode(last_tokens[max(0, len(last_tokens)-overlap_tokens):])
                cur = [overlap_slice, unit]
                cur_tok = len(enc.encode(overlap_slice)) + utoks
            else:
                cur = [unit]
                cur_tok = utoks

            # if a single unit is gigantic, we hard-split it by tokens
            while cur_tok > target_tokens * 1.3:
                toks = enc.encode("\n\n".join(cur))
                part = enc.decode(toks[:target_tokens])
                rest = enc.decode(toks[target_tokens - overlap_tokens:])
                chunks.append(part)
                cur = [rest]
                cur_tok = len(enc.encode(rest))

    if cur:
        chunks.append("\n\n".join(cur))

    return chunks