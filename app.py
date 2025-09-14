# app.py
# modern flashcard generator with in-app study mode
# fixed: needs-review doesn't inflate count repeatedly; proper state transitions

from __future__ import annotations

import os
import random
import pandas as pd
import streamlit as st

from utils.parsing import read_file, clean_text
from utils.llm import generate_flashcards_turbo
from utils.formatting import cards_to_dataframe, validate_cards
from utils.export import to_csv_bytes, to_anki_txt_bytes

APP_BRAND = "FlashcardGPT"
APP_TAGLINE = "AI-powered tool that transforms notes into test-ready flashcards in seconds."
NAV_LINK = "athrav-portfolio.vercel.app"  # your portfolio link

# --- page config + minimal chrome ---
st.set_page_config(page_title=APP_BRAND, page_icon="🃏", layout="wide")
st.markdown(
    """
    <style>
      #MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
      .block-container { max-width: 1180px; }
      body {
        background:
          radial-gradient(1000px 400px at 15% -8%, rgba(124,58,237,.25), rgba(0,0,0,0)),
          radial-gradient(1200px 500px at 85% -10%, rgba(236,72,153,.18), rgba(0,0,0,0)),
          #0B0F19 !important;
      }
      .hero {
        margin-top:18px; border-radius:22px; padding:30px;
        background:linear-gradient(180deg, rgba(18,24,38,.65) 0%, rgba(12,16,26,.55) 100%);
        border:1px solid rgba(255,255,255,.06);
      }
      .hero h1 {
        font-size:42px; font-weight:800; margin:0;
        background:linear-gradient(135deg,#A78BFA 0%, #60A5FA 35%, #EC4899 80%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      }
      .pill {
        display:inline-block; padding:4px 11px; border-radius:999px; margin-right:8px;
        background:#111827; border:1px solid #263142; font-size:12px; color:#E5E7EB;
      }
      .card {
        border-radius:16px; padding:28px; border:1px solid rgba(255,255,255,.08);
        background:linear-gradient(180deg, rgba(18,24,38,.6) 0%, rgba(12,16,26,.6) 100%);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
        min-height: 160px; display:flex; align-items:center; justify-content:center;
        text-align:center; font-size:22px; color:#E5E7EB;
      }
      .card .label {
        display:inline-block; font-size:12px; color:#aab2c5; background:#111827; border:1px solid #283043;
        padding:2px 8px; border-radius:999px; margin-bottom:8px;
      }
      .subtle { color:#9CA3AF; font-size:13px; }
      .chip { display:inline-block; font-size:11px; padding:2px 8px; border-radius:999px; border:1px solid #334155; background:#111827; color:#cbd5e1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- session state init ---
if "raw_text" not in st.session_state:
    st.session_state["raw_text"] = ""
if "cards_df" not in st.session_state:
    st.session_state["cards_df"] = pd.DataFrame(columns=["question", "answer", "source_chunk"])
# study state
if "study_order" not in st.session_state:
    st.session_state["study_order"] = []
if "study_idx" not in st.session_state:
    st.session_state["study_idx"] = 0
if "study_show_answer" not in st.session_state:
    st.session_state["study_show_answer"] = False
if "study_correct" not in st.session_state:
    st.session_state["study_correct"] = 0
if "study_review" not in st.session_state:
    st.session_state["study_review"] = 0
if "study_last_card" not in st.session_state:
    st.session_state["study_last_card"] = None
# per-card status: "new" | "review" | "correct"
if "study_status" not in st.session_state:
    st.session_state["study_status"] = {}  # {card_idx: status}

def reset_study(shuffle: bool = True):
    df = st.session_state["cards_df"]
    n = len(df)
    st.session_state["study_order"] = list(range(n))
    if shuffle:
        random.shuffle(st.session_state["study_order"])
    st.session_state["study_idx"] = 0
    st.session_state["study_show_answer"] = False
    st.session_state["study_correct"] = 0
    st.session_state["study_review"] = 0
    st.session_state["study_last_card"] = None
    # reset statuses
    st.session_state["study_status"] = {i: "new" for i in range(n)}

# --- hero ---
st.markdown(
    f"""
    <div class="hero">
      <h1>{APP_BRAND}</h1>
      <p style="color:#9CA3AF;">{APP_TAGLINE}</p>
      <div><span class="pill">PDF</span><span class="pill">DOCX</span><span class="pill">PPTX</span><span class="pill">TXT</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- input ---
st.markdown("### 1) add your notes")
tabs = st.tabs(["📄 upload", "✍️ paste"])

with tabs[0]:
    up = st.file_uploader("drop a file", type=["pdf", "docx", "pptx", "txt"], label_visibility="collapsed")
    if up is not None:
        try:
            text = read_file(up)
            text = clean_text(text)
            st.session_state["raw_text"] = text
            st.success(f"parsed **{up.name}**")
        except Exception as e:
            st.error(f"could not parse file: {e}")

with tabs[1]:
    pasted = st.text_area(
        "paste text",
        value=st.session_state["raw_text"],
        height=220,
        placeholder="paste your notes here…",
        label_visibility="collapsed",
    )
    if pasted != st.session_state["raw_text"]:
        st.session_state["raw_text"] = pasted

# --- settings ---
st.markdown("### 2) settings")
target_cards = st.slider("target flashcards", 6, 120, 30, 2)

# --- generate ---
st.markdown("### 3) generate")
if st.button("generate flashcards", type="primary", use_container_width=True):
    raw_text = st.session_state["raw_text"]
    if not raw_text.strip():
        st.error("please add some notes first.")
    else:
        with st.spinner("generating flashcards…"):
            try:
                cards = generate_flashcards_turbo(
                    full_text=raw_text,
                    n_total=int(target_cards),
                    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
                    temperature=0.2,
                    topics=None,
                )
                df = cards_to_dataframe(cards)
                st.session_state["cards_df"] = df
                if len(df) == 0:
                    st.warning("the model returned no usable flashcards. try a bit more text or a different model.")
                else:
                    reset_study(shuffle=True)
                    st.success(f"done — generated {len(df)} cards")
            except Exception as e:
                st.error(f"generation failed: {e}")

# --- study mode ---
st.markdown("### 4) study")

df = st.session_state["cards_df"]
if df.empty:
    st.info("no cards yet. generate cards to start studying.")
else:
    # controls
    c1, c2, c3, c4, c5 = st.columns([0.18, 0.18, 0.2, 0.22, 0.22])
    with c1:
        if st.button("restart 🔄", use_container_width=True, key="btn_restart"):
            reset_study(shuffle=False)
            st.rerun()
    with c2:
        if st.button("shuffle 🔀", use_container_width=True, key="btn_shuffle"):
            reset_study(shuffle=True)
            st.rerun()
    with c3:
        st.write("")
        st.caption(f"progress: **{st.session_state['study_idx']}/{len(st.session_state['study_order'])}**")
    with c4:
        st.write("")
        st.caption(f"correct: **{st.session_state['study_correct']}**")
    with c5:
        st.write("")
        st.caption(f"needs review: **{st.session_state['study_review']}**")

    # current card
    order = st.session_state["study_order"]
    idx_in_deck = st.session_state["study_idx"]
    if not order:
        reset_study(shuffle=True)
        order = st.session_state["study_order"]

    if idx_in_deck >= len(order):
        st.success("deck complete! restart or shuffle to practice again.")
    else:
        card_idx = order[idx_in_deck]

        # reset flip state when card changes
        if st.session_state["study_last_card"] != card_idx:
            st.session_state["study_show_answer"] = False
            st.session_state["study_last_card"] = card_idx

        # ensure status exists (handles regenerate after initial reset)
        if card_idx not in st.session_state["study_status"]:
            st.session_state["study_status"][card_idx] = "new"

        row = df.iloc[card_idx]
        q, a = row["question"], row["answer"]

        # card ui
        st.markdown("<div class='card'><div><div class='label'>question</div>" + q + "</div></div>", unsafe_allow_html=True)
        status = st.session_state["study_status"].get(card_idx, "new")
        if status == "review":
            st.markdown("<span class='chip'>marked for review</span>", unsafe_allow_html=True)
        elif status == "correct":
            st.markdown("<span class='chip'>mastered</span>", unsafe_allow_html=True)

        st.write("")
        if st.session_state["study_show_answer"]:
            st.markdown("<div class='card'><div><div class='label'>answer</div>" + a + "</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='subtle'>click <strong>show answer</strong> to reveal</div>", unsafe_allow_html=True)

        st.write("")
        # you can only grade after flipping
        can_grade = bool(st.session_state["study_show_answer"])

        b1, b2, b3, b4 = st.columns([0.2, 0.2, 0.3, 0.3])
        with b1:
            if st.button("◀ prev", use_container_width=True, disabled=idx_in_deck <= 0, key=f"prev_{card_idx}"):
                st.session_state["study_idx"] = max(0, idx_in_deck - 1)
                st.rerun()
        with b2:
            if st.button("show answer", use_container_width=True, key=f"show_{card_idx}"):
                st.session_state["study_show_answer"] = True
                st.rerun()
        with b3:
            if st.button("✓ correct", use_container_width=True, disabled=not can_grade, key=f"correct_{card_idx}"):
                if can_grade:
                    prev_status = st.session_state["study_status"].get(card_idx, "new")
                    # transition accounting
                    if prev_status != "correct":
                        st.session_state["study_correct"] += 1
                        if prev_status == "review":
                            st.session_state["study_review"] = max(0, st.session_state["study_review"] - 1)
                        st.session_state["study_status"][card_idx] = "correct"
                    # advance to next card
                    st.session_state["study_idx"] = min(len(order), idx_in_deck + 1)
                    st.rerun()
        with b4:
            if st.button("↺ needs review", use_container_width=True, disabled=not can_grade, key=f"review_{card_idx}"):
                if can_grade:
                    prev_status = st.session_state["study_status"].get(card_idx, "new")
                    # only count the first time we move into review
                    if prev_status != "review":
                        st.session_state["study_review"] += 1
                        # if it was previously correct and now marked review again, adjust correct down
                        if prev_status == "correct":
                            st.session_state["study_correct"] = max(0, st.session_state["study_correct"] - 1)
                    st.session_state["study_status"][card_idx] = "review"
                    # move this card to end so it resurfaces later
                    order.append(order.pop(idx_in_deck))
                    st.session_state["study_order"] = order
                    # keep index pointing at the next new card at this position
                    st.rerun()

# --- export ---
st.markdown("### 5) export")
if not df.empty:
    out = df[["question", "answer", "source_chunk"]].copy()

    csv_bytes = to_csv_bytes(out.rename(columns={"question": "front", "answer": "back"}))
    st.download_button("download csv", csv_bytes, "flashcards.csv", "text/csv", use_container_width=True)

    anki_bytes = to_anki_txt_bytes(out)
    st.download_button("download anki txt", anki_bytes, "flashcards_anki.txt", "text/plain", use_container_width=True)
else:
    st.info("no cards to export yet.")
