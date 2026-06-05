"""
app.py  —  ENT Research Assistant

Modes:
  1. Ask the Literature  — Q&A + ranked semantic passages (merged)
  2. Paper Explorer      — Browse corpus, full text, references, LLM summary
"""
from __future__ import annotations

import re

import streamlit as st

from ui_helpers import (
    get_corpus_stats,
    stream_answer,
    search_papers_by_title,
    get_all_papers,
    generate_deep_summary,
    format_score,
    highlight_terms,
    get_retriever,
    filter_by_relevance,
)

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ENT Research Assistant",
    page_icon="👂",  # simpler, widely-supported ENT-related emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:            #0a0f1c;
    --bg-card:       rgba(15, 23, 42, 0.80);
    --bg-card-hover: rgba(22, 32, 58, 0.90);
    --accent:        #3b82f6;
    --accent-dim:    rgba(59,130,246,0.12);
    --accent-cyan:   #06b6d4;
    --accent-green:  #10b981;
    --accent-amber:  #f59e0b;
    --accent-rose:   #f43f5e;
    --txt:           #e2e8f0;
    --txt-2:         #94a3b8;
    --txt-3:         #64748b;
    --border:        rgba(255,255,255,0.07);
    --radius:        12px;
    --tr:            0.2s cubic-bezier(.4,0,.2,1);
}

/* ── Universal font ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Full-page dark background (covers every Streamlit wrapper layer) ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main,
.main > .block-container,
[data-testid="block-container"],
[data-testid="stMainBlockContainer"] {
    background-color: var(--bg) !important;
    color: var(--txt) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0d1424 0%, #0a0f1c 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Markdown / text colour in main area ── */
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p {
    color: var(--txt) !important;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: rgba(15,23,42,0.75) !important;
    border: 1px solid rgba(59,130,246,0.22) !important;
    border-radius: 10px !important;
    color: var(--txt) !important;
    caret-color: var(--accent);
}
.stTextInput > div > div > input::placeholder { color: var(--txt-3) !important; }
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.18) !important;
    outline: none;
}

/* ── Buttons ── */
div.stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(59,130,246,0.22) !important;
    background: rgba(15,23,42,0.55) !important;
    color: var(--txt) !important;
    font-weight: 500 !important;
    transition: all var(--tr) !important;
}
div.stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(59,130,246,0.1) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2563eb, #0891b2) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8, #0e7490) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.3) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div,
[data-testid="stSelectbox"] > div > div {
    background: rgba(15,23,42,0.7) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 10px !important;
    color: var(--txt) !important;
}

/* ── Expander ── */
details[data-testid="stExpander"],
[data-testid="stExpander"] {
    background: rgba(15,23,42,0.5) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: var(--txt) !important;
    font-weight: 500;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(15,23,42,0.5);
    border-radius: 11px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500;
    color: var(--txt-2) !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.18) !important;
    color: #93c5fd !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1rem;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--txt-2) !important; }

/* ── Info / warning boxes ── */
[data-testid="stAlert"],
.stAlert {
    background: rgba(59,130,246,0.08) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 10px !important;
    color: var(--txt) !important;
}

/* ── Divider ── */
hr { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }

/* ── Highlight ── */
mark { background: rgba(59,130,246,0.18); color: #93c5fd !important;
       border-radius: 3px; padding: 0 2px; }
strong { color: #93c5fd !important; }

/* ── Glass card ── */
.glass {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.85rem;
    transition: border-color var(--tr), box-shadow var(--tr), transform var(--tr);
}
.glass:hover {
    border-color: rgba(59,130,246,0.22);
    box-shadow: 0 6px 24px rgba(59,130,246,0.09);
    transform: translateY(-1px);
}

/* ── Hero ── */
.hero { padding: 1.5rem 0 0.75rem; }
.hero h1 {
    font-size: 1.75rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 55%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 0.35rem;
}
.hero p {
    font-size: 0.87rem;
    color: var(--txt-2);
    line-height: 1.65;
    max-width: 600px;
}

/* ── Stat pill ── */
.stat-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 13px;
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.13);
    border-radius: 10px;
    margin-bottom: 7px;
}
.stat-n { font-size: 1.15rem; font-weight: 700; color: #60a5fa; min-width: 38px; }
.stat-l { font-size: 0.69rem; color: var(--txt-3); text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Badges ── */
.badge {
    display: inline-flex; align-items: center;
    padding: 2px 9px;
    border-radius: 99px;
    font-size: 0.7rem; font-weight: 600; white-space: nowrap;
}
.b-high { background: rgba(16,185,129,0.15); color: #34d399; }
.b-mid  { background: rgba(245,158,11,0.15);  color: #fbbf24; }
.b-low  { background: rgba(244,63,94,0.15);   color: #fb7185; }
.b-sec  {
    background: rgba(6,182,212,0.12); color: #22d3ee;
    border-radius: 5px; text-transform: uppercase;
    letter-spacing: 0.07em; font-size: 0.66rem; padding: 2px 6px;
}
.b-pmc  { background: rgba(59,130,246,0.12); color: #60a5fa; }

/* ── Score bar ── */
.strack  { background: rgba(255,255,255,0.06); border-radius: 99px; height: 4px; overflow: hidden; margin-top: 6px; }
.sfill   { height: 100%; border-radius: 99px; }

/* ── Passage card ── */
.pcard {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.95rem 1.2rem;
    margin-bottom: 0.7rem;
    transition: border-color var(--tr), box-shadow var(--tr);
}
.pcard:hover { border-color: rgba(59,130,246,0.22); box-shadow: 0 4px 18px rgba(59,130,246,0.08); }
.pcard-meta { font-size: 0.72rem; color: var(--txt-3); margin-top: 7px; display: flex; flex-wrap: wrap; gap: 10px; }

/* ── Paper grid card ── */
.pgcard {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    height: 100%;
    transition: border-color var(--tr), transform var(--tr);
}
.pgcard:hover { border-color: rgba(59,130,246,0.24); transform: translateY(-2px); box-shadow: 0 8px 28px rgba(59,130,246,0.1); }
.pgcard h4 {
    font-size: 0.85rem; font-weight: 600; color: var(--txt); line-height: 1.4; margin-bottom: 6px;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.pgcard-m { font-size: 0.72rem; color: var(--txt-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }

/* ── Paper detail header ── */
.detail-hdr {
    background: linear-gradient(135deg, rgba(37,99,235,0.1) 0%, rgba(6,182,212,0.06) 100%);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Answer box ── */
.answer-box {
    background: rgba(15,23,42,0.65);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    padding: 1.15rem 1.4rem;
    font-size: 0.89rem;
    line-height: 1.8;
    color: var(--txt);
}

/* ── Citation strip ── */
.cite-strip {
    display: flex; flex-wrap: wrap; gap: 6px;
    padding: 0.55rem 0.9rem;
    background: rgba(15,23,42,0.5);
    border: 1px solid var(--border);
    border-radius: 9px;
    margin-top: 0.65rem;
}
.cite-chip {
    font-size: 0.7rem; color: #60a5fa;
    background: rgba(59,130,246,0.12);
    padding: 2px 9px; border-radius: 99px;
}

/* ── Section / reference text ── */
.sec-body {
    font-size: 0.86rem; line-height: 1.75;
    color: var(--txt-2); white-space: pre-wrap;
}
.ref-row {
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem; line-height: 1.55; color: var(--txt-2);
}
.ref-row:last-child { border-bottom: none; }
.ref-t   { font-weight: 600; color: var(--txt); }
.ref-doi { color: var(--accent); text-decoration: none; font-size: 0.75rem; }
.ref-doi:hover { text-decoration: underline; }

/* ── Fade-up animation ── */
@keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
.fi { animation: fadeUp 0.3s ease-out both; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Use the same icons as the page content so navigation and hero are visually aligned
MODE_ASK      = "🔬 Ask the Literature"
MODE_EXPLORER = "📄 Paper Explorer"


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _badge(pct: int) -> str:
    cls = "b-high" if pct >= 75 else ("b-mid" if pct >= 50 else "b-low")
    return f'<span class="badge {cls}">{pct}%</span>'


def _bar(pct: int) -> str:
    c = "#10b981" if pct >= 75 else ("#f59e0b" if pct >= 50 else "#f43f5e")
    return f'<div class="strack"><div class="sfill" style="width:{pct}%;background:{c};"></div></div>'


def _doi(doi: str) -> str:
    if not doi:
        return ""
    return f'<a class="ref-doi" href="https://doi.org/{doi}" target="_blank">DOI ↗</a>'


def _strip_prefix(text: str, section: str) -> str:
    prefix = f"[{section}]\n\n"
    if text.startswith(prefix):
        return text[len(prefix):]
    return re.sub(r'^\[[^\]]*\]\n\n', '', text, count=1)


def _abstract_only(raw_doc: str) -> str:
    marker = "Abstract:\n"
    idx = raw_doc.find(marker)
    return raw_doc[idx + len(marker):].strip() if idx != -1 else raw_doc.strip()


def _authors_str(raw: object, limit: int = 160) -> str:
    if isinstance(raw, list):
        text = ", ".join(str(a) for a in raw)
    else:
        text = str(raw) if raw else ""
    if len(text) > limit:
        cut = text[:limit].rsplit(",", 1)[0]
        return cut + " et al."
    return text


def _passage_card(chunk, idx: int, query: str = "") -> None:
    pct     = format_score(chunk.score)
    section = chunk.section if not chunk.is_summary else "Abstract"
    raw     = _strip_prefix(chunk.text, chunk.section)
    preview = highlight_terms(raw[:520], query) if query else raw[:520]
    doi_html = _doi(chunk.metadata.get("doi", ""))

    st.markdown(
        f'<div class="pcard fi" style="animation-delay:{idx*0.04}s;">'
        f'  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">'
        f'    <div style="min-width:0;flex:1;">'
        f'      <span class="badge b-sec">{section}</span>'
        f'      <span style="font-size:0.83rem;font-weight:600;color:var(--txt);margin-left:6px;">{chunk.citation_label}</span>'
        f'      <span style="font-size:0.75rem;color:var(--txt-3);margin-left:4px;">· {chunk.metadata.get("title","")[:85]}</span>'
        f'    </div>'
        f'    {_badge(pct)}'
        f'  </div>'
        f'  {_bar(pct)}'
        f'  <div style="font-size:0.83rem;line-height:1.65;color:var(--txt-2);margin-top:9px;">{preview}</div>'
        f'  <div class="pcard-meta">'
        f'    <span>📅 {chunk.metadata.get("year","—")}</span>'
        f'    <span>🏥 {chunk.metadata.get("journal","—")[:55]}</span>'
        f'    {doi_html}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Full passage", expanded=False):
        st.markdown(
            f'<div class="sec-body">{highlight_terms(raw, query) if query else raw}</div>',
            unsafe_allow_html=True,
        )


def _paper_card_html(paper: dict) -> str:
    year    = paper.get("year", "—")
    journal = (paper.get("journal") or "")[:55]
    author  = _authors_str(paper.get("authors", ""), limit=60)
    pmc_id  = paper.get("pmc_id", "")
    title   = (paper.get("title") or "Untitled")[:140]
    return (
        f'<div class="pgcard">'
        f'  <h4>{title}</h4>'
        f'  <div class="pgcard-m">📅 {year} · {journal}</div>'
        f'  <div class="pgcard-m">✍️ {author}</div>'
        f'  <div style="margin-top:7px;"><span class="badge b-pmc">{pmc_id}</span></div>'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:0.4rem 0 1.1rem;">'
        '<div style="font-size:1.9rem;margin-bottom:0.35rem;">👂</div>'
        '<div style="font-size:1rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.01em;">ENT Research</div>'
        '<div style="font-size:0.71rem;color:#64748b;margin-top:2px;">Otorhinolaryngology Assistant</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr/>', unsafe_allow_html=True)

    mode = st.radio(
        "Navigation",
        [MODE_ASK, MODE_EXPLORER],
        label_visibility="collapsed",
    )

    st.markdown('<hr/>', unsafe_allow_html=True)

    try:
        stats = get_corpus_stats()
        st.markdown(
            f'<div class="stat-pill">'
            f'  <div class="stat-n">{stats["total_papers"]}</div>'
            f'  <div class="stat-l">Papers indexed</div>'
            f'</div>'
            f'<div class="stat-pill">'
            f'  <div class="stat-n">{stats["total_chunks"]:,}</div>'
            f'  <div class="stat-l">Text chunks</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.caption("⚠️ Could not load stats.")
        stats = {}

    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.67rem;color:#64748b;line-height:1.7;">'
        'ChromaDB · OpenRouter<br/>Nemotron Ultra 253B'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MODE 1 — Ask the Literature
# ──────────────────────────────────────────────────────────────────────────────
if mode == MODE_ASK:
    st.markdown(
        '<div class="hero">'
        '<h1>🔬 Ask the Literature</h1>'
        '<p>Ask any ENT / otorhinolaryngology question and get a citation-backed answer '
        'synthesised from peer-reviewed ENT papers. '
        'Switch to "Ranked Passages" to browse the raw semantic results.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    q_col, j_col = st.columns([5, 1])
    with q_col:
        query = st.text_input(
            "Question",
            placeholder="e.g. How do outcomes compare after TORS vs open surgery for oropharyngeal cancer?",
            label_visibility="collapsed",
            key="qa_query",
        )
    with j_col:
        try:
            journals = ["All journals"] + stats.get("journals", [])
        except Exception:
            journals = ["All journals"]
        journal_sel = st.selectbox("Journal", journals,
                                   label_visibility="collapsed", key="qa_j")

    where: dict | None = (
        {"journal": journal_sel}
        if journal_sel and journal_sel != "All journals" else None
    )

    if query:
        st.markdown('<hr/>', unsafe_allow_html=True)
        tab_ans, tab_passages = st.tabs(["💡  AI Answer", "🎯  Ranked Passages"])

        with tab_ans:
            with st.spinner("Retrieving passages and generating answer…"):
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.write_stream(stream_answer(query, where=where))
                st.markdown('</div>', unsafe_allow_html=True)

            sources = st.session_state.get("_last_sources", [])
            seen: set[str] = set()
            unique = []
            for c in sources:
                if not c.is_summary and c.pmc_id not in seen:
                    seen.add(c.pmc_id)
                    unique.append(c)
            if unique:
                chips = " ".join(
                    f'<span class="cite-chip">{c.citation_label}</span>' for c in unique[:10]
                )
                extra = "<span class='cite-chip'>…</span>" if len(unique) > 10 else ""
                st.markdown(
                    f'<div class="cite-strip">'
                    f'<span style="font-size:0.69rem;color:var(--txt-3);margin-right:5px;">Sources:</span>'
                    f'{chips}{extra}</div>',
                    unsafe_allow_html=True,
                )

        with tab_passages:
            sources = st.session_state.get("_last_sources", [])
            display = [c for c in sources if not c.is_summary]
            if not display:
                st.info("ℹ️ Submit a question to see ranked passages here.")
            else:
                st.markdown(
                    f'<p style="font-size:0.77rem;color:var(--txt-2);margin-bottom:0.75rem;">'
                    f'🔍 <strong style="color:var(--txt);">{len(display)}</strong> '
                    f'passages above relevance threshold</p>',
                    unsafe_allow_html=True,
                )
                for i, chunk in enumerate(display):
                    _passage_card(chunk, i, query=query)


# ──────────────────────────────────────────────────────────────────────────────
# MODE 2 — Paper Explorer
# ──────────────────────────────────────────────────────────────────────────────
elif mode == MODE_EXPLORER:

    # ── Detail panel (shown at top when a paper is selected) ──
    if "selected_paper" in st.session_state:
        paper  = st.session_state["selected_paper"]
        pmc_id = paper.get("pmc_id", "")

        authors_disp = _authors_str(paper.get("authors", ""), limit=200)
        doi_html     = _doi(paper.get("doi", ""))

        st.markdown(
            f'<div class="detail-hdr">'
            f'  <div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;line-height:1.35;margin-bottom:8px;">'
            f'  {paper.get("title", "Untitled")}'
            f'  </div>'
            f'  <div style="font-size:0.81rem;color:var(--txt-2);margin-bottom:6px;">✍️ {authors_disp}</div>'
            f'  <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;font-size:0.77rem;color:var(--txt-2);">'
            f'    <span>📅 {paper.get("year","—")}</span>'
            f'    <span>🏥 {paper.get("journal","—")}</span>'
            f'    <span class="badge b-pmc">{pmc_id}</span>'
            f'    {doi_html}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        b1, b2, _ = st.columns([1.5, 1, 2])
        with b1:
            do_summary = st.button(
                "🧠  Summarize full paper",
                key="sum_btn",
                use_container_width=True,
                type="primary",
            )
        with b2:
            if st.button("← Back to list", key="back_btn", use_container_width=True):
                del st.session_state["selected_paper"]
                st.session_state.pop("_sum_done", None)
                st.session_state.pop("_sum_id", None)
                st.rerun()

        if do_summary:
            st.session_state["_sum_id"]   = pmc_id
            st.session_state["_sum_done"] = False

        if (
            st.session_state.get("_sum_id") == pmc_id
            and not st.session_state.get("_sum_done")
        ):
            with st.expander("🧠  AI Structured Summary", expanded=True):
                with st.spinner("Reading all sections…"):
                    st.write_stream(generate_deep_summary(pmc_id))
            st.session_state["_sum_done"] = True

        # Abstract
        abstract = _abstract_only(paper.get("summary_text", ""))
        if abstract:
            with st.expander("📌  Abstract", expanded=True):
                st.markdown(f'<div class="sec-body">{abstract}</div>', unsafe_allow_html=True)

        # Full sections
        with st.expander("📑  Full paper sections", expanded=False):
            retriever = get_retriever()
            chunks = [c for c in retriever.retrieve_by_paper(pmc_id) if not c.is_summary]
            if not chunks:
                st.caption("No section text available.")
            else:
                for c in chunks:
                    clean = _strip_prefix(c.text, c.section)
                    st.markdown(
                        f'<div style="margin-bottom:6px;"><span class="badge b-sec">{c.section}</span></div>'
                        f'<div class="sec-body">{clean}</div><hr/>',
                        unsafe_allow_html=True,
                    )

        # References
        with st.expander("📚  References", expanded=False):
            retriever = get_retriever()
            all_chunks = retriever.retrieve_by_paper(pmc_id)
            seen_keys: set = set()
            refs: list[dict] = []
            for c in all_chunks:
                for r in c.references:
                    raw_au = r.get("authors", "")
                    au_str = ", ".join(str(a) for a in raw_au) if isinstance(raw_au, list) else str(raw_au)
                    key = (r.get("title",""), r.get("doi",""), r.get("pmid",""), r.get("pmcid",""))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    refs.append({**r, "_au": au_str})

            if not refs:
                st.caption("No structured references stored for this paper.")
            else:
                st.markdown(
                    f'<p style="font-size:0.75rem;color:var(--txt-3);margin-bottom:0.6rem;">{len(refs)} references</p>',
                    unsafe_allow_html=True,
                )
                rows = ""
                for i, r in enumerate(refs, 1):
                    au = r["_au"]
                    au_short = au[:100] + (" et al." if len(au) > 100 else "")
                    parts = []
                    if au_short: parts.append(au_short)
                    if r.get("year"): parts.append(f'({r["year"]})')
                    if r.get("journal"): parts.append(f'<em>{r["journal"][:60]}</em>')
                    if r.get("doi"): parts.append(f'<a class="ref-doi" href="https://doi.org/{r["doi"]}" target="_blank">DOI ↗</a>')
                    rows += (
                        f'<div class="ref-row">'
                        f'  <span style="color:var(--txt-3);margin-right:5px;">{i}.</span>'
                        f'  <span class="ref-t">{r.get("title","Untitled")}</span>'
                        f'  <div style="margin-top:3px;">{" ".join(parts)}</div>'
                        f'</div>'
                    )
                st.markdown(rows, unsafe_allow_html=True)

        st.markdown('<hr/>', unsafe_allow_html=True)

    # ── Paper grid ──
    st.markdown(
        '<div class="hero">'
        '<h1>📄 Paper Explorer</h1>'
        '<p>Browse the ENT corpus. Click any paper to view its full text, abstract, and references.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    sc, _ = st.columns([3, 2])
    with sc:
        search_q = st.text_input(
            "Search",
            placeholder="Title, otorhinolaryngology topic, author, or PMC ID…",
            label_visibility="collapsed",
            key="exp_search",
        )

    try:
        papers = search_papers_by_title(search_q) if search_q else get_all_papers()
        n = len(papers)
        label = (
            f'{n} result{"s" if n != 1 else ""} for "{search_q}"'
            if search_q else f'Showing all {n} papers'
        )
        st.markdown(f'<p style="font-size:0.76rem;color:var(--txt-2);margin-bottom:0.75rem;">{label}</p>',
                    unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Could not load papers: {exc}")
        papers = []

    COLS = 3
    for row_start in range(0, len(papers), COLS):
        cols = st.columns(COLS)
        for col, paper in zip(cols, papers[row_start: row_start + COLS]):
            with col:
                st.markdown(_paper_card_html(paper), unsafe_allow_html=True)
                if st.button("View details",
                             key=f"v_{paper.get('pmc_id','')}_{row_start}",
                             use_container_width=True):
                    st.session_state["selected_paper"] = paper
                    st.session_state.pop("_sum_done", None)
                    st.session_state.pop("_sum_id", None)
                    st.rerun()
