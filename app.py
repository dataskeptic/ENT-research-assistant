"""
app.py  —  ENT Research Assistant

Modes:
  1. Ask the Literature  — Q&A + ranked semantic passages (merged)
  2. Paper Explorer      — Browse corpus, full text, references, LLM summary
"""
from __future__ import annotations

import re

import streamlit as st
import streamlit.components.v1 as components

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
    page_icon="image/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --ink-black: #011627;
    --porcelain: #fdfffc;
    --light-sea-green: #2ec4b6;

    --bg:            var(--ink-black);
    --bg-card:       rgba(253, 255, 252, 0.03);
    --bg-card-hover: rgba(253, 255, 252, 0.06);
    --accent:        var(--light-sea-green);
    --accent-dim:    rgba(46, 196, 182, 0.15);
    --accent-light:  rgba(46, 196, 182, 0.8);
    --accent-bright: var(--light-sea-green);
    --txt:           var(--porcelain);
    --txt-2:         rgba(253, 255, 252, 0.75);
    --txt-3:         rgba(253, 255, 252, 0.5);
    --border:        rgba(46, 196, 182, 0.2);
    --radius:        14px;
    --tr:            0.3s cubic-bezier(.4,0,.2,1);
    
    /* Elegant Shadows */
    --shadow-base:   0 4px 14px rgba(0, 0, 0, 0.3);
    --shadow-hover:  0 10px 24px rgba(0, 0, 0, 0.5);
    --shadow-glow:   0 0 20px rgba(46, 196, 182, 0.2);
}

/* ── Universal font ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Full-page dark background ── */
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
[data-testid="stSidebar"] {
    background-color: var(--bg) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.5) !important;
}

/* ── Markdown / text colour ── */
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p {
    color: var(--txt) !important;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: rgba(253, 255, 252, 0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.2rem !important;
    color: var(--txt) !important;
    caret-color: var(--accent);
    box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.2);
    transition: all var(--tr);
}
.stTextInput > div > div > input::placeholder { color: var(--txt-3) !important; font-weight: 300; }
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.2), 0 0 0 3px var(--accent-dim) !important;
    outline: none;
}

/* ── Buttons ── */
div.stButton > button {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--txt) !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
    transition: all var(--tr) !important;
    letter-spacing: 0.02em;
}
div.stButton > button:hover {
    border-color: var(--accent) !important;
    background: var(--bg-card-hover) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.3), var(--shadow-glow) !important;
}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: var(--ink-black) !important;
    font-weight: 600 !important;
    text-shadow: none;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
    background: var(--porcelain) !important;
    border-color: var(--porcelain) !important;
    color: var(--ink-black) !important;
    box-shadow: 0 8px 20px rgba(46, 196, 182, 0.4), var(--shadow-glow) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div,
[data-testid="stSelectbox"] > div > div {
    background: rgba(253, 255, 252, 0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--txt) !important;
    box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.2);
}

/* ── Expander ── */
details[data-testid="stExpander"],
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-base);
    transition: all var(--tr);
}
details[data-testid="stExpander"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stExpander"] summary {
    color: var(--txt) !important;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    padding: 0;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0 !important;
    font-weight: 500;
    color: var(--txt-2) !important;
    background: transparent !important;
    padding: 10px 16px;
    transition: all var(--tr);
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.5rem;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Info / warning boxes ── */
[data-testid="stAlert"],
.stAlert {
    background: var(--bg-card) !important;
    border: 1px solid var(--accent-dim) !important;
    border-left: 4px solid var(--accent) !important;
    border-radius: 12px !important;
    color: var(--txt) !important;
    box-shadow: var(--shadow-base);
}

/* ── Divider ── */
hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── Highlight (HTML mark tag used by highlight_terms) ── */
mark {
    background: var(--accent-dim);
    color: var(--accent) !important;
    border-radius: 4px;
    padding: 0 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
strong { color: var(--porcelain) !important; font-weight: 700; }

/* ── Sidebar logo mark ── */
.ent-logo {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.8rem;
}
.ent-logo img {
    width: 120px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}

/* ── Hero ── */
.hero { padding: 2rem 0 1.5rem; }
.hero h1 {
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--porcelain);
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 0.8rem;
    text-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.hero p {
    font-size: 1.05rem;
    color: var(--txt-2);
    line-height: 1.6;
    max-width: 700px;
}

/* ── Stat pill ── */
.stat-pill {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 18px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 12px;
    box-shadow: var(--shadow-base);
    transition: all var(--tr);
}
.stat-pill:hover {
    border-color: var(--accent-dim);
    background: var(--bg-card-hover);
    transform: translateY(-2px);
}
.stat-n { font-size: 1.5rem; font-weight: 700; color: var(--accent); min-width: 40px; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
.stat-l { font-size: 0.75rem; color: var(--txt-2); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 500; }

/* ── Badges ── */
.badge {
    display: inline-flex; align-items: center;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem; font-weight: 600; white-space: nowrap;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.b-high { background: var(--accent-dim); color: var(--accent); border: 1px solid var(--accent); }
.b-mid  { background: rgba(253, 255, 252, 0.1); color: var(--porcelain); border: 1px solid var(--border); }
.b-low  { background: rgba(253, 255, 252, 0.05); color: var(--txt-3); border: 1px solid rgba(253, 255, 252, 0.1); }
.b-sec  {
    background: rgba(46, 196, 182, 0.1); color: var(--accent);
    border-radius: 6px; text-transform: uppercase;
    letter-spacing: 0.08em; font-size: 0.7rem; padding: 4px 10px;
    border: 1px solid var(--accent-dim);
}
.b-doi  { background: rgba(253, 255, 252, 0.05); color: var(--txt-2); border: 1px solid var(--border); }

/* ── Score bar ── */
.strack  { background: rgba(253, 255, 252, 0.1); border-radius: 99px; height: 4px; overflow: hidden; margin-top: 10px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.4); }
.sfill   { height: 100%; border-radius: 99px; box-shadow: 0 0 8px var(--accent-dim); }

/* ── Passage card ── */
.pcard {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow-base);
    backdrop-filter: blur(10px);
    transition: all var(--tr);
}
.pcard:hover { 
    border-color: var(--accent); 
    box-shadow: var(--shadow-hover), var(--shadow-glow); 
    transform: translateY(-3px); 
}
.pcard-meta { font-size: 0.8rem; color: var(--txt-3); margin-top: 12px; display: flex; flex-wrap: wrap; gap: 14px; font-weight: 400; }

/* ── Paper grid card ── */
.pgcard {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    height: 100%;
    box-shadow: var(--shadow-base);
    backdrop-filter: blur(10px);
    transition: all var(--tr);
}
.pgcard:hover { 
    border-color: var(--accent); 
    transform: translateY(-4px); 
    box-shadow: var(--shadow-hover), var(--shadow-glow); 
}
.pgcard h4 {
    font-size: 1rem; font-weight: 600; color: var(--porcelain); line-height: 1.5; margin-bottom: 10px;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.pgcard-m { font-size: 0.8rem; color: var(--txt-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 6px; font-weight: 400; }

/* ── Paper detail header ── */
.detail-hdr {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-hover);
}

/* ── Answer box ── */
.answer-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    font-size: 1rem;
    line-height: 1.8;
    color: var(--txt);
    box-shadow: var(--shadow-base);
}

/* ── Citation strip ── */
.cite-strip {
    display: flex; flex-wrap: wrap; gap: 8px;
    padding: 1rem 1.2rem;
    background: rgba(253, 255, 252, 0.02);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-top: 1rem;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
}
.cite-chip {
    font-size: 0.75rem; color: var(--porcelain); font-weight: 500;
    background: rgba(253, 255, 252, 0.08);
    padding: 4px 12px; border-radius: 99px;
    border: 1px solid rgba(253, 255, 252, 0.15);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    transition: all var(--tr);
}
.cite-chip:hover {
    background: var(--accent);
    color: var(--ink-black);
    border-color: var(--accent);
}

/* ── Section / reference text ── */
.sec-body {
    font-size: 0.95rem; line-height: 1.7;
    color: var(--txt); white-space: pre-wrap;
}
.ref-row {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem; line-height: 1.6; color: var(--txt-2);
}
.ref-row:last-child { border-bottom: none; }
.ref-t   { font-weight: 500; color: var(--porcelain); }
.ref-doi { color: var(--accent); text-decoration: none; font-size: 0.8rem; font-weight: 500; }
.ref-doi:hover { text-decoration: underline; color: var(--accent-light); }

/* ── Fade-up animation ── */
@keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
.fi { animation: fadeUp 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) both; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MODE_ASK      = "🔬 Ask the Literature"
MODE_EXPLORER = "📂 Paper Explorer"

if "app_mode" not in st.session_state:
    st.session_state.app_mode = MODE_ASK

mode = st.session_state.app_mode


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _badge(pct: int) -> str:
    cls = "b-high" if pct >= 75 else ("b-mid" if pct >= 50 else "b-low")
    return f'<span class="badge {cls}">{pct}%</span>'


def _bar(pct: int) -> str:
    c = "var(--accent)" if pct >= 75 else ("var(--accent-dim)" if pct >= 50 else "rgba(253, 255, 252, 0.1)")
    return f'<div class="strack"><div class="sfill" style="width:{pct}%;background:{c};"></div></div>'


def _doi(doi: str) -> str:
    if not doi:
        return ""
    return f'<a class="ref-doi" href="https://doi.org/{doi}" target="_blank">DOI ↗</a>'


def _strip_prefix(text: str, section: str) -> str:
    """Remove leading '[SECTION]\n\n' prefix that the ingestion pipeline prepends."""
    prefix = f"[{section}]\n\n"
    if text.startswith(prefix):
        return text[len(prefix):]
    # fallback: strip any bracketed prefix
    return re.sub(r'^\[[^\]]*\]\n\n', '', text, count=1)


def _abstract_only(raw_doc: str) -> str:
    """Extract just the abstract body from a summary doc."""
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
    """Render a single passage card with relevance badge, score bar, and full-text expander."""
    pct     = format_score(chunk.score)
    section = chunk.section if not chunk.is_summary else "Abstract"
    raw     = _strip_prefix(chunk.text, chunk.section)
    # highlight_terms returns HTML with <mark> tags — safe to embed
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
        f'    <span>{chunk.metadata.get("year","—")}</span>'
        f'    <span>{chunk.metadata.get("journal","—")[:55]}</span>'
        f'    {doi_html}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Full passage", expanded=False):
        full_html = highlight_terms(raw, query) if query else raw
        st.markdown(
            f'<div class="sec-body">{full_html}</div>',
            unsafe_allow_html=True,
        )


def _paper_card_html(paper: dict) -> str:
    year    = paper.get("year", "—")
    journal = (paper.get("journal") or "")[:55]
    author  = _authors_str(paper.get("authors", ""), limit=60)
    doi     = paper.get("doi", "")
    title   = (paper.get("title") or "Untitled")[:140]
    return (
        f'<div class="pgcard">'
        f'  <h4>{title}</h4>'
        f'  <div class="pgcard-m">{year} · {journal}</div>'
        f'  <div class="pgcard-m">{author}</div>'
        f'  <div style="margin-top:7px;"><span class="badge b-doi">{doi}</span></div>'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("image/logo.png", use_container_width=True)
    st.markdown(
        """
        <div style="padding:0.4rem 0 1.1rem; text-align: center;">
          <div style="font-size:1.5rem;font-weight:800;color:var(--porcelain);
                      letter-spacing:0.1em;font-family:'Inter',sans-serif;
                      text-shadow: 0 4px 8px rgba(0,0,0,0.5);">
            ENT AI
          </div>
          <div style="font-size:0.75rem;color:var(--accent);margin-top:6px;font-weight:600;letter-spacing:0.08em;">
            LATEST PAPERS (PAST MONTH)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr/>', unsafe_allow_html=True)

    if st.button(MODE_ASK, use_container_width=True, type="primary" if mode == MODE_ASK else "secondary"):
        st.session_state.app_mode = MODE_ASK
        st.rerun()
    
    if st.button(MODE_EXPLORER, use_container_width=True, type="primary" if mode == MODE_EXPLORER else "secondary"):
        st.session_state.app_mode = MODE_EXPLORER
        st.rerun()

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
        st.caption("Could not load stats.")
        stats = {}

    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.7rem;color:var(--txt-3);line-height:1.7;">'
        'ChromaDB · OpenRouter<br/>Nemotron Ultra 253B'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MODE 1 — Ask the Literature  (Q&A + semantic passages merged)
# ──────────────────────────────────────────────────────────────────────────────
if mode == MODE_ASK:
    st.markdown(
        '<div class="hero">'
        '<h1>Ask the Literature</h1>'
        '<p>Ask any ENT / otorhinolaryngology question and get a citation-backed answer '
        'synthesised from peer-reviewed papers published <strong>this last month</strong>. '
        'Switch to <strong style="color:var(--accent);">Ranked Passages</strong> to browse the raw semantic results.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── query row + optional journal filter ──
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
        journal_sel = st.selectbox(
            "Journal", journals,
            label_visibility="collapsed",
            key="qa_j",
        )

    where: dict | None = (
        {"journal": journal_sel}
        if journal_sel and journal_sel != "All journals" else None
    )

    if query:
        st.markdown('<hr/>', unsafe_allow_html=True)
        tab_ans, tab_passages = st.tabs(["💡 AI Answer", "🎯 Ranked Passages"])

        with tab_ans:
            with st.spinner("Retrieving passages and generating answer…"):
                answer_box = st.empty()
                # Collect the full streamed answer so we can wrap it in the styled div.
                # st.write_stream returns the full string when the generator is exhausted.
                full_answer = answer_box.write_stream(
                    stream_answer(query, where=where)
                )

            # Citation strip — deduplicated paper labels from retrieved chunks
            sources = st.session_state.get("_last_sources", [])
            seen: set[str] = set()
            unique = []
            for c in sources:
                if not c.is_summary and c.doi not in seen:
                    seen.add(c.doi)
                    unique.append(c)
            if unique:
                chips = " ".join(
                    f'<span class="cite-chip">{c.citation_label}</span>'
                    for c in unique[:10]
                )
                extra = "<span class='cite-chip'>…</span>" if len(unique) > 10 else ""
                st.markdown(
                    f'<div class="cite-strip">'
                    f'<span style="font-size:0.69rem;color:var(--txt-3);margin-right:5px;">Sources:</span>'
                    f'{chips}{extra}</div>',
                    unsafe_allow_html=True,
                )

        with tab_passages:
            # Re-read sources (populated during stream_answer above)
            sources = st.session_state.get("_last_sources", [])
            display = [c for c in sources if not c.is_summary]
            if not display:
                st.info("Submit a question first to see ranked passages here.")
            else:
                st.markdown(
                    f'<p style="font-size:0.77rem;color:var(--txt-2);margin-bottom:0.75rem;">'
                    f'<strong style="color:var(--txt);">{len(display)}</strong> '
                    f'passages above relevance threshold</p>',
                    unsafe_allow_html=True,
                )
                for i, chunk in enumerate(display):
                    _passage_card(chunk, i, query=query)


# ──────────────────────────────────────────────────────────────────────────────
# MODE 2 — Paper Explorer
# ──────────────────────────────────────────────────────────────────────────────
elif mode == MODE_EXPLORER:

    # ── Detail panel — shown ABOVE the grid when a paper is selected ──
    if "selected_paper" in st.session_state:
        paper  = st.session_state["selected_paper"]
        doi_id = paper.get("doi", "")

        authors_disp = _authors_str(paper.get("authors", ""), limit=200)
        doi_html     = _doi(paper.get("doi", ""))

        # ── Header card ──
        st.markdown(
            f'<div class="detail-hdr">'
            f'  <div style="font-size:1.1rem;font-weight:700;color:var(--porcelain);line-height:1.4;margin-bottom:8px;">'
            f'    {paper.get("title", "Untitled")}'
            f'  </div>'
            f'  <div style="font-size:0.85rem;color:var(--txt-2);margin-bottom:8px;">{authors_disp}</div>'
            f'  <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;font-size:0.8rem;color:var(--txt-2);">'
            f'    <span>{paper.get("year","—")}</span>'
            f'    <span>{paper.get("journal","—")}</span>'
            f'    <span class="badge b-doi">{doi_id}</span>'
            f'    {doi_html}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Action buttons ──
        b1, b2, _ = st.columns([1.5, 1, 2])
        with b1:
            do_summary = st.button(
                "🧠 Summarize full paper",
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

        # ── LLM summary (streamed on demand) ──
        if do_summary:
            st.session_state["_sum_id"]   = doi_id
            st.session_state["_sum_done"] = False

        if (
            st.session_state.get("_sum_id") == doi_id
            and not st.session_state.get("_sum_done")
        ):
            with st.expander("🤖 AI Structured Summary", expanded=True):
                with st.spinner("Reading all paper sections…"):
                    st.write_stream(generate_deep_summary(doi_id))
            st.session_state["_sum_done"] = True

        # ── Abstract ──
        abstract = _abstract_only(paper.get("summary_text", ""))
        if abstract:
            with st.expander("📋 Abstract", expanded=True):
                st.markdown(
                    f'<div class="sec-body">{abstract}</div>',
                    unsafe_allow_html=True,
                )

        # ── Full sections (no truncation) ──
        with st.expander("📑 Full paper sections", expanded=False):
            retriever = get_retriever()
            chunks = [
                c for c in retriever.retrieve_by_paper(doi_id)
                if not c.is_summary
            ]
            if not chunks:
                st.caption("No section text available for this paper.")
            else:
                for c in chunks:
                    clean = _strip_prefix(c.text, c.section)
                    st.markdown(
                        f'<div style="margin-bottom:6px;">'
                        f'  <span class="badge b-sec">{c.section}</span>'
                        f'</div>'
                        f'<div class="sec-body">{clean}</div>'
                        f'<hr/>',
                        unsafe_allow_html=True,
                    )

        # ── References ──
        with st.expander("📚 References", expanded=False):
            retriever   = get_retriever()
            all_chunks  = retriever.retrieve_by_paper(doi_id)
            seen_keys: set = set()
            refs: list[dict] = []
            for c in all_chunks:
                for r in (c.references or []):
                    raw_au = r.get("authors", "")
                    au_str = (
                        ", ".join(str(a) for a in raw_au)
                        if isinstance(raw_au, list)
                        else str(raw_au)
                    )
                    key = (
                        r.get("title", ""),
                        r.get("doi", ""),
                        r.get("pmid", ""),
                        r.get("pmcid", ""),
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    refs.append({**r, "_au": au_str})

            if not refs:
                st.caption("No structured references stored for this paper.")
            else:
                st.markdown(
                    f'<p style="font-size:0.75rem;color:var(--txt-3);margin-bottom:0.6rem;">'
                    f'{len(refs)} references</p>',
                    unsafe_allow_html=True,
                )
                rows = ""
                for i, r in enumerate(refs, 1):
                    au      = r["_au"]
                    au_disp = au[:100] + (" et al." if len(au) > 100 else "")
                    parts   = []
                    if au_disp:       parts.append(au_disp)
                    if r.get("year"): parts.append(f'({r["year"]})')
                    if r.get("journal"):
                        parts.append(f'<em>{r["journal"][:60]}</em>')
                    if r.get("doi"):
                        parts.append(
                            f'<a class="ref-doi" href="https://doi.org/{r["doi"]}"'
                            f' target="_blank">DOI ↗</a>'
                        )
                    rows += (
                        f'<div class="ref-row">'
                        f'  <span style="color:var(--txt-3);margin-right:5px;">{i}.</span>'
                        f'  <span class="ref-t">{r.get("title", "Untitled")}</span>'
                        f'  <div style="margin-top:3px;">{" ".join(parts)}</div>'
                        f'</div>'
                    )
                st.markdown(rows, unsafe_allow_html=True)

        st.markdown('<hr/>', unsafe_allow_html=True)

    # ── Paper grid (ONLY SHOWN IF NO PAPER IS SELECTED) ──
    else:
        st.markdown(
            '<div class="hero">'
            '<h1>Paper Explorer</h1>'
            '<p>Browse the cutting-edge ENT corpus from <strong>this last month</strong>. Click any paper to view its full text, abstract, references, '
            'and trigger an AI-generated structured summary.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        sc, _ = st.columns([3, 2])
        with sc:
            search_q = st.text_input(
                "Search",
                placeholder="Title, topic, author, or DOI…",
                label_visibility="collapsed",
                key="exp_search",
            )

        try:
            papers = search_papers_by_title(search_q) if search_q else get_all_papers()
            n      = len(papers)
            label  = (
                f'{n} result{"s" if n != 1 else ""} for "{search_q}"'
                if search_q
                else f'Showing all {n} papers'
            )
            st.markdown(
                f'<p style="font-size:0.76rem;color:var(--txt-2);margin-bottom:0.75rem;">{label}</p>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Could not load papers: {exc}")
            papers = []

        COLS = 3
        for row_start in range(0, len(papers), COLS):
            cols = st.columns(COLS)
            for col, paper in zip(cols, papers[row_start : row_start + COLS]):
                with col:
                    st.markdown(_paper_card_html(paper), unsafe_allow_html=True)
                    if st.button(
                        "View details",
                        key=f"v_{paper.get('doi', '')}_{row_start}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_paper"] = paper
                        st.session_state["_scroll_to_top"] = True
                        st.session_state.pop("_sum_done", None)
                        st.session_state.pop("_sum_id", None)
                        st.rerun()
