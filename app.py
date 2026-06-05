"""
app.py

Streamlit front-end for the ENT Research Assistant RAG pipeline.

Features:
  1. 🔬 Ask the Literature  — Q&A with citation-backed, streamed answers
                            + ranked semantic passages in a second tab
  2. 📄 Paper Explorer     — Browse & read full papers with references,
                            trigger LLM structured summary

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
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

# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ENT Research Assistant",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..700;1,14..32,300..500&display=swap');

/* ─── Base tokens ───────────────────────────────────────────────────── */
:root {
    --bg:            #0b0f1a;
    --surface:       #111827;
    --surface-2:     #161d2e;
    --surface-3:     #1c2540;
    --border:        rgba(255,255,255,0.07);
    --border-accent: rgba(99,130,246,0.22);
    --primary:       #4f8ef7;
    --primary-dim:   rgba(79,142,247,0.12);
    --cyan:          #22d3ee;
    --emerald:       #34d399;
    --amber:         #fbbf24;
    --rose:          #fb7185;
    --txt:           #e2e8f0;
    --txt-2:         #94a3b8;
    --txt-3:         #64748b;
    --radius-sm:     8px;
    --radius-md:     12px;
    --radius-lg:     16px;
    --transition:    0.2s cubic-bezier(.4,0,.2,1);
    --shadow-sm:     0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md:     0 4px 16px rgba(0,0,0,0.5);
    --shadow-lg:     0 12px 40px rgba(0,0,0,0.6);
}

/* ─── Global reset / base ────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    color: var(--txt);
}

.main, .main .block-container {
    background: var(--bg) !important;
}

/* ─── Sidebar ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

/* ─── Typography ────────────────────────────────────────────────── */
.page-title {
    font-size: 1.55rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    line-height: 1.2;
    color: var(--txt);
    margin-bottom: 0.3rem;
}
.page-subtitle {
    font-size: 0.875rem;
    color: var(--txt-2);
    line-height: 1.55;
    max-width: 620px;
    margin-bottom: 1.5rem;
}

/* ─── Stat cards ───────────────────────────────────────────────── */
.stat-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    margin-bottom: 8px;
}
.stat-num {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1;
    min-width: 44px;
}
.stat-lbl {
    font-size: 0.75rem;
    color: var(--txt-2);
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ─── Score badges ───────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    white-space: nowrap;
}
.badge-high   { background: rgba(52,211,153,0.12); color: #34d399; }
.badge-mid    { background: rgba(251,191,36,0.12);  color: #fbbf24; }
.badge-low    { background: rgba(251,113,133,0.12); color: #fb7185; }
.badge-section{
    background: rgba(34,211,238,0.10);
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 4px;
}
.badge-pmc {
    background: rgba(79,142,247,0.10);
    color: var(--primary);
    font-size: 0.72rem;
    padding: 2px 9px;
    border-radius: 99px;
}

/* ─── Score bar ──────────────────────────────────────────────────── */
.score-track {
    background: rgba(255,255,255,0.05);
    border-radius: 99px;
    height: 4px;
    overflow: hidden;
    margin-top: 8px;
}
.score-fill { height: 100%; border-radius: 99px; }

/* ─── Passage cards ────────────────────────────────────────────── */
.pcard {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color var(--transition), box-shadow var(--transition);
}
.pcard:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-sm);
}
.pcard-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--txt);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pcard-text {
    font-size: 0.84rem;
    line-height: 1.65;
    color: var(--txt-2);
    margin-top: 8px;
}
.pcard-meta {
    font-size: 0.73rem;
    color: var(--txt-3);
    margin-top: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
}

/* ─── Paper grid cards ─────────────────────────────────────────── */
.pgcard {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.9rem 1rem;
    height: 100%;
    transition: border-color var(--transition), transform var(--transition);
    cursor: pointer;
}
.pgcard:hover {
    border-color: var(--border-accent);
    transform: translateY(-2px);
    box-shadow: var(--shadow-sm);
}
.pgcard h4 {
    font-size: 0.85rem;
    font-weight: 600;
    line-height: 1.4;
    color: var(--txt);
    margin-bottom: 6px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.pgcard-meta {
    font-size: 0.73rem;
    color: var(--txt-2);
    margin-bottom: 3px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ─── Paper detail panel ───────────────────────────────────────── */
.detail-header {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.detail-title {
    font-size: 1.1rem;
    font-weight: 650;
    color: var(--txt);
    line-height: 1.3;
    margin-bottom: 0.6rem;
}
.detail-authors {
    font-size: 0.82rem;
    color: var(--txt-2);
    margin-bottom: 0.5rem;
}
.detail-meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    font-size: 0.78rem;
    color: var(--txt-2);
}

/* ─── Section body text (strip ingestion prefix) ─────────────────── */
.section-body {
    font-size: 0.88rem;
    line-height: 1.75;
    color: var(--txt-2);
    white-space: pre-wrap;
}

/* ─── References ─────────────────────────────────────────────────── */
.ref-row {
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.81rem;
    line-height: 1.55;
    color: var(--txt-2);
}
.ref-row:last-child { border-bottom: none; }
.ref-title { font-weight: 600; color: var(--txt); }
.ref-doi { color: var(--primary); text-decoration: none; font-size: 0.76rem; }
.ref-doi:hover { text-decoration: underline; }

/* ─── Answer box ─────────────────────────────────────────────────── */
.answer-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    font-size: 0.9rem;
    line-height: 1.8;
    color: var(--txt);
    margin-bottom: 0.75rem;
}

/* ─── Citation strip ────────────────────────────────────────────── */
.cite-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 0.75rem;
    padding: 0.65rem 1rem;
    background: var(--surface-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
}
.cite-chip {
    font-size: 0.72rem;
    color: var(--primary);
    background: var(--primary-dim);
    padding: 2px 10px;
    border-radius: 99px;
    white-space: nowrap;
}

/* ─── Streamlit widget overrides ─────────────────────────────────── */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--txt) !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.15) !important;
}
div.stButton > button {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--txt) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    transition: all var(--transition) !important;
}
div.stButton > button:hover {
    border-color: var(--border-accent) !important;
    background: var(--surface-3) !important;
}
div.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #3d7de8 !important;
    box-shadow: 0 0 16px rgba(79,142,247,0.35) !important;
}
.stSelectbox > label, .stTextInput > label {
    font-size: 0.78rem !important;
    color: var(--txt-2) !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--txt) !important;
}
details[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
details[data-testid="stExpander"] summary {
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: var(--txt-2) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: var(--radius-sm) !important;
    padding: 3px;
    gap: 2px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: var(--txt-2) !important;
    padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface-3) !important;
    color: var(--txt) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
hr { border: none; border-top: 1px solid var(--border); margin: 1.25rem 0; }

/* Highlight terms */
mark, b { color: #93c5fd; font-weight: 600; }

/* Fade in */
@keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
.fade-in { animation: fadeIn 0.3s ease-out both; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _score_badge(pct: int) -> str:
    cls = "badge-high" if pct >= 75 else ("badge-mid" if pct >= 50 else "badge-low")
    return f'<span class="badge {cls}">{pct}% match</span>'


def _score_bar(pct: int) -> str:
    colour = "var(--emerald)" if pct >= 75 else ("var(--amber)" if pct >= 50 else "var(--rose)")
    return (
        f'<div class="score-track">'
        f'<div class="score-fill" style="width:{pct}%;background:{colour};"></div>'
        f'</div>'
    )


def _doi_link(doi: str) -> str:
    if doi:
        return f'<a class="ref-doi" href="https://doi.org/{doi}" target="_blank">DOI ↗</a>'
    return ""


def _strip_section_prefix(text: str, section_name: str) -> str:
    """
    Ingestion prefixes each chunk with "[SectionName]\n\n".
    Strip that prefix before displaying so the title doesn't repeat.
    """
    prefix = f"[{section_name}]\n\n"
    if text.startswith(prefix):
        return text[len(prefix):]
    # Fallback: strip any leading [Anything]\n\n pattern
    return re.sub(r'^\[[^\]]*\]\n\n', '', text, count=1)


def _extract_abstract_from_summary_doc(raw_doc: str) -> str:
    """
    The stored summary doc is:
        Title: ...
        Authors: ...
        Journal: ...
        DOI: ...

        Abstract:
        <abstract text>

    Return only the abstract text (everything after 'Abstract:\n').
    If no such marker is found, return raw_doc as fallback.
    """
    marker = "Abstract:\n"
    idx = raw_doc.find(marker)
    if idx != -1:
        return raw_doc[idx + len(marker):].strip()
    return raw_doc.strip()


def _authors_display(raw: object, max_chars: int = 120) -> str:
    """Safely render authors regardless of whether stored as str or list."""
    if isinstance(raw, list):
        text = ", ".join(str(a) for a in raw)
    else:
        text = str(raw) if raw else ""
    if len(text) > max_chars:
        # Truncate at last comma before limit
        cut = text[:max_chars].rsplit(",", 1)[0]
        return cut + " et al."
    return text


def _render_passage_card(chunk, idx: int, query: str = "") -> None:
    pct = format_score(chunk.score)
    doi_html = _doi_link(chunk.metadata.get("doi", ""))
    section = chunk.section if not chunk.is_summary else "Abstract"
    title_short = chunk.metadata.get('title', '')[:100]
    raw_text = _strip_section_prefix(chunk.text, chunk.section)
    preview = highlight_terms(raw_text[:550], query) if query else raw_text[:550]

    st.markdown(
        f'<div class="pcard fade-in" style="animation-delay:{idx * 0.04}s;">'
        f'  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">'
        f'    <div style="min-width:0;">'
        f'      <span class="badge badge-section">{section}</span>'
        f'      <span class="pcard-title" style="display:inline;vertical-align:middle;margin-left:6px;">{chunk.citation_label}</span>'
        f'      <span style="font-size:0.77rem;color:var(--txt-3);margin-left:4px;">· {title_short}</span>'
        f'    </div>'
        f'    {_score_badge(pct)}'
        f'  </div>'
        f'  {_score_bar(pct)}'
        f'  <div class="pcard-text">{preview}</div>'
        f'  <div class="pcard-meta">'
        f'    <span>📅 {chunk.metadata.get("year", "—")}</span>'
        f'    <span>🏥 {chunk.metadata.get("journal", "—")[:55]}</span>'
        f'    {doi_html}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("📖 Full passage", expanded=False):
        full = highlight_terms(raw_text, query) if query else raw_text
        st.markdown(
            f'<div class="section-body">{full}</div>',
            unsafe_allow_html=True,
        )


def _render_paper_grid_card(paper: dict) -> str:
    year    = paper.get("year", "—")
    journal = paper.get("journal", "")
    first_author = _authors_display(paper.get("authors", ""), max_chars=60)
    pmc_id  = paper.get("pmc_id", "")
    return (
        f'<div class="pgcard">'
        f'  <h4>{paper.get("title", "Untitled")[:140]}</h4>'
        f'  <div class="pgcard-meta">📅 {year} · {journal[:55]}</div>'
        f'  <div class="pgcard-meta" style="margin-top:3px;">✍️ {first_author}</div>'
        f'  <div style="margin-top:6px;"><span class="badge badge-pmc">{pmc_id}</span></div>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="padding:0.25rem 0 1rem;">'
        '<div style="font-size:1.8rem;line-height:1;margin-bottom:0.5rem;">��</div>'
        '<div style="font-size:1rem;font-weight:650;color:var(--txt);letter-spacing:-0.01em;">ENT Research</div>'
        '<div style="font-size:0.72rem;color:var(--txt-3);margin-top:2px;">Research Assistant</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="margin:0 0 1rem;"/>', unsafe_allow_html=True)

    mode = st.radio(
        "Mode",
        ["\U0001f52c Ask the Literature", "\U0001f4c4 Paper Explorer"],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="margin:0.75rem 0;"/>', unsafe_allow_html=True)

    # Corpus stats
    try:
        stats = get_corpus_stats()
        st.markdown(
            f'<div class="stat-pill">'
            f'  <div class="stat-num">{stats["total_papers"]}</div>'
            f'  <div class="stat-lbl">Papers indexed</div>'
            f'</div>'
            f'<div class="stat-pill">'
            f'  <div class="stat-num">{stats["total_chunks"]:,}</div>'
            f'  <div class="stat-lbl">Text chunks</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.caption("⚠️ Could not load stats.")
        stats = {}

    st.markdown('<hr style="margin:0.75rem 0;"/>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.68rem;color:var(--txt-3);line-height:1.6;">'
        'ChromaDB · OpenRouter<br/>Nemotron Ultra 253B'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Ask the Literature
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "f52c Ask the Literature":
    st.markdown('<div class="page-title">🔬 Ask the Literature</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Ask any clinical or research question. '
        'Get a citation-backed answer synthesised from peer-reviewed ENT papers, '
        'with ranked source passages in the second tab.</div>',
        unsafe_allow_html=True,
    )

    query_col, filter_col = st.columns([5, 1])
    with query_col:
        query = st.text_input(
            "Question",
            placeholder="e.g. What are the outcomes after TORS for oropharyngeal cancer?",
            label_visibility="collapsed",
            key="qa_query",
        )
    with filter_col:
        try:
            all_journals = ["All journals"] + stats.get("journals", [])
        except Exception:
            all_journals = ["All journals"]
        journal_filter = st.selectbox(
            "Journal", all_journals,
            label_visibility="collapsed",
            key="qa_journal",
        )

    where: dict | None = None
    if journal_filter and journal_filter != "All journals":
        where = {"journal": journal_filter}

    if query:
        st.markdown('<hr/>', unsafe_allow_html=True)
        answer_tab, passages_tab = st.tabs(["\U0001f4a1 AI Answer", "\U0001f3af Ranked Passages"])

        with answer_tab:
            with st.spinner("Retrieving passages and generating answer…"):
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.write_stream(stream_answer(query, where=where))
                st.markdown('</div>', unsafe_allow_html=True)

            sources = st.session_state.get("_last_sources", [])
            if sources:
                seen_pmc: set[str] = set()
                unique_papers = []
                for c in sources:
                    if not c.is_summary and c.pmc_id not in seen_pmc:
                        seen_pmc.add(c.pmc_id)
                        unique_papers.append(c)
                if unique_papers:
                    chips = " ".join(
                        f'<span class="cite-chip">{c.citation_label}</span>'
                        for c in unique_papers[:10]
                    )
                    ellipsis = "<span class='cite-chip'>…</span>" if len(unique_papers) > 10 else ""
                    st.markdown(
                        f'<div class="cite-strip">'
                        f'<span style="font-size:0.7rem;color:var(--txt-3);margin-right:4px;">Sources:</span>'
                        f'{chips}{ellipsis}</div>',
                        unsafe_allow_html=True,
                    )

        with passages_tab:
            sources = st.session_state.get("_last_sources", [])
            display_chunks = [c for c in sources if not c.is_summary]
            if not display_chunks:
                st.info("ℹ️ Submit a question first to see ranked passages here.")
            else:
                st.markdown(
                    f'<p style="font-size:0.78rem;color:var(--txt-2);margin-bottom:0.75rem;">'
                    f'🔎 <strong style="color:var(--txt);">{len(display_chunks)}</strong> passages above relevance threshold</p>',
                    unsafe_allow_html=True,
                )
                for idx, chunk in enumerate(display_chunks):
                    _render_passage_card(chunk, idx, query=query)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Paper Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "f4c4 Paper Explorer":

    # ── Detail view (shown at top when a paper is selected) ─────────────────
    if "selected_paper" in st.session_state:
        paper  = st.session_state["selected_paper"]
        pmc_id = paper.get("pmc_id", "")

        # —— Header block ————————————————————————————————————
        authors_str = _authors_display(paper.get("authors", ""), max_chars=180)
        doi_html    = _doi_link(paper.get("doi", ""))
        st.markdown(
            f'<div class="detail-header">'
            f'  <div class="detail-title">{paper.get("title", "Untitled")}</div>'
            f'  <div class="detail-authors">✍️ {authors_str}</div>'
            f'  <div class="detail-meta-row">'
            f'    <span>📅 {paper.get("year", "—")}</span>'
            f'    <span>🏥 {paper.get("journal", "—")}</span>'
            f'    <span class="badge badge-pmc">{pmc_id}</span>'
            f'    {doi_html}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        btn_col1, btn_col2, _ = st.columns([1.4, 1, 2.5])
        with btn_col1:
            summarize_clicked = st.button(
                "🧠 Summarize full paper",
                key="deep_summary_btn",
                use_container_width=True,
                type="primary",
            )
        with btn_col2:
            if st.button("← Back to list", key="close_paper", use_container_width=True):
                del st.session_state["selected_paper"]
                st.session_state.pop("_deep_summary_done", None)
                st.session_state.pop("_trigger_summary", None)
                st.rerun()

        # —— LLM summary (streamed on demand) ——————————————————————
        if summarize_clicked:
            st.session_state["_trigger_summary"] = pmc_id
            st.session_state["_deep_summary_done"] = False

        if (
            st.session_state.get("_trigger_summary") == pmc_id
            and not st.session_state.get("_deep_summary_done")
        ):
            with st.expander("🧠 AI Structured Summary", expanded=True):
                with st.spinner("Reading all paper sections…"):
                    st.write_stream(generate_deep_summary(pmc_id))
            st.session_state["_deep_summary_done"] = True
        elif st.session_state.get("_deep_summary_done") and st.session_state.get("_trigger_summary") == pmc_id:
            with st.expander("🧠 AI Structured Summary", expanded=True):
                st.caption("Summary generated above. Click \"Summarize full paper\" again to regenerate.")

        # —— Abstract only (stripped of metadata header) ————————————————
        raw_summary_doc = paper.get("summary_text", "")
        abstract_text   = _extract_abstract_from_summary_doc(raw_summary_doc)
        if abstract_text:
            with st.expander("📌 Abstract", expanded=True):
                st.markdown(
                    f'<div class="section-body">{abstract_text}</div>',
                    unsafe_allow_html=True,
                )

        # —— Full paper sections (strip [SectionName]\n\n prefix) —————————
        with st.expander("📑 Full paper sections", expanded=False):
            retriever  = get_retriever()
            all_chunks = retriever.retrieve_by_paper(pmc_id)
            content_chunks = [
                c for c in all_chunks
                if not c.is_summary
            ]
            if not content_chunks:
                st.caption("No section text available for this paper.")
            else:
                for c in content_chunks:
                    clean_text = _strip_section_prefix(c.text, c.section)
                    st.markdown(
                        f'<div style="margin-bottom:0.5rem;">'
                        f'  <span class="badge badge-section">{c.section}</span>'
                        f'</div>'
                        f'<div class="section-body">{clean_text}</div>'
                        f'<hr/>',
                        unsafe_allow_html=True,
                    )

        # —— References ————————————————————————————————————————
        with st.expander("📚 References", expanded=False):
            retriever  = get_retriever()
            all_chunks = retriever.retrieve_by_paper(pmc_id)
            refs_seen: set = set()
            refs: list[dict] = []
            for c in all_chunks:
                for r in c.references:
                    # r.get("authors") may be list or str — normalise to str for key
                    raw_authors = r.get("authors", "")
                    if isinstance(raw_authors, list):
                        authors_key: str = ", ".join(str(a) for a in raw_authors)
                    else:
                        authors_key = str(raw_authors)
                    key = (
                        r.get("title", ""),
                        r.get("doi", ""),
                        r.get("pmid", ""),
                        r.get("pmcid", ""),
                    )
                    if key in refs_seen:
                        continue
                    refs_seen.add(key)
                    refs.append({**r, "_authors_str": authors_key})

            if not refs:
                st.caption("No structured references stored for this paper.")
            else:
                st.markdown(
                    f'<p style="font-size:0.78rem;color:var(--txt-3);margin-bottom:0.75rem;">{len(refs)} references</p>',
                    unsafe_allow_html=True,
                )
                rows_html = ""
                for i, r in enumerate(refs, 1):
                    title   = r.get("title", "Untitled")
                    authors = r["_authors_str"]
                    journal = r.get("journal", "")
                    year    = r.get("year", "")
                    doi     = r.get("doi", "")

                    # Build author/meta line — all guaranteed strings now
                    meta_parts = []
                    if authors:
                        a_short = authors[:100] + (" et al." if len(authors) > 100 else "")
                        meta_parts.append(a_short)
                    if year:
                        meta_parts.append(f"({year})")
                    if journal:
                        meta_parts.append(f"<em>{journal[:60]}</em>")
                    if doi:
                        meta_parts.append(f'<a class="ref-doi" href="https://doi.org/{doi}" target="_blank">🔗 DOI</a>')

                    meta_line = " ".join(meta_parts)
                    rows_html += (
                        f'<div class="ref-row">'
                        f'  <span style="color:var(--txt-3);margin-right:6px;">{i}.</span>'
                        f'  <span class="ref-title">{title}</span>'
                        f'  <div style="margin-top:3px;">{meta_line}</div>'
                        f'</div>'
                    )
                st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown('<hr/>', unsafe_allow_html=True)

    # ── Paper grid (always shown below detail or alone) ───────────────────────
    st.markdown('<div class="page-title">📄 Paper Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Browse the full corpus. '
        'Click any paper to read its full text, abstract, and references.</div>',
        unsafe_allow_html=True,
    )

    # Search bar
    search_col, _ = st.columns([3, 2])
    with search_col:
        search_q = st.text_input(
            "Search papers",
            placeholder="Title, author, or PMC ID…",
            label_visibility="collapsed",
            key="explorer_search",
        )

    try:
        if search_q:
            papers = search_papers_by_title(search_q)
            st.markdown(
                f'<p style="font-size:0.78rem;color:var(--txt-2);margin-bottom:0.75rem;">'
                f'{len(papers)} result{"s" if len(papers) != 1 else ""} for \u201c{search_q}\u201d</p>',
                unsafe_allow_html=True,
            )
        else:
            papers = get_all_papers()
            st.markdown(
                f'<p style="font-size:0.78rem;color:var(--txt-2);margin-bottom:0.75rem;">'
                f'Showing all {len(papers)} papers</p>',
                unsafe_allow_html=True,
            )
    except Exception as exc:
        st.error(f"Could not load papers: {exc}")
        papers = []

    # Render grid (3 columns)
    COLS = 3
    for row_start in range(0, len(papers), COLS):
        row_papers = papers[row_start : row_start + COLS]
        cols = st.columns(COLS)
        for col, paper in zip(cols, row_papers):
            with col:
                pmc_id = paper.get("pmc_id", "")
                st.markdown(_render_paper_grid_card(paper), unsafe_allow_html=True)
                if st.button(
                    "View details",
                    key=f"view_{pmc_id}_{row_start}",
                    use_container_width=True,
                ):
                    st.session_state["selected_paper"] = paper
                    st.session_state.pop("_deep_summary_done", None)
                    st.session_state.pop("_trigger_summary", None)
                    st.rerun()
