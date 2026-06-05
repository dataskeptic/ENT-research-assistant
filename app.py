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
    page_icon="\U0001fa7a",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:    #0a0f1c;
    --bg-card:       rgba(17, 24, 39, 0.70);
    --bg-card-hover: rgba(25, 34, 56, 0.85);
    --accent:        #3b82f6;
    --accent-dim:    rgba(59,130,246,0.12);
    --accent-cyan:   #06b6d4;
    --accent-green:  #10b981;
    --accent-amber:  #f59e0b;
    --accent-rose:   #f43f5e;
    --txt:           #e2e8f0;
    --txt-2:         #94a3b8;
    --txt-3:         #64748b;
    --border:        rgba(255,255,255,0.06);
    --radius:        14px;
    --tr:            0.22s cubic-bezier(.4,0,.2,1);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* Main background */
.main .block-container { background: var(--bg-primary) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #0a0f1c 100%) !important;
    border-right: 1px solid var(--border);
}

/* Glass card */
.glass {
    background: var(--bg-card);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    transition: border-color var(--tr), box-shadow var(--tr), transform var(--tr);
}
.glass:hover {
    border-color: rgba(59,130,246,0.2);
    box-shadow: 0 6px 28px rgba(59,130,246,0.08);
    transform: translateY(-1px);
}

/* Hero */
.hero {
    padding: 1.75rem 0 1rem;
    margin-bottom: 0.25rem;
}
.hero h1 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 55%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}
.hero p {
    font-size: 0.88rem;
    color: var(--txt-2);
    max-width: 580px;
    line-height: 1.6;
}

/* Stat pill */
.stat-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 10px;
    margin-bottom: 7px;
}
.stat-n {
    font-size: 1.2rem;
    font-weight: 700;
    color: #60a5fa;
    min-width: 40px;
    line-height: 1;
}
.stat-l {
    font-size: 0.7rem;
    color: var(--txt-3);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 99px;
    font-size: 0.71rem;
    font-weight: 600;
    white-space: nowrap;
}
.b-high { background: rgba(16,185,129,0.14); color: #34d399; }
.b-mid  { background: rgba(245,158,11,0.14);  color: #fbbf24; }
.b-low  { background: rgba(244,63,94,0.14);   color: #fb7185; }
.b-sec  {
    background: rgba(6,182,212,0.1);
    color: #22d3ee;
    border-radius: 5px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-size: 0.67rem;
    padding: 2px 7px;
}
.b-pmc  { background: rgba(59,130,246,0.1); color: #60a5fa; }

/* Score bar */
.strack {
    background: rgba(255,255,255,0.05);
    border-radius: 99px;
    height: 4px;
    overflow: hidden;
    margin-top: 7px;
}
.sfill { height: 100%; border-radius: 99px; }

/* Passage card */
.pcard {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color var(--tr), box-shadow var(--tr);
}
.pcard:hover { border-color: rgba(59,130,246,0.2); box-shadow: 0 4px 20px rgba(59,130,246,0.07); }
.pcard-meta { font-size: 0.73rem; color: var(--txt-3); margin-top: 8px; display: flex; flex-wrap: wrap; gap: 10px; }

/* Paper grid card */
.pgcard {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    height: 100%;
    transition: border-color var(--tr), transform var(--tr);
}
.pgcard:hover { border-color: rgba(59,130,246,0.22); transform: translateY(-2px); box-shadow: 0 8px 28px rgba(59,130,246,0.1); }
.pgcard h4 {
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--txt);
    line-height: 1.4;
    margin-bottom: 6px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.pgcard-m { font-size: 0.73rem; color: var(--txt-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }

/* Paper detail */
.detail-hdr {
    background: linear-gradient(135deg, rgba(59,130,246,0.07) 0%, rgba(6,182,212,0.05) 100%);
    border: 1px solid rgba(59,130,246,0.14);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.1rem;
}

/* Answer container */
.answer-box {
    background: rgba(17,24,39,0.6);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 0.9rem;
    line-height: 1.8;
    color: var(--txt);
}

/* Citation strip */
.cite-strip {
    display: flex; flex-wrap: wrap; gap: 6px;
    padding: 0.6rem 1rem;
    background: rgba(17,24,39,0.5);
    border: 1px solid var(--border);
    border-radius: 9px;
    margin-top: 0.7rem;
}
.cite-chip {
    font-size: 0.71rem;
    color: #60a5fa;
    background: rgba(59,130,246,0.1);
    padding: 2px 10px;
    border-radius: 99px;
}

/* Section / reference body */
.sec-body {
    font-size: 0.87rem;
    line-height: 1.75;
    color: var(--txt-2);
    white-space: pre-wrap;
}
.ref-row {
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.81rem;
    line-height: 1.55;
    color: var(--txt-2);
}
.ref-row:last-child { border-bottom: none; }
.ref-t { font-weight: 600; color: var(--txt); }
.ref-doi { color: var(--accent); text-decoration: none; font-size: 0.76rem; }
.ref-doi:hover { text-decoration: underline; }

/* Streamlit widget overrides */
.stTextInput > div > div > input {
    background: rgba(17,24,39,0.65) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 10px !important;
    color: var(--txt) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.14) !important;
}
div.stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(59,130,246,0.22) !important;
    background: rgba(17,24,39,0.5) !important;
    color: var(--txt) !important;
    font-weight: 500 !important;
    transition: all var(--tr) !important;
}
div.stButton > button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 14px rgba(59,130,246,0.18) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
}
details[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    background: rgba(17,24,39,0.45) !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: rgba(17,24,39,0.4);
    border-radius: 11px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500;
    color: var(--txt-2) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.16) !important;
    color: #60a5fa !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stSelectbox > div > div {
    background: rgba(17,24,39,0.6) !important;
    border: 1px solid rgba(59,130,246,0.18) !important;
    border-radius: 10px !important;
}
hr { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }
mark, strong { color: #93c5fd !important; }

@keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.fi { animation: fadeUp 0.35s ease-out both; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Navigation mode labels — plain ASCII keys to avoid any encoding issues
MODE_ASK      = "Ask the Literature"
MODE_EXPLORER = "Paper Explorer"


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
    return f'<a class="ref-doi" href="https://doi.org/{doi}" target="_blank">DOI \u2197</a>' if doi else ""


def _strip_prefix(text: str, section: str) -> str:
    """Remove the [SectionName]\n\n prefix that ingestion prepends."""
    prefix = f"[{section}]\n\n"
    if text.startswith(prefix):
        return text[len(prefix):]
    return re.sub(r'^\[[^\]]*\]\n\n', '', text, count=1)


def _abstract_only(raw_doc: str) -> str:
    """Extract only the abstract text from a stored summary doc."""
    marker = "Abstract:\n"
    idx = raw_doc.find(marker)
    return raw_doc[idx + len(marker):].strip() if idx != -1 else raw_doc.strip()


def _authors_str(raw: object, limit: int = 160) -> str:
    """Coerce authors (str or list) to a display string."""
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
    preview = highlight_terms(raw[:560], query) if query else raw[:560]
    doi_html = _doi(chunk.metadata.get("doi", ""))

    st.markdown(
        f'<div class="pcard fi" style="animation-delay:{idx*0.04}s;">'
        f'  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">'
        f'    <div style="min-width:0;flex:1;">'
        f'      <span class="badge b-sec">{section}</span>'
        f'      <span style="font-size:0.84rem;font-weight:600;color:var(--txt);margin-left:6px;">{chunk.citation_label}</span>'
        f'      <span style="font-size:0.76rem;color:var(--txt-3);margin-left:4px;">\u00b7 {chunk.metadata.get("title","")[:90]}</span>'
        f'    </div>'
        f'    {_badge(pct)}'
        f'  </div>'
        f'  {_bar(pct)}'
        f'  <div style="font-size:0.84rem;line-height:1.65;color:var(--txt-2);margin-top:9px;">{preview}</div>'
        f'  <div class="pcard-meta">'
        f'    <span>\U0001f4c5 {chunk.metadata.get("year","\u2014")}</span>'
        f'    <span>\U0001f3e5 {chunk.metadata.get("journal","\u2014")[:55]}</span>'
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
    year    = paper.get("year", "\u2014")
    journal = (paper.get("journal") or "")[:55]
    author  = _authors_str(paper.get("authors", ""), limit=60)
    pmc_id  = paper.get("pmc_id", "")
    title   = (paper.get("title") or "Untitled")[:140]
    return (
        f'<div class="pgcard">'
        f'  <h4>{title}</h4>'
        f'  <div class="pgcard-m">\U0001f4c5 {year} \u00b7 {journal}</div>'
        f'  <div class="pgcard-m">\u270d\ufe0f {author}</div>'
        f'  <div style="margin-top:7px;"><span class="badge b-pmc">{pmc_id}</span></div>'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:0.5rem 0 1.25rem;">'
        '<div style="font-size:2rem;margin-bottom:0.4rem;">\U0001fa7a</div>'
        '<div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.01em;">ENT Research</div>'
        '<div style="font-size:0.73rem;color:#64748b;margin-top:1px;">Research Assistant</div>'
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
        st.caption("\u26a0\ufe0f Could not load stats.")
        stats = {}

    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.68rem;color:#64748b;line-height:1.7;">'
        'ChromaDB \u00b7 OpenRouter<br/>Nemotron Ultra 253B'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MODE 1 \u2014 Ask the Literature
# ──────────────────────────────────────────────────────────────────────────────
if mode == MODE_ASK:
    st.markdown(
        '<div class="hero">'
        '<h1>\U0001f52c Ask the Literature</h1>'
        '<p>Ask any clinical or research question and get a citation-backed answer '
        'synthesised from peer-reviewed ENT papers. '
        'Switch to \u201cRanked Passages\u201d to browse the raw semantic results.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    q_col, j_col = st.columns([5, 1])
    with q_col:
        query = st.text_input(
            "Question",
            placeholder="e.g. What are the outcomes after TORS for oropharyngeal cancer?",
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
        tab_ans, tab_passages = st.tabs(["\U0001f4a1  AI Answer", "\U0001f3af  Ranked Passages"])

        with tab_ans:
            with st.spinner("Retrieving passages and generating answer\u2026"):
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
                extra = "<span class='cite-chip'>\u2026</span>" if len(unique) > 10 else ""
                st.markdown(
                    f'<div class="cite-strip">'
                    f'<span style="font-size:0.7rem;color:var(--txt-3);margin-right:5px;">Sources:</span>'
                    f'{chips}{extra}</div>',
                    unsafe_allow_html=True,
                )

        with tab_passages:
            sources = st.session_state.get("_last_sources", [])
            display = [c for c in sources if not c.is_summary]
            if not display:
                st.info("\u2139\ufe0f Submit a question to see ranked passages here.")
            else:
                st.markdown(
                    f'<p style="font-size:0.78rem;color:var(--txt-2);margin-bottom:0.75rem;">'
                    f'\U0001f50e <strong style="color:var(--txt);">{len(display)}</strong> '
                    f'passages above relevance threshold</p>',
                    unsafe_allow_html=True,
                )
                for i, chunk in enumerate(display):
                    _passage_card(chunk, i, query=query)


# ──────────────────────────────────────────────────────────────────────────────
# MODE 2 \u2014 Paper Explorer
# ──────────────────────────────────────────────────────────────────────────────
elif mode == MODE_EXPLORER:

    # Detail panel (shown at top when a paper is selected)
    if "selected_paper" in st.session_state:
        paper  = st.session_state["selected_paper"]
        pmc_id = paper.get("pmc_id", "")

        authors_disp = _authors_str(paper.get("authors", ""), limit=200)
        doi_html     = _doi(paper.get("doi", ""))

        st.markdown(
            f'<div class="detail-hdr">'
            f'  <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;line-height:1.3;margin-bottom:8px;">'
            f'  {paper.get("title", "Untitled")}'
            f'  </div>'
            f'  <div style="font-size:0.82rem;color:var(--txt-2);margin-bottom:6px;">'
            f'  \u270d\ufe0f {authors_disp}</div>'
            f'  <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;font-size:0.78rem;color:var(--txt-2);">'
            f'    <span>\U0001f4c5 {paper.get("year","\u2014")}</span>'
            f'    <span>\U0001f3e5 {paper.get("journal","\u2014")}</span>'
            f'    <span class="badge b-pmc">{pmc_id}</span>'
            f'    {doi_html}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        b1, b2, _ = st.columns([1.4, 1, 2.5])
        with b1:
            do_summary = st.button(
                "\U0001f9e0  Summarize full paper",
                key="sum_btn",
                use_container_width=True,
                type="primary",
            )
        with b2:
            if st.button("\u2190  Back to list", key="back_btn", use_container_width=True):
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
            with st.expander("\U0001f9e0  AI Structured Summary", expanded=True):
                with st.spinner("Reading all sections\u2026"):
                    st.write_stream(generate_deep_summary(pmc_id))
            st.session_state["_sum_done"] = True

        # Abstract (metadata-free)
        abstract = _abstract_only(paper.get("summary_text", ""))
        if abstract:
            with st.expander("\U0001f4cc  Abstract", expanded=True):
                st.markdown(f'<div class="sec-body">{abstract}</div>', unsafe_allow_html=True)

        # Full sections (prefix stripped)
        with st.expander("\U0001f4d1  Full paper sections", expanded=False):
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
        with st.expander("\U0001f4da  References", expanded=False):
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
                    f'<p style="font-size:0.76rem;color:var(--txt-3);margin-bottom:0.7rem;">{len(refs)} references</p>',
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
                    if r.get("doi"): parts.append(f'<a class="ref-doi" href="https://doi.org/{r["doi"]}" target="_blank">DOI \u2197</a>')
                    rows += (
                        f'<div class="ref-row">'
                        f'  <span style="color:var(--txt-3);margin-right:5px;">{i}.</span>'
                        f'  <span class="ref-t">{r.get("title","Untitled")}</span>'
                        f'  <div style="margin-top:3px;">{" ".join(parts)}</div>'
                        f'</div>'
                    )
                st.markdown(rows, unsafe_allow_html=True)

        st.markdown('<hr/>', unsafe_allow_html=True)

    # Paper grid
    st.markdown(
        '<div class="hero">'
        '<h1>\U0001f4c4 Paper Explorer</h1>'
        '<p>Browse the full corpus. Click any paper to view its full text, abstract, and references.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    sc, _ = st.columns([3, 2])
    with sc:
        search_q = st.text_input(
            "Search",
            placeholder="Title, author, or PMC ID\u2026",
            label_visibility="collapsed",
            key="exp_search",
        )

    try:
        papers = search_papers_by_title(search_q) if search_q else get_all_papers()
        n = len(papers)
        label = (
            f'{n} result{"s" if n != 1 else ""} for \u201c{search_q}\u201d'
            if search_q else f'Showing all {n} papers'
        )
        st.markdown(f'<p style="font-size:0.77rem;color:var(--txt-2);margin-bottom:0.75rem;">{label}</p>',
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
