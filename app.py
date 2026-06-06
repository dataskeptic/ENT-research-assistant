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
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Palette: #01161E / #124559 / #598392 / #AEC3B0 / #EFF6E0 ── */
:root {
    --c-deepest:  #01161E;
    --c-navy:     #124559;
    --c-teal:     #598392;
    --c-sage:     #AEC3B0;
    --c-cream:    #EFF6E0;

    /* Semantic roles */
    --bg:            #01161E;
    --bg-mid:        #0b2535;
    --bg-card:       rgba(18, 69, 89, 0.28);
    --bg-card-hover: rgba(18, 69, 89, 0.48);
    --bg-input:      rgba(18, 69, 89, 0.35);

    --accent:        #598392;
    --accent-bright: #AEC3B0;
    --accent-dim:    rgba(89, 131, 146, 0.18);
    --accent-glow:   rgba(89, 131, 146, 0.22);

    --txt:           #EFF6E0;
    --txt-2:         rgba(239, 246, 224, 0.72);
    --txt-3:         rgba(239, 246, 224, 0.42);

    --border:        rgba(89, 131, 146, 0.22);
    --border-hover:  rgba(174, 195, 176, 0.45);

    --radius:        12px;
    --radius-sm:     8px;
    --radius-lg:     16px;
    --tr:            0.22s cubic-bezier(.4,0,.2,1);

    --shadow-sm:     0 2px 8px rgba(1, 22, 30, 0.45);
    --shadow-md:     0 6px 20px rgba(1, 22, 30, 0.55);
    --shadow-lg:     0 16px 40px rgba(1, 22, 30, 0.65);
    --shadow-glow:   0 0 24px rgba(89, 131, 146, 0.18);

    /* Constrain sidebar resize: prevent it from going beyond its natural width */
    --sidebar-max: 320px;
}

/* ── Universal font reset ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
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

/* ── Sidebar — constrained, no bleeding expansion ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b2535 0%, #01161E 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 32px rgba(1, 22, 30, 0.7) !important;
    max-width: var(--sidebar-max) !important;
    min-width: 220px !important;
}

/* Constrain the resize handle so it can't drag beyond max width */
[data-testid="stSidebar"] > div:first-child {
    max-width: var(--sidebar-max) !important;
}

/* ── Markdown / text colour ── */
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p {
    color: var(--txt) !important;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 1.1rem !important;
    color: var(--txt) !important;
    caret-color: var(--accent-bright);
    font-size: 0.92rem !important;
    box-shadow: inset 0 1px 4px rgba(1, 22, 30, 0.3);
    transition: border-color var(--tr), box-shadow var(--tr);
}
.stTextInput > div > div > input::placeholder {
    color: var(--txt-3) !important;
    font-weight: 300;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: inset 0 1px 4px rgba(1, 22, 30, 0.3), 0 0 0 3px var(--accent-dim) !important;
    outline: none;
}

/* ── Buttons ── */
div.stButton > button {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--txt-2) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
    box-shadow: var(--shadow-sm) !important;
    transition: all var(--tr) !important;
}
div.stButton > button:hover {
    border-color: var(--border-hover) !important;
    background: var(--bg-card-hover) !important;
    color: var(--txt) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md), var(--shadow-glow) !important;
}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--c-navy) 0%, var(--c-teal) 100%) !important;
    border: 1px solid var(--accent) !important;
    color: var(--c-cream) !important;
    font-weight: 600 !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, var(--c-teal) 0%, var(--c-sage) 100%) !important;
    border-color: var(--accent-bright) !important;
    color: var(--c-deepest) !important;
    box-shadow: var(--shadow-md), var(--shadow-glow) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div,
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--txt) !important;
}

/* ── Expander ── */
details[data-testid="stExpander"],
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm);
    transition: border-color var(--tr);
}
details[data-testid="stExpander"]:hover {
    border-color: var(--border-hover) !important;
}
[data-testid="stExpander"] summary {
    color: var(--txt) !important;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.015em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    padding: 0;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0 !important;
    font-weight: 500;
    font-size: 0.875rem;
    color: var(--txt-3) !important;
    background: transparent !important;
    padding: 10px 18px;
    transition: color var(--tr);
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-bright) !important;
    border-bottom: 2px solid var(--accent-bright) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.5rem;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Alert / info boxes ── */
[data-testid="stAlert"], .stAlert {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: var(--radius) !important;
    color: var(--txt) !important;
    box-shadow: var(--shadow-sm);
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.25rem 0;
    opacity: 0.6;
}

/* ── Highlight mark ── */
mark {
    background: var(--accent-dim);
    color: var(--accent-bright) !important;
    border-radius: 3px;
    padding: 1px 4px;
}
strong { color: var(--c-cream) !important; font-weight: 600; }

/* ── Sidebar branding ── */
.ent-logo {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.ent-logo img {
    width: 96px;
    border-radius: var(--radius-sm);
    box-shadow: var(--shadow-md);
}

/* ── Hero heading — uses Instrument Serif for elegance ── */
.hero { padding: 2rem 0 1.5rem; }
.hero h1 {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: clamp(2rem, 4vw, 2.8rem);
    font-weight: 400;
    font-style: italic;
    color: var(--c-cream);
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin-bottom: 0.75rem;
}
.hero p {
    font-size: 0.95rem;
    color: var(--txt-2);
    line-height: 1.65;
    max-width: 680px;
    font-weight: 300;
}

/* ── Stat pill ── */
.stat-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin-bottom: 8px;
    box-shadow: var(--shadow-sm);
    transition: all var(--tr);
}
.stat-pill:hover {
    border-color: var(--border-hover);
    background: var(--bg-card-hover);
}
.stat-n {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--accent-bright);
    min-width: 36px;
    font-variant-numeric: tabular-nums;
}
.stat-l {
    font-size: 0.7rem;
    color: var(--txt-3);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}

/* ── Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 600;
    white-space: nowrap;
    letter-spacing: 0.02em;
}
.b-high { background: rgba(174, 195, 176, 0.15); color: var(--accent-bright); border: 1px solid rgba(174, 195, 176, 0.35); }
.b-mid  { background: rgba(239, 246, 224, 0.08); color: var(--txt-2); border: 1px solid var(--border); }
.b-low  { background: rgba(239, 246, 224, 0.04); color: var(--txt-3); border: 1px solid rgba(239, 246, 224, 0.08); }
.b-sec  {
    background: rgba(89, 131, 146, 0.15);
    color: var(--accent-bright);
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-size: 0.65rem;
    padding: 3px 8px;
    border: 1px solid var(--border);
}
.b-doi  { background: rgba(239, 246, 224, 0.04); color: var(--txt-3); border: 1px solid var(--border); font-family: 'Inter', monospace; font-size: 0.68rem; }

/* ── Score bar ── */
.strack  {
    background: rgba(239, 246, 224, 0.06);
    border-radius: 99px;
    height: 3px;
    overflow: hidden;
    margin-top: 10px;
}
.sfill   {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--c-teal), var(--c-sage));
}

/* ── Passage card ── */
.pcard {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    backdrop-filter: blur(8px);
    transition: border-color var(--tr), box-shadow var(--tr), transform var(--tr);
}
.pcard:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-md), var(--shadow-glow);
    transform: translateY(-2px);
}
.pcard-meta {
    font-size: 0.76rem;
    color: var(--txt-3);
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-weight: 400;
}

/* ── Paper grid card ── */
.pgcard {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    height: 100%;
    box-shadow: var(--shadow-sm);
    backdrop-filter: blur(8px);
    transition: border-color var(--tr), box-shadow var(--tr), transform var(--tr);
}
.pgcard:hover {
    border-color: var(--border-hover);
    transform: translateY(-3px);
    box-shadow: var(--shadow-md), var(--shadow-glow);
}
.pgcard h4 {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1rem;
    font-weight: 400;
    color: var(--c-cream);
    line-height: 1.5;
    margin-bottom: 8px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.pgcard-m {
    font-size: 0.76rem;
    color: var(--txt-3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 4px;
}

/* ── Paper detail header ── */
.detail-hdr {
    background: linear-gradient(135deg, rgba(18, 69, 89, 0.5) 0%, rgba(1, 22, 30, 0.7) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-lg);
}

/* ── Answer box ── */
.answer-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-bright);
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem;
    font-size: 0.95rem;
    line-height: 1.8;
    color: var(--txt);
    box-shadow: var(--shadow-sm);
}

/* ── Citation strip ── */
.cite-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 0.8rem 1rem;
    background: rgba(1, 22, 30, 0.4);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin-top: 0.75rem;
}
.cite-chip {
    font-size: 0.72rem;
    color: var(--txt-2);
    font-weight: 500;
    background: rgba(239, 246, 224, 0.06);
    padding: 3px 10px;
    border-radius: 99px;
    border: 1px solid var(--border);
    transition: all var(--tr);
    cursor: default;
}
.cite-chip:hover {
    background: var(--c-navy);
    color: var(--c-cream);
    border-color: var(--accent);
}

/* ── Section / reference text ── */
.sec-body {
    font-size: 0.9rem;
    line-height: 1.75;
    color: var(--txt-2);
    white-space: pre-wrap;
}
.ref-row {
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
    line-height: 1.6;
    color: var(--txt-2);
}
.ref-row:last-child { border-bottom: none; }
.ref-t   { font-weight: 500; color: var(--txt); }
.ref-doi { color: var(--accent-bright); text-decoration: none; font-size: 0.78rem; font-weight: 500; }
.ref-doi:hover { text-decoration: underline; color: var(--c-sage); }

/* ── Fade-up animation ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fi { animation: fadeUp 0.45s cubic-bezier(0.2, 0.8, 0.2, 1) both; }

/* ── Sidebar resize constraint via JS injected style ── */
/* The following rule limits the sidebar drag handle so it never expands
   the sidebar beyond its natural/collapsed width (~320px) */
[data-testid="stSidebarResizeHandle"] {
    /* Restrict visual feedback if handle goes too far */
    pointer-events: auto;
}
</style>

<script>
// Enforce sidebar max-width after any resize interaction
(function() {
    const MAX_WIDTH = 320;
    function clampSidebar() {
        const sb = document.querySelector('[data-testid="stSidebar"]');
        if (sb) {
            const w = sb.getBoundingClientRect().width;
            if (w > MAX_WIDTH) {
                sb.style.setProperty('width', MAX_WIDTH + 'px', 'important');
                sb.style.setProperty('min-width', MAX_WIDTH + 'px', 'important');
                sb.style.setProperty('max-width', MAX_WIDTH + 'px', 'important');
            }
        }
    }
    // MutationObserver to catch Streamlit dynamic resizes
    const obs = new MutationObserver(clampSidebar);
    document.addEventListener('DOMContentLoaded', function() {
        const sb = document.querySelector('[data-testid="stSidebar"]');
        if (sb) obs.observe(sb, { attributes: true, attributeFilter: ['style'] });
    });
    // Also listen on mouseup in case of drag
    document.addEventListener('mouseup', clampSidebar);
})();
</script>
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
    return f'<div class="strack"><div class="sfill" style="width:{pct}%;"></div></div>'


def _doi(doi: str) -> str:
    if not doi:
        return ""
    return f'<a class="ref-doi" href="https://doi.org/{doi}" target="_blank">DOI ↗</a>'


def _strip_prefix(text: str, section: str) -> str:
    """Remove leading '[SECTION]\n\n' prefix that the ingestion pipeline prepends."""
    prefix = f"[{section}]\n\n"
    if text.startswith(prefix):
        return text[len(prefix):]
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
    preview = highlight_terms(raw[:520], query) if query else raw[:520]
    doi_html = _doi(chunk.metadata.get("doi", ""))

    st.markdown(
        f'<div class="pcard fi" style="animation-delay:{idx*0.04}s;">'
        f'  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">'
        f'    <div style="min-width:0;flex:1;">'
        f'      <span class="badge b-sec">{section}</span>'
        f'      <span style="font-size:0.82rem;font-weight:600;color:var(--txt);margin-left:6px;">{chunk.citation_label}</span>'
        f'      <span style="font-size:0.74rem;color:var(--txt-3);margin-left:4px;">· {chunk.metadata.get("title","")[:85]}</span>'
        f'    </div>'
        f'    {_badge(pct)}'
        f'  </div>'
        f'  {_bar(pct)}'
        f'  <div style="font-size:0.82rem;line-height:1.7;color:var(--txt-2);margin-top:10px;">{preview}</div>'
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
        f'  <div style="margin-top:8px;"><span class="badge b-doi">{doi}</span></div>'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("image/logo.png", use_container_width=False, width=88)
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem; text-align: left;">
          <div style="font-family: 'Instrument Serif', Georgia, serif;
                      font-size: 1.4rem; font-weight: 400; font-style: italic;
                      color: #EFF6E0; letter-spacing: 0.01em;">
            ENT AI
          </div>
          <div style="font-size: 0.65rem; color: #598392; margin-top: 4px;
                      font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;">
            Latest Papers · Past Month
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
        '<div style="font-size:0.66rem;color:var(--txt-3);line-height:1.8;">'
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
        'Switch to <strong style="color:var(--accent-bright);">Ranked Passages</strong> to browse the raw semantic results.</p>'
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
                full_answer = answer_box.write_stream(
                    stream_answer(query, where=where)
                )

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
                    f'<span style="font-size:0.67rem;color:var(--txt-3);margin-right:6px;">Sources:</span>'
                    f'{chips}{extra}</div>',
                    unsafe_allow_html=True,
                )

        with tab_passages:
            sources = st.session_state.get("_last_sources", [])
            display = [c for c in sources if not c.is_summary]
            if not display:
                st.info("Submit a question first to see ranked passages here.")
            else:
                st.markdown(
                    f'<p style="font-size:0.75rem;color:var(--txt-2);margin-bottom:0.75rem;">'
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

    if "selected_paper" in st.session_state:
        paper  = st.session_state["selected_paper"]
        doi_id = paper.get("doi", "")

        authors_disp = _authors_str(paper.get("authors", ""), limit=200)
        doi_html     = _doi(paper.get("doi", ""))

        st.markdown(
            f'<div class="detail-hdr">'
            f'  <div style="font-family:\'Instrument Serif\', Georgia, serif; font-size:1.25rem; font-weight:400; color:var(--c-cream); line-height:1.4; margin-bottom:8px;">'
            f'    {paper.get("title", "Untitled")}'
            f'  </div>'
            f'  <div style="font-size:0.83rem;color:var(--txt-2);margin-bottom:10px;">{authors_disp}</div>'
            f'  <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:0.78rem;color:var(--txt-3);">'
            f'    <span>{paper.get("year","—")}</span>'
            f'    <span>{paper.get("journal","—")}</span>'
            f'    <span class="badge b-doi">{doi_id}</span>'
            f'    {doi_html}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

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

        abstract = _abstract_only(paper.get("summary_text", ""))
        if abstract:
            with st.expander("📋 Abstract", expanded=True):
                st.markdown(
                    f'<div class="sec-body">{abstract}</div>',
                    unsafe_allow_html=True,
                )

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
                    f'<p style="font-size:0.73rem;color:var(--txt-3);margin-bottom:0.5rem;">'
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
                f'<p style="font-size:0.74rem;color:var(--txt-2);margin-bottom:0.75rem;">{label}</p>',
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
