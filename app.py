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
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:    #0a0f1c;
    --bg-card:       rgba(17, 24, 39, 0.70);
    --bg-card-hover: rgba(25, 34, 56, 0.85);
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --accent-emerald:#10b981;
    --accent-amber:  #f59e0b;
    --accent-rose:   #f43f5e;
    --text-primary:  #e2e8f0;
    --text-muted:    #94a3b8;
    --border-glass:  rgba(255,255,255,0.06);
    --blur-glass:    12px;
    --radius:        14px;
    --transition:    0.25s cubic-bezier(.4,0,.2,1);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #0a0f1c 100%);
    border-right: 1px solid var(--border-glass);
}

/* —— Main content background —— */
.main .block-container {
    background: #0a0f1c;
    padding-top: 1.5rem;
}

.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(var(--blur-glass));
    -webkit-backdrop-filter: blur(var(--blur-glass));
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: transform var(--transition), box-shadow var(--transition), border-color var(--transition);
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(59,130,246,0.10);
    border-color: rgba(59,130,246,0.18);
}

.score-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.score-high   { background: rgba(16,185,129,0.15); color: #34d399; }
.score-mid    { background: rgba(245,158,11,0.15);  color: #fbbf24; }
.score-low    { background: rgba(244,63,94,0.15);   color: #fb7185; }

.section-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: rgba(6,182,212,0.12);
    color: #22d3ee;
    margin-right: 8px;
}

.doi-link {
    color: var(--accent-blue);
    text-decoration: none;
    font-size: 0.82rem;
    transition: color var(--transition);
}
.doi-link:hover { color: var(--accent-cyan); text-decoration: underline; }

.stat-card {
    text-align: center;
    padding: 0.9rem 0.6rem;
    border-radius: 12px;
    background: rgba(59,130,246,0.06);
    border: 1px solid rgba(59,130,246,0.10);
}
.stat-number {
    font-size: 1.65rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

.hero-header {
    text-align: center;
    padding: 2rem 1rem 1.25rem;
}
.hero-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
    line-height: 1.25;
}
.hero-header p {
    color: var(--text-muted);
    font-size: 0.93rem;
    max-width: 640px;
    margin: 0 auto;
}

mark, .highlight-term {
    background: rgba(59,130,246,0.22);
    color: #93c5fd;
    padding: 1px 3px;
    border-radius: 3px;
}

.score-bar-container {
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    height: 6px;
    width: 100%;
    margin-top: 6px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}

/* Paper grid ———————————————————————— */
.paper-grid-card {
    background: var(--bg-card);
    backdrop-filter: blur(var(--blur-glass));
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 1rem 1.15rem;
    transition: transform var(--transition), box-shadow var(--transition), border-color var(--transition);
    height: 100%;
}
.paper-grid-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 35px rgba(59,130,246,0.12);
    border-color: rgba(59,130,246,0.22);
}
.paper-grid-card h4 {
    font-size: 0.88rem;
    font-weight: 600;
    line-height: 1.35;
    color: var(--text-primary);
    margin-bottom: 0.45rem;
}
.paper-grid-card .meta-line {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: 3px;
}

/* —— Paper detail panel —— */
.paper-detail-header {
    background: linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(6,182,212,0.06) 100%);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* —— References list —— */
.ref-item {
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border-glass);
    font-size: 0.83rem;
    line-height: 1.55;
    color: var(--text-muted);
}
.ref-item strong { color: var(--text-primary); }
.ref-item a { color: var(--accent-blue); font-size: 0.80rem; }

/* —— Streamlit overrides —— */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1px solid rgba(59,130,246,0.20) !important;
    background: rgba(17,24,39,0.6) !important;
    color: var(--text-primary) !important;
    transition: border-color var(--transition) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}
div.stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    font-weight: 500 !important;
    transition: all var(--transition) !important;
}
div.stButton > button:hover {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 15px rgba(59,130,246,0.15) !important;
    transform: translateY(-1px) !important;
}
/* Primary action button */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
    border: none !important;
    color: #fff !important;
}

details[data-testid="stExpander"] {
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    background: rgba(17,24,39,0.45) !important;
}

hr { border: none; border-top: 1px solid var(--border-glass); margin: 1.5rem 0; }

/* Tabs ————————————————————————— */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(17,24,39,0.4);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500;
    color: var(--text-muted) !important;
    transition: all var(--transition);
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.18) !important;
    color: #60a5fa !important;
}

/* Selectbox and slider ——————————————— */
.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1px solid rgba(59,130,246,0.20) !important;
    background: rgba(17,24,39,0.6) !important;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeSlideIn 0.4s ease-out forwards; }

/* Answer output area ————————————— */
.answer-container {
    background: rgba(17,24,39,0.55);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    font-size: 0.92rem;
    line-height: 1.75;
    color: var(--text-primary);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _score_badge_html(pct: int) -> str:
    cls = "score-high" if pct >= 75 else ("score-mid" if pct >= 50 else "score-low")
    return f'<span class="score-badge {cls}">{pct}% relevant</span>'


def _score_bar_html(pct: int) -> str:
    colour = "#10b981" if pct >= 75 else ("#f59e0b" if pct >= 50 else "#f43f5e")
    return (
        f'<div class="score-bar-container">'
        f'<div class="score-bar-fill" style="width:{pct}%;background:{colour};"></div>'
        f'</div>'
    )


def _doi_link_html(doi: str, style: str = "doi-link") -> str:
    if doi:
        return f'<a class="{style}" href="https://doi.org/{doi}" target="_blank">DOI: {doi}</a>'
    return ""


def _render_passage_card(chunk, idx: int, query: str = "") -> None:
    """Glass card for a single retrieved passage (used in both search tabs)."""
    pct = format_score(chunk.score)
    doi = chunk.metadata.get("doi", "")
    doi_html = _doi_link_html(doi)
    section_html = (
        f'<span class="section-badge">{chunk.section}</span>'
        if not chunk.is_summary else
        '<span class="section-badge">Abstract</span>'
    )
    title_short = chunk.metadata.get('title', '')[:110]
    text_preview = highlight_terms(chunk.text[:600], query) if query else chunk.text[:600]

    card_html = f"""
    <div class="glass-card fade-in" style="animation-delay:{idx * 0.05}s;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">
            <div>
                {section_html}
                <strong style="font-size:0.9rem;">{chunk.citation_label}</strong>
                <span style="font-size:0.81rem; color:var(--text-muted); margin-left:6px;">{title_short}</span>
            </div>
            <div>{_score_badge_html(pct)}</div>
        </div>
        {_score_bar_html(pct)}
        <div style="margin-top:10px; font-size:0.85rem; line-height:1.6; color:var(--text-primary);">
            {text_preview}
        </div>
        <div style="margin-top:8px; display:flex; gap:12px; align-items:center;">
            <span style="font-size:0.76rem; color:var(--text-muted);">
                📅 {chunk.metadata.get('year', '—')}  ·  🏥 {chunk.metadata.get('journal', '—')[:60]}
            </span>
            {doi_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander("📖 Full passage", expanded=False):
        full_text = highlight_terms(chunk.text, query) if query else chunk.text
        st.markdown(full_text)


def _render_paper_card_html(paper: dict) -> str:
    year    = paper.get("year", "—")
    journal = paper.get("journal", "")
    pmc_id  = paper.get("pmc_id", "")
    authors_raw  = paper.get("authors", "")
    first_author = authors_raw.split(",")[0].strip() if authors_raw else ""
    if first_author and len(authors_raw.split(",")) > 1:
        first_author += " et al."
    return f"""
    <div class="paper-grid-card">
        <h4>{paper.get('title', 'Untitled')[:130]}</h4>
        <div class="meta-line">📅 {year}  ·  🏥 {journal[:60]}</div>
        <div class="meta-line">✍️ {first_author[:80]}</div>
        <div class="meta-line" style="margin-top:4px;">
            <span class="section-badge" style="background:rgba(59,130,246,0.10);color:#60a5fa;">{pmc_id}</span>
        </div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; margin-bottom:0.5rem;">'
        '<span style="font-size:2.5rem;">🩺</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h2 style="text-align:center;margin:0;font-weight:700;font-size:1.1rem;">'
        'ENT Research Assistant</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center;color:var(--text-muted);font-size:0.78rem;margin-bottom:1.5rem;">'
        'Ask questions, explore & search 310 ENT papers</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    mode = st.radio(
        "Navigation",
        [
            "🔬 Ask the Literature",
            "📄 Paper Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Corpus stats
    try:
        stats = get_corpus_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-number">{stats["total_papers"]}</div>'
                f'<div class="stat-label">Papers</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-number">{stats["total_chunks"]:,}</div>'
                f'<div class="stat-label">Chunks</div></div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("⚠️ Could not load corpus stats.")
        stats = {}

    st.markdown(
        '<p style="text-align:center;color:var(--text-muted);font-size:0.68rem;margin-top:2rem;">'
        'ChromaDB · OpenRouter · Nemotron Ultra 253B</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1: Ask the Literature  (merged with semantic search)
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "🔬 Ask the Literature":
    st.markdown(
        '<div class="hero-header">'
        '<h1>🔬 Ask the Literature</h1>'
        '<p>Ask any clinical or research question and get a citation-backed answer '
        'synthesised from 310 peer-reviewed ENT papers. '
        'Switch to the “Ranked Passages” tab to browse the raw semantic results.</p></div>',
        unsafe_allow_html=True,
    )

    # —— Search bar + journal filter (no k slider, no year range) ——
    query_col, filter_col = st.columns([4, 1])
    with query_col:
        query = st.text_input(
            "Your question",
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
            "Journal", all_journals, label_visibility="collapsed", key="qa_journal"
        )

    # Build Chroma where-filter (journal only; year filter removed)
    where: dict | None = None
    if journal_filter and journal_filter != "All journals":
        where = {"journal": journal_filter}

    if query:
        st.markdown("---")
        answer_tab, passages_tab = st.tabs(["💡 AI Answer", "🎯 Ranked Passages"])

        with answer_tab:
            answer_box = st.container()
            with st.spinner("Retrieving relevant passages and generating answer…"):
                with answer_box:
                    st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                    st.write_stream(stream_answer(query, where=where))
                    st.markdown('</div>', unsafe_allow_html=True)

            # Citation strip
            sources = st.session_state.get("_last_sources", [])
            if sources:
                # deduplicate to unique papers for the strip
                seen_pmc: set[str] = set()
                unique_papers: list = []
                for c in sources:
                    if not c.is_summary and c.pmc_id not in seen_pmc:
                        seen_pmc.add(c.pmc_id)
                        unique_papers.append(c)
                if unique_papers:
                    st.markdown(
                        '<p style="font-size:0.78rem; color:var(--text-muted); margin-top:0.8rem;">'
                        '📚 Sources used: '
                        + ' · '.join(
                            f'<span style="color:#60a5fa;">{c.citation_label}</span>'
                            for c in unique_papers[:8]
                        )
                        + ('...' if len(unique_papers) > 8 else '')
                        + '</p>',
                        unsafe_allow_html=True,
                    )

        with passages_tab:
            sources = st.session_state.get("_last_sources", [])
            # Filter summaries from display; they are already surfaced in the answer
            display_chunks = [c for c in sources if not c.is_summary]

            if not display_chunks:
                st.info("ℹ️ Submit a question first to see ranked passages here.")
            else:
                st.caption(
                    f"🔎 {len(display_chunks)} passage{'s' if len(display_chunks) != 1 else ''} "
                    f"retrieved above relevance threshold"
                )
                for idx, chunk in enumerate(display_chunks):
                    _render_passage_card(chunk, idx, query=query)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2: Paper Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "📄 Paper Explorer":

    # ────────────────────────────────────────────────────────────────────────────
    # A) If a paper is selected, show detail view at the top
    # ────────────────────────────────────────────────────────────────────────────
    if "selected_paper" in st.session_state:
        paper  = st.session_state["selected_paper"]
        pmc_id = paper.get("pmc_id", "")

        # Header card
        doi_html = _doi_link_html(paper.get("doi", ""))
        st.markdown(
            f'<div class="paper-detail-header">'
            f'<h3 style="margin:0 0 8px 0;font-size:1.15rem;color:#e2e8f0;">'
            f'{paper.get("title", "Untitled")}</h3>'
            f'<div style="font-size:0.83rem;color:var(--text-muted);">'
            f'✍️ {paper.get("authors", "—")[:220]}</div>'
            f'<div style="font-size:0.83rem;color:var(--text-muted);margin-top:5px;">'
            f'📅 {paper.get("year", "—")}  ·  '
            f'🏥 {paper.get("journal", "—")}  ·  '
            f'<span class="section-badge" style="background:rgba(59,130,246,0.10);color:#60a5fa;">{pmc_id}</span>'
            f'  {doi_html}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Action buttons row
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
        with btn_col1:
            summarize_clicked = st.button(
                "🧠 Summarize full paper",
                key="deep_summary_btn",
                use_container_width=True,
                type="primary",
            )
        with btn_col2:
            if st.button("✕ Close paper", key="close_paper", use_container_width=True):
                del st.session_state["selected_paper"]
                st.session_state.pop("_deep_summary_done", None)
                st.rerun()

        # —— LLM structured summary (streamed on demand) ——
        if summarize_clicked:
            st.session_state["_deep_summary_done"] = False
            st.session_state["_trigger_summary"] = pmc_id

        if st.session_state.get("_trigger_summary") == pmc_id and not st.session_state.get("_deep_summary_done"):
            st.markdown(
                '<div class="glass-card" style="border-color:rgba(59,130,246,0.2);">'
                '<p style="font-size:0.8rem;color:var(--text-muted);margin-bottom:8px;">'
                '🧠 AI-generated structured summary</p>',
                unsafe_allow_html=True,
            )
            with st.spinner("Reading all paper sections…"):
                st.write_stream(generate_deep_summary(pmc_id))
            st.markdown('</div>', unsafe_allow_html=True)
            st.session_state["_deep_summary_done"] = True

        # —— Abstract / stored summary ——
        summary_text = paper.get("summary_text", "")
        if summary_text:
            with st.expander("📋 Abstract / Stored Summary", expanded=True):
                st.markdown(
                    f'<div style="font-size:0.88rem;line-height:1.7;color:var(--text-primary);">'
                    f'{summary_text}</div>',
                    unsafe_allow_html=True,
                )

        # —— Full paper sections (no truncation) ——
        with st.expander("📑 Full paper sections", expanded=False):
            retriever = get_retriever()
            all_chunks = retriever.retrieve_by_paper(pmc_id)
            body_chunks = [c for c in all_chunks if not c.is_summary]
            if not body_chunks:
                st.caption("No section chunks found for this paper.")
            for c in body_chunks:
                st.markdown(
                    f'<span class="section-badge">{c.section}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="font-size:0.86rem;line-height:1.7;color:var(--text-primary);'
                    f'margin:6px 0 12px 0;">'
                    f'{c.text}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("---")

        # —— References ——
        with st.expander("📚 References", expanded=False):
            retriever = get_retriever()
            all_chunks = retriever.retrieve_by_paper(pmc_id)
            refs_seen: set = set()
            refs: list[dict] = []
            for c in all_chunks:
                for r in c.references:
                    key = (
                        r.get("title"),
                        r.get("doi"),
                        r.get("pmid"),
                        r.get("pmcid"),
                    )
                    if key in refs_seen:
                        continue
                    refs_seen.add(key)
                    refs.append(r)

            if not refs:
                st.caption("No structured references stored for this paper.")
            else:
                st.caption(f"{len(refs)} references")
                for i, r in enumerate(refs, 1):
                    title   = r.get("title", "Untitled")
                    authors = r.get("authors", "")
                    journal = r.get("journal", "")
                    year    = r.get("year", "")
                    doi     = r.get("doi", "")

                    doi_part = ""
                    if doi:
                        doi_part = f' · <a class="doi-link" href="https://doi.org/{doi}" target="_blank">{doi}</a>'

                    meta_parts = []
                    if authors: meta_parts.append(authors[:80] + (" et al." if len(authors) > 80 else ""))
                    if year:    meta_parts.append(f"({year})")
                    if journal: meta_parts.append(f"<em>{journal[:60]}</em>")
                    meta_str = " ".join(meta_parts)

                    st.markdown(
                        f'<div class="ref-item">'
                        f'<strong>{i}. {title[:160]}</strong><br>'
                        f'{meta_str}{doi_part}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")

    # ────────────────────────────────────────────────────────────────────────────
    # B) Paper browser / search grid
    # ────────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero-header">'
        '<h1>📄 Paper Explorer</h1>'
        '<p>Browse 310 peer-reviewed ENT papers. '
        'Search by title, author or PMC ID, then click a card to read the full text and references.</p></div>',
        unsafe_allow_html=True,
    )

    search_query = st.text_input(
        "Search papers",
        placeholder="Title keyword, author name or PMC ID (e.g. PMC9012345)…",
        label_visibility="collapsed",
        key="paper_search",
    )

    papers = search_papers_by_title(search_query) if search_query else get_all_papers(limit=60)

    if not papers:
        st.info("No papers found. Try a different search term.")
    else:
        st.caption(f"Showing {len(papers)} paper{'s' if len(papers) != 1 else ''}")

        cols_per_row = 3
        for row_start in range(0, len(papers), cols_per_row):
            row_papers = papers[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, paper in zip(cols, row_papers):
                with col:
                    st.markdown(_render_paper_card_html(paper), unsafe_allow_html=True)
                    pmc_id = paper.get("pmc_id", "")
                    if st.button(
                        "View details",
                        key=f"view_{pmc_id}_{row_start}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_paper"] = paper
                        st.session_state.pop("_trigger_summary", None)
                        st.session_state.pop("_deep_summary_done", None)
                        st.rerun()
