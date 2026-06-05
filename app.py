"""
app.py

Streamlit front-end for the ETN Research Assistant RAG pipeline.

Features:
  1. 🔬 Ask the Literature — Q&A with citation-backed, streamed answers
  2. 📄 Paper Explorer     — Browse & summarise individual papers
  3. 🔍 Semantic Search    — Pure vector search with filters

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
)

# ═══════════════════════════════════════════════════════════════════════════════
# Page config — must be the first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ETN Research Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CSS Design System
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
/* ── Google Font ───────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root tokens ──────────────────────────────────────────────────────── */
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

/* ── Global ───────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar polish ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #0a0f1c 100%);
    border-right: 1px solid var(--border-glass);
}
section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 500;
}

/* ── Glass card ───────────────────────────────────────────────────────── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(var(--blur-glass));
    -webkit-backdrop-filter: blur(var(--blur-glass));
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: transform var(--transition), box-shadow var(--transition),
                border-color var(--transition);
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(59,130,246,0.10);
    border-color: rgba(59,130,246,0.18);
}

/* ── Score badge ──────────────────────────────────────────────────────── */
.score-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.score-high   { background: rgba(16,185,129,0.15); color: #34d399; }
.score-mid    { background: rgba(245,158,11,0.15); color: #fbbf24; }
.score-low    { background: rgba(244,63,94,0.15);  color: #fb7185; }

/* ── Section badge ────────────────────────────────────────────────────── */
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

/* ── DOI link ─────────────────────────────────────────────────────────── */
.doi-link {
    color: var(--accent-blue);
    text-decoration: none;
    font-size: 0.82rem;
    transition: color var(--transition);
}
.doi-link:hover {
    color: var(--accent-cyan);
    text-decoration: underline;
}

/* ── Stat card (sidebar) ──────────────────────────────────────────────── */
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

/* ── Hero header ──────────────────────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
    line-height: 1.25;
}
.hero-header p {
    color: var(--text-muted);
    font-size: 0.95rem;
    max-width: 600px;
    margin: 0 auto;
}

/* ── Highlighted terms ────────────────────────────────────────────────── */
mark, .highlight-term {
    background: rgba(59,130,246,0.22);
    color: #93c5fd;
    padding: 1px 3px;
    border-radius: 3px;
}

/* ── Progress bar for scores ──────────────────────────────────────────── */
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

/* ── Paper grid card ──────────────────────────────────────────────────── */
.paper-grid-card {
    background: var(--bg-card);
    backdrop-filter: blur(var(--blur-glass));
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    transition: transform var(--transition), box-shadow var(--transition),
                border-color var(--transition);
    height: 100%;
    cursor: pointer;
}
.paper-grid-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 35px rgba(59,130,246,0.12);
    border-color: rgba(59,130,246,0.22);
}
.paper-grid-card h4 {
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.35;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.paper-grid-card .meta-line {
    font-size: 0.76rem;
    color: var(--text-muted);
    margin-bottom: 3px;
}

/* ── Streamlit overrides ──────────────────────────────────────────────── */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1px solid rgba(59,130,246,0.20) !important;
    background: rgba(17,24,39,0.6) !important;
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
}

/* ── Expander restyle ─────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    background: rgba(17,24,39,0.45) !important;
}

/* ── Divider ──────────────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid var(--border-glass);
    margin: 1.5rem 0;
}

/* ── Animate fade-in for results ──────────────────────────────────────── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeSlideIn 0.4s ease-out forwards;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper rendering functions
# ═══════════════════════════════════════════════════════════════════════════════

def _score_badge_html(pct: int) -> str:
    """Return a coloured badge span for a relevance percentage."""
    if pct >= 75:
        cls = "score-high"
    elif pct >= 50:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{pct}% relevant</span>'


def _score_bar_html(pct: int) -> str:
    """Thin progress bar for relevance score."""
    if pct >= 75:
        colour = "#10b981"
    elif pct >= 50:
        colour = "#f59e0b"
    else:
        colour = "#f43f5e"
    return (
        f'<div class="score-bar-container">'
        f'<div class="score-bar-fill" style="width:{pct}%;background:{colour};"></div>'
        f'</div>'
    )


def _render_source_card(chunk, idx: int) -> None:
    """Render a single source chunk as a glass card inside the Sources panel."""
    pct = format_score(chunk.score)
    doi = chunk.metadata.get("doi", "")
    doi_html = (
        f'<a class="doi-link" href="https://doi.org/{doi}" target="_blank">DOI: {doi}</a>'
        if doi else ""
    )
    section_html = (
        f'<span class="section-badge">{chunk.section}</span>'
        if not chunk.is_summary else
        '<span class="section-badge">Abstract / Summary</span>'
    )

    card_html = f"""
    <div class="glass-card fade-in" style="animation-delay:{idx * 0.06}s;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div>
                {section_html}
                <strong style="font-size:0.88rem;">{chunk.citation_label}</strong>
            </div>
            <div>{_score_badge_html(pct)}</div>
        </div>
        <div style="font-size:0.82rem; color:var(--text-muted); margin-bottom:6px;">
            {chunk.metadata.get('title', '')[:120]}
            {' — ' + chunk.metadata.get('journal', '') if chunk.metadata.get('journal') else ''}
        </div>
        {_score_bar_html(pct)}
        <div style="margin-top:4px;">{doi_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander(f"📖 View passage text", expanded=False):
        st.caption(chunk.text[:1500])


def _render_paper_card_html(paper: dict) -> str:
    """Return HTML for a paper card in the Paper Explorer grid."""
    year = paper.get("year", "—")
    journal = paper.get("journal", "")
    pmc_id = paper.get("pmc_id", "")
    authors_raw = paper.get("authors", "")
    first_author = authors_raw.split(",")[0].strip() if authors_raw else ""
    if first_author and len(authors_raw.split(",")) > 1:
        first_author += " et al."

    return f"""
    <div class="paper-grid-card">
        <h4>{paper.get('title', 'Untitled')[:130]}</h4>
        <div class="meta-line">📅 {year}  ·  🏥 {journal[:60]}</div>
        <div class="meta-line">✍️ {first_author[:80]}</div>
        <div class="meta-line" style="margin-top:4px;">
            <span class="section-badge" style="background:rgba(59,130,246,0.10);color:#60a5fa;">
                {pmc_id}
            </span>
        </div>
    </div>
    """


def _doi_link(paper: dict) -> str:
    """Return a DOI anchor tag or empty string."""
    doi = paper.get("doi", "")
    if doi:
        return (
            f'  ·  <a class="doi-link" href="https://doi.org/{doi}" '
            f'target="_blank">DOI: {doi}</a>'
        )
    return ""


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
        '<h2 style="text-align:center; margin:0; font-weight:700; font-size:1.15rem;">'
        'ETN Research Assistant</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center; color:var(--text-muted); font-size:0.78rem; margin-bottom:1.5rem;">'
        'AI-powered ENT literature search</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    mode = st.radio(
        "Navigation",
        [
            "🔬 Ask the Literature",
            "📄 Paper Explorer",
            "🔍 Semantic Search",
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

    st.markdown(
        '<p style="text-align:center; color:var(--text-muted); font-size:0.68rem; margin-top:2rem;">'
        'Powered by ChromaDB + OpenRouter<br>Nemotron Ultra 253B</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 1: Ask the Literature (Q&A)
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "🔬 Ask the Literature":
    st.markdown(
        '<div class="hero-header">'
        '<h1>Ask the Literature</h1>'
        '<p>Ask any clinical question and get a citation-backed answer '
        'synthesised from 310 peer-reviewed ENT papers.</p></div>',
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Your question",
        placeholder="e.g. What are the outcomes after TORS for oropharyngeal cancer?",
        label_visibility="collapsed",
        key="qa_query",
    )

    # Advanced filters
    with st.expander("⚙️ Advanced filters", expanded=False):
        filter_cols = st.columns(3)
        with filter_cols[0]:
            try:
                all_years = stats.get("years", [])
            except Exception:
                all_years = []
            year_filter = st.select_slider(
                "Year range",
                options=all_years if all_years else ["—"],
                value=(all_years[0], all_years[-1]) if len(all_years) >= 2 else None,
                disabled=len(all_years) < 2,
            )
        with filter_cols[1]:
            try:
                all_journals = ["All"] + stats.get("journals", [])
            except Exception:
                all_journals = ["All"]
            journal_filter = st.selectbox("Journal", all_journals)
        with filter_cols[2]:
            top_k = st.slider("Passages to retrieve", 2, 20, 8)

    # Build Chroma where-filter
    where: dict | None = None
    where_clauses = []
    try:
        if year_filter and year_filter != ("—",) and len(all_years) >= 2:
            where_clauses.append({"year": {"$gte": str(year_filter[0])}})
            where_clauses.append({"year": {"$lte": str(year_filter[1])}})
    except Exception:
        pass
    if journal_filter and journal_filter != "All":
        where_clauses.append({"journal": journal_filter})

    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    # Submit
    if query:
        st.markdown("---")
        st.markdown("#### 💡 Answer")

        answer_container = st.empty()
        with st.spinner("Retrieving passages & generating answer…"):
            full_answer = answer_container.write_stream(
                stream_answer(query, top_k=top_k, where=where)
            )

        # Sources panel
        sources = st.session_state.get("_last_sources", [])
        if sources:
            st.markdown("---")
            st.markdown(f"#### 📚 Sources ({len(sources)} passages)")
            # Deduplicate by paper for the top-level view
            for idx, chunk in enumerate(sources):
                _render_source_card(chunk, idx)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 2: Paper Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "📄 Paper Explorer":
    st.markdown(
        '<div class="hero-header">'
        '<h1>Paper Explorer</h1>'
        '<p>Browse the full corpus, search by title or PMC ID, '
        'and generate structured deep summaries.</p></div>',
        unsafe_allow_html=True,
    )

    search_query = st.text_input(
        "Search papers",
        placeholder="Search by title keyword or PMC ID (e.g. PMC9012345)…",
        label_visibility="collapsed",
        key="paper_search",
    )

    # Show results
    if search_query:
        papers = search_papers_by_title(search_query)
    else:
        papers = get_all_papers(limit=60)

    if not papers:
        st.info("No papers found. Try a different search term.")
    else:
        st.caption(f"Showing {len(papers)} paper{'s' if len(papers) != 1 else ''}")

        # Grid display
        cols_per_row = 3
        for row_start in range(0, len(papers), cols_per_row):
            row_papers = papers[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, paper in zip(cols, row_papers):
                with col:
                    st.markdown(
                        _render_paper_card_html(paper),
                        unsafe_allow_html=True,
                    )
                    pmc_id = paper.get("pmc_id", "")
                    if st.button(
                        f"View details",
                        key=f"view_{pmc_id}_{row_start}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_paper"] = paper

    # ── Detail panel ────────────────────────────────────────────────────────
    if "selected_paper" in st.session_state:
        paper = st.session_state["selected_paper"]
        pmc_id = paper.get("pmc_id", "")

        st.markdown("---")

        # Header
        st.markdown(
            f'<div class="glass-card">'
            f'<h3 style="margin:0 0 8px 0; font-size:1.15rem;">{paper.get("title", "Untitled")}</h3>'
            f'<div style="font-size:0.84rem; color:var(--text-muted);">'
            f'✍️ {paper.get("authors", "—")[:200]}</div>'
            f'<div style="font-size:0.84rem; color:var(--text-muted); margin-top:4px;">'
            f'📅 {paper.get("year", "—")}  ·  🏥 {paper.get("journal", "—")}'
            f'  ·  <span class="section-badge" style="background:rgba(59,130,246,0.10);color:#60a5fa;">{pmc_id}</span>'
            f'{_doi_link(paper)}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Instant summary
        summary_text = paper.get("summary_text", "")
        if summary_text:
            st.markdown("##### 📋 Abstract / Stored Summary")
            st.markdown(
                f'<div class="glass-card" style="font-size:0.88rem; line-height:1.6;">'
                f'{summary_text[:3000]}</div>',
                unsafe_allow_html=True,
            )

        # Deep summary
        col_btn, col_clear = st.columns([1, 3])
        with col_btn:
            generate_clicked = st.button(
                "🧠 Generate Deep Summary",
                key="deep_summary_btn",
                use_container_width=True,
            )
        with col_clear:
            if st.button("✕ Close paper", key="close_paper"):
                del st.session_state["selected_paper"]
                st.rerun()

        if generate_clicked:
            st.markdown("##### 🧠 AI-Generated Structured Summary")
            summary_box = st.empty()
            with st.spinner("Reading all paper sections…"):
                summary_box.write_stream(generate_deep_summary(pmc_id))

        # Full section view
        with st.expander("📑 View all paper sections", expanded=False):
            retriever = get_retriever()
            all_chunks = retriever.retrieve_by_paper(pmc_id)
            for c in all_chunks:
                if c.is_summary:
                    continue
                st.markdown(
                    f'<span class="section-badge">{c.section}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(c.text[:2000])
                st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 3: Semantic Search
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "🔍 Semantic Search":
    st.markdown(
        '<div class="hero-header">'
        '<h1>Semantic Search</h1>'
        '<p>Search the literature by meaning, not just keywords. '
        'Adjust filters and see ranked results with relevance scores.</p></div>',
        unsafe_allow_html=True,
    )

    search_query = st.text_input(
        "Search query",
        placeholder="e.g. transoral robotic surgery complications",
        label_visibility="collapsed",
        key="sem_query",
    )

    # Controls
    ctrl_cols = st.columns([1, 1, 1])
    with ctrl_cols[0]:
        sem_top_k = st.slider("Number of results", 1, 20, 8, key="sem_topk")
    with ctrl_cols[1]:
        try:
            sem_years = stats.get("years", [])
        except Exception:
            sem_years = []
        sem_year_filter = st.select_slider(
            "Year range",
            options=sem_years if sem_years else ["—"],
            value=(sem_years[0], sem_years[-1]) if len(sem_years) >= 2 else None,
            disabled=len(sem_years) < 2,
            key="sem_year_filter",
        )
    with ctrl_cols[2]:
        try:
            sem_journals = ["All"] + stats.get("journals", [])
        except Exception:
            sem_journals = ["All"]
        sem_journal = st.selectbox("Journal", sem_journals, key="sem_journal")

    if search_query:
        # Build filter
        sem_where_clauses: list[dict] = []
        try:
            if sem_year_filter and sem_year_filter != ("—",) and len(sem_years) >= 2:
                sem_where_clauses.append({"year": {"$gte": str(sem_year_filter[0])}})
                sem_where_clauses.append({"year": {"$lte": str(sem_year_filter[1])}})
        except Exception:
            pass
        if sem_journal and sem_journal != "All":
            sem_where_clauses.append({"journal": sem_journal})

        sem_where: dict | None = None
        if len(sem_where_clauses) == 1:
            sem_where = sem_where_clauses[0]
        elif len(sem_where_clauses) > 1:
            sem_where = {"$and": sem_where_clauses}

        # Run retrieval
        retriever = get_retriever()
        with st.spinner("Searching…"):
            results = retriever.retrieve(search_query, top_k=sem_top_k, where=sem_where)

        if not results:
            st.info("No results found. Try broadening your query or adjusting filters.")
        else:
            st.markdown("---")
            st.markdown(f"#### 🎯 {len(results)} results")

            for idx, chunk in enumerate(results):
                pct = format_score(chunk.score)
                doi = chunk.metadata.get("doi", "")
                doi_html = (
                    f'<a class="doi-link" href="https://doi.org/{doi}" target="_blank">DOI ↗</a>'
                    if doi else ""
                )
                section_html = (
                    f'<span class="section-badge">{chunk.section}</span>'
                    if not chunk.is_summary else
                    '<span class="section-badge">Summary</span>'
                )

                card_html = f"""
                <div class="glass-card fade-in" style="animation-delay:{idx * 0.05}s;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">
                        <div>
                            {section_html}
                            <strong style="font-size:0.9rem;">{chunk.citation_label}</strong>
                            <span style="font-size:0.82rem; color:var(--text-muted); margin-left:6px;">
                                {chunk.metadata.get('title', '')[:100]}
                            </span>
                        </div>
                        <div>{_score_badge_html(pct)}</div>
                    </div>
                    {_score_bar_html(pct)}
                    <div style="margin-top:10px; font-size:0.85rem; line-height:1.55; color:var(--text-primary);">
                        {highlight_terms(chunk.text[:600], search_query)}
                    </div>
                    <div style="margin-top:8px; display:flex; gap:12px; align-items:center;">
                        <span style="font-size:0.76rem; color:var(--text-muted);">
                            📅 {chunk.metadata.get('year', '—')}  ·  🏥 {chunk.metadata.get('journal', '—')[:50]}
                        </span>
                        {doi_html}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                with st.expander(f"📖 Full passage", expanded=False):
                    highlighted = highlight_terms(chunk.text, search_query)
                    st.markdown(highlighted)

