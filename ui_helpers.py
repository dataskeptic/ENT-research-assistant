"""
ui_helpers.py

Thin adapter between Streamlit and the existing rag/ package.
All expensive objects (pipeline, retriever, chroma client) are cached
via Streamlit's @st.cache_resource so they survive reruns.
"""

from __future__ import annotations

import re
from typing import Generator

import streamlit as st
import chromadb

from rag.config import (
    CHROMA_DIR,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_HEADERS,
    SYSTEM_PROMPT,
    get_model_config, get_retriever_config,
)
from rag.ingestion import make_openrouter_client
from rag.retriever import Retriever, RetrievedChunk
from rag.pipeline import RAGPipeline, _build_context


# ── cached singletons ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource(show_spinner=False)
def _get_chroma_collection():
    ret_cfg = get_retriever_config()
    client  = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=ret_cfg.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ── corpus stats ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def get_corpus_stats() -> dict:
    """Return aggregate stats about the Chroma collection.

    Year metadata is intentionally excluded — the corpus contains only
    recent papers so a year-range filter adds no value.
    """
    col        = _get_chroma_collection()
    total_chunks = col.count()
    all_meta   = col.get(include=["metadatas"], limit=total_chunks)
    dois       = {m.get("doi", "") for m in all_meta["metadatas"] if m}
    dois.discard("")
    journals   = sorted({
        m.get("journal", "")
        for m in all_meta["metadatas"]
        if m and m.get("journal")
    })
    return {
        "total_chunks":  total_chunks,
        "total_papers":  len(dois),
        "journals":      journals,
        # 'years' intentionally omitted
    }


# ── relevance threshold filter ───────────────────────────────────────────────

# Fetch this many candidates from ChromaDB, then apply the threshold.
# Never exposed to the user as a slider.
_RETRIEVAL_CANDIDATE_K   = 20
_DEFAULT_RELEVANCE_THRESHOLD = 68   # percent (0–100)


def filter_by_relevance(
    chunks: list[RetrievedChunk],
    min_pct: int = _DEFAULT_RELEVANCE_THRESHOLD,
) -> list[RetrievedChunk]:
    """Keep chunks whose relevance score meets the threshold.

    Summary / abstract chunks are always kept so the LLM has paper context
    even when the full-text passages score slightly below the threshold.
    """
    return [
        c for c in chunks
        if c.is_summary or format_score(c.score) >= min_pct
    ]


# ── streaming answer ─────────────────────────────────────────────────────────

def stream_answer(
    query: str,
    where: dict | None = None,
    min_relevance_pct: int = _DEFAULT_RELEVANCE_THRESHOLD,
) -> Generator[str, None, None]:
    """Yield answer tokens for st.write_stream.

    Internally fetches _RETRIEVAL_CANDIDATE_K candidates and filters by
    relevance threshold.  The top-k value is never surfaced to the user.
    Retrieved chunks are stored in st.session_state["_last_sources"] so
    the Ranked Passages tab can read them without a second retrieval.
    """
    retriever = get_retriever()
    model_cfg = get_model_config()
    client    = make_openrouter_client()

    # 1. Retrieve candidates
    raw_chunks = retriever.retrieve(
        query,
        top_k=_RETRIEVAL_CANDIDATE_K,
        where=where,
    )

    # 2. Apply relevance threshold (summaries always pass)
    chunks = filter_by_relevance(raw_chunks, min_pct=min_relevance_pct)
    st.session_state["_last_sources"] = chunks

    if not chunks:
        st.session_state["_last_usage"] = {}
        yield (
            "No passages met the relevance threshold for this query. "
            "Try rephrasing or using broader clinical terminology."
        )
        return

    # 3. Build context string
    context = _build_context(chunks)

    # 4. Stream LLM response
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{context}\n\nQuestion: {query}"},
    ]

    stream = client.chat.completions.create(
        model=model_cfg.llm,
        messages=messages,
        max_tokens=model_cfg.llm_max_tokens,
        temperature=model_cfg.llm_temperature,
        stream=True,
    )

    for event in stream:
        delta = event.choices[0].delta
        if delta.content:
            yield delta.content


# ── paper search (metadata-based) ────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=60)
def search_papers_by_title(query: str, limit: int = 60) -> list[dict]:
    """Search papers by title substring, author, or PMC ID.

    Returns a deduplicated list of paper-level metadata dicts, each with
    an extra 'summary_text' key containing the stored abstract/summary doc.
    """
    col   = _get_chroma_collection()
    total = col.count()
    if total == 0:
        return []

    results = col.get(
        where={"section": "__summary__"},
        include=["metadatas", "documents"],
        limit=total,
    )

    query_lower = query.lower().strip()
    papers: list[dict] = []
    seen:   set[str]   = set()

    for meta, doc in zip(results["metadatas"], results["documents"]):
        doi = meta.get("doi", "")
        if doi in seen:
            continue

        title   = meta.get("title",   "").lower()
        authors = meta.get("authors", "").lower()

        if (
            query_lower in title
            or query_lower in doi.lower()
            or query_lower in authors
        ):
            seen.add(doi)
            papers.append({**meta, "summary_text": doc})

        if len(papers) >= limit:
            break

    return papers


def get_all_papers(limit: int = 500) -> list[dict]:
    """Return all papers sorted by year descending, then title."""
    col   = _get_chroma_collection()
    total = col.count()
    if total == 0:
        return []

    results = col.get(
        where={"section": "__summary__"},
        include=["metadatas", "documents"],
        limit=total,
    )

    papers: list[dict] = []
    seen:   set[str]   = set()
    for meta, doc in zip(results["metadatas"], results["documents"]):
        doi = meta.get("doi", "")
        if doi in seen:
            continue
        seen.add(doi)
        papers.append({**meta, "summary_text": doc})

    papers.sort(
        key=lambda p: (p.get("year", "0"), p.get("title", "")),
        reverse=True,
    )
    return papers[:limit]


# ── deep / full-paper summary ─────────────────────────────────────────────────

DEEP_SUMMARY_PROMPT = """\
You are an expert ENT surgical research assistant. Given the full text of a \
medical research paper (provided as section-by-section chunks), produce a \
concise structured summary using EXACTLY these headings:

## Objective
## Study Design & Methods
## Key Findings
## Clinical Implications
## Limitations

Use clear, clinical language. Be specific with numbers, outcomes, and p-values \
where available. Keep each section to 2–4 sentences.
"""


def generate_deep_summary(doi: str) -> Generator[str, None, None]:
    """Stream a structured LLM summary covering all sections of a paper."""
    retriever = get_retriever()
    model_cfg = get_model_config()
    client    = make_openrouter_client()

    chunks = retriever.retrieve_by_paper(doi)
    if not chunks:
        yield "No chunks found for this paper."
        return

    # Build ordered full-text block from all sections
    parts: list[str] = []
    for c in chunks:
        label = c.section if not c.is_summary else "ABSTRACT/SUMMARY"
        parts.append(f"[{label}]\n{c.text.strip()}\n")

    paper_text = "\n---\n".join(parts)

    messages = [
        {"role": "system", "content": DEEP_SUMMARY_PROMPT},
        {"role": "user",   "content": paper_text},
    ]

    stream = client.chat.completions.create(
        model=model_cfg.llm,
        messages=messages,
        max_tokens=model_cfg.llm_max_tokens,
        temperature=0.15,
        stream=True,
    )

    for event in stream:
        delta = event.choices[0].delta
        if delta.content:
            yield delta.content


# ── score & highlight helpers ─────────────────────────────────────────────────

def format_score(distance: float) -> int:
    """Convert cosine distance (0 = identical, 2 = opposite) to relevance %."""
    similarity = max(0.0, 1.0 - distance)
    return int(round(similarity * 100))


def highlight_terms(text: str, query: str) -> str:
    """Wrap query terms in HTML <mark> tags for rendering inside st.markdown blocks.

    Uses <mark> instead of **bold** markdown so the styling is applied
    correctly when the surrounding container uses unsafe_allow_html=True.
    """
    words = {w for w in query.lower().split() if len(w) > 2}
    if not words:
        return text
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r')\b',
        re.IGNORECASE,
    )
    return pattern.sub(r'<mark>\1</mark>', text)
