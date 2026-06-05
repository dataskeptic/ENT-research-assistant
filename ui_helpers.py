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


# ── cached singletons ────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource(show_spinner=False)
def _get_chroma_collection():
    ret_cfg = get_retriever_config()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=ret_cfg.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ── corpus stats ────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def get_corpus_stats() -> dict:
    col = _get_chroma_collection()
    total_chunks = col.count()
    all_meta = col.get(include=["metadatas"], limit=total_chunks)
    pmc_ids = {m.get("pmc_id", "") for m in all_meta["metadatas"] if m}
    pmc_ids.discard("")
    journals = sorted({m.get("journal", "") for m in all_meta["metadatas"] if m and m.get("journal")})
    return {
        "total_chunks": total_chunks,
        "total_papers": len(pmc_ids),
        "journals": journals,
    }


# ── relevance threshold filter ──────────────────────────────────────────────────────────

# Internal default: fetch up to 20 candidates and filter by relevance %
_RETRIEVAL_CANDIDATE_K = 20
_DEFAULT_RELEVANCE_THRESHOLD = 68  # percent (0–100)


def filter_by_relevance(
    chunks: list[RetrievedChunk],
    min_pct: int = _DEFAULT_RELEVANCE_THRESHOLD,
) -> list[RetrievedChunk]:
    """
    Keep only chunks whose cosine-similarity relevance meets the threshold.
    Summary chunks (injected by the retriever for context) are kept regardless
    so the LLM always has paper context.
    """
    return [
        c for c in chunks
        if c.is_summary or format_score(c.score) >= min_pct
    ]


# ── streaming answer ──────────────────────────────────────────────────────────────────────

def stream_answer(
    query: str,
    where: dict | None = None,
    min_relevance_pct: int = _DEFAULT_RELEVANCE_THRESHOLD,
) -> Generator[str, None, None]:
    """
    Generator that yields answer tokens for st.write_stream.

    Retrieves up to _RETRIEVAL_CANDIDATE_K candidates, filters by
    relevance threshold (not a user-visible slider), then streams the
    LLM answer. Sources are stored in st.session_state["_last_sources"].
    """
    retriever  = get_retriever()
    model_cfg  = get_model_config()
    client     = make_openrouter_client()

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

    # 3. Build context
    context = _build_context(chunks)

    # 4. Stream LLM
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


# ── paper search (metadata-based) ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=60)
def search_papers_by_title(query: str, limit: int = 30) -> list[dict]:
    """
    Search for papers by title substring, author, or PMC ID.
    Returns deduplicated list of paper-level metadata dicts.
    """
    col = _get_chroma_collection()
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
    seen: set[str] = set()

    for meta, doc in zip(results["metadatas"], results["documents"]):
        pmc_id = meta.get("pmc_id", "")
        if pmc_id in seen:
            continue

        title   = meta.get("title", "").lower()
        authors = meta.get("authors", "").lower()

        if (
            query_lower in title
            or query_lower in pmc_id.lower()
            or query_lower in authors
        ):
            seen.add(pmc_id)
            papers.append({**meta, "summary_text": doc})

        if len(papers) >= limit:
            break

    return papers


def get_all_papers(limit: int = 500) -> list[dict]:
    """Return all papers sorted by year desc, then title."""
    col = _get_chroma_collection()
    total = col.count()
    if total == 0:
        return []

    results = col.get(
        where={"section": "__summary__"},
        include=["metadatas", "documents"],
        limit=total,
    )

    papers: list[dict] = []
    seen: set[str] = set()
    for meta, doc in zip(results["metadatas"], results["documents"]):
        pmc_id = meta.get("pmc_id", "")
        if pmc_id in seen:
            continue
        seen.add(pmc_id)
        papers.append({**meta, "summary_text": doc})

    papers.sort(
        key=lambda p: (p.get("year", "0"), p.get("title", "")),
        reverse=True,
    )
    return papers[:limit]


# ── deep summary generation ─────────────────────────────────────────────────────────────

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


def generate_deep_summary(pmc_id: str) -> Generator[str, None, None]:
    """Stream a structured LLM summary for a given paper."""
    retriever  = get_retriever()
    model_cfg  = get_model_config()
    client     = make_openrouter_client()

    chunks = retriever.retrieve_by_paper(pmc_id)
    if not chunks:
        yield "No chunks found for this paper."
        return

    parts = []
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


# ── score helpers ────────────────────────────────────────────────────────────────────────────

def format_score(distance: float) -> int:
    """Convert cosine distance (0=identical, 2=opposite) to relevance %."""
    similarity = max(0.0, 1.0 - distance)
    return int(round(similarity * 100))


def highlight_terms(text: str, query: str) -> str:
    """Wrap query terms in the text with bold markdown for highlighting."""
    words = {w for w in query.lower().split() if len(w) > 2}
    if not words:
        return text
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b',
        re.IGNORECASE,
    )
    return pattern.sub(r'**\1**', text)
