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
    """Return a singleton RAGPipeline (shared across all sessions)."""
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    """Return a singleton Retriever (shared across all sessions)."""
    return Retriever()


@st.cache_resource(show_spinner=False)
def _get_chroma_collection():
    """Direct access to the Chroma collection for metadata queries."""
    ret_cfg = get_retriever_config()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=ret_cfg.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ── corpus stats ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def get_corpus_stats() -> dict:
    """Return corpus-level counts for the sidebar."""
    col = _get_chroma_collection()
    total_chunks = col.count()

    # Count distinct PMC IDs
    all_meta = col.get(include=["metadatas"], limit=total_chunks)
    pmc_ids = {m.get("pmc_id", "") for m in all_meta["metadatas"] if m}
    pmc_ids.discard("")

    # Collect distinct years and journals for filters
    years = sorted({m.get("year", "") for m in all_meta["metadatas"] if m and m.get("year")})
    journals = sorted({m.get("journal", "") for m in all_meta["metadatas"] if m and m.get("journal")})

    return {
        "total_chunks": total_chunks,
        "total_papers": len(pmc_ids),
        "years": years,
        "journals": journals,
    }


# ── streaming answer ──────────────────────────────────────────────────────────

def stream_answer(
    query: str,
    top_k: int | None = None,
    where: dict | None = None,
) -> Generator[str, None, tuple[list[RetrievedChunk], dict]]:
    """
    Generator that yields answer tokens one-by-one for st.write_stream.

    After exhausting the generator, call .send(None) or iterate fully —
    the sources and usage are stashed in st.session_state["_last_sources"]
    and st.session_state["_last_usage"].
    """
    retriever = get_retriever()
    model_cfg = get_model_config()
    client = make_openrouter_client()

    # 1. retrieve
    chunks = retriever.retrieve(query, top_k=top_k, where=where)
    st.session_state["_last_sources"] = chunks

    if not chunks:
        st.session_state["_last_usage"] = {}
        yield "No relevant passages found in the literature database for this query."
        return

    # 2. build context
    context = _build_context(chunks)

    # 3. stream LLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nQuestion: {query}"},
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
def search_papers_by_title(query: str, limit: int = 20) -> list[dict]:
    """
    Search for papers by title substring or PMC ID in Chroma metadata.
    Returns deduplicated list of paper-level metadata dicts.
    """
    col = _get_chroma_collection()
    total = col.count()
    if total == 0:
        return []

    # Get all summary chunks (one per paper)
    results = col.get(
        where={"section": "__summary__"},
        include=["metadatas", "documents"],
        limit=total,
    )

    query_lower = query.lower().strip()
    papers = []
    seen = set()

    for meta, doc in zip(results["metadatas"], results["documents"]):
        pmc_id = meta.get("pmc_id", "")
        title = meta.get("title", "")

        if pmc_id in seen:
            continue

        # Match against title or PMC ID
        if (query_lower in title.lower() or
                query_lower in pmc_id.lower()):
            seen.add(pmc_id)
            papers.append({
                **meta,
                "summary_text": doc,
            })

        if len(papers) >= limit:
            break

    return papers


def get_all_papers(limit: int = 500) -> list[dict]:
    """Return all papers from Chroma summary chunks."""
    col = _get_chroma_collection()
    total = col.count()
    if total == 0:
        return []

    results = col.get(
        where={"section": "__summary__"},
        include=["metadatas", "documents"],
        limit=total,
    )

    papers = []
    seen = set()
    for meta, doc in zip(results["metadatas"], results["documents"]):
        pmc_id = meta.get("pmc_id", "")
        if pmc_id in seen:
            continue
        seen.add(pmc_id)
        papers.append({**meta, "summary_text": doc})

    # Sort by year descending, then title
    papers.sort(key=lambda p: (p.get("year", "0"), p.get("title", "")), reverse=True)
    return papers[:limit]


# ── deep summary generation ──────────────────────────────────────────────────

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
    retriever = get_retriever()
    model_cfg = get_model_config()
    client = make_openrouter_client()

    chunks = retriever.retrieve_by_paper(pmc_id)
    if not chunks:
        yield "No chunks found for this paper."
        return

    # Assemble full text
    parts = []
    for c in chunks:
        label = c.section if not c.is_summary else "ABSTRACT/SUMMARY"
        parts.append(f"[{label}]\n{c.text.strip()}\n")

    paper_text = "\n---\n".join(parts)

    messages = [
        {"role": "system", "content": DEEP_SUMMARY_PROMPT},
        {"role": "user", "content": paper_text},
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


# ── score helpers ─────────────────────────────────────────────────────────────

def format_score(distance: float) -> int:
    """Convert cosine distance (0=identical, 2=opposite) to relevance %."""
    similarity = max(0.0, 1.0 - distance)
    return int(round(similarity * 100))


def highlight_terms(text: str, query: str) -> str:
    """Wrap query terms in the text with <mark> tags for highlighting."""
    words = set(query.lower().split())
    # Remove very short / common words
    words = {w for w in words if len(w) > 2}
    if not words:
        return text
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b',
        re.IGNORECASE,
    )
    return pattern.sub(r'**\1**', text)
