"""
rag/retriever.py

Retrieves relevant chunks from ChromaDB for a given query.

Strategy:
  1. Embed the query with the same model used during ingestion
  2. Vector search for top_k chunks
  3. Optionally expand results by pulling the __summary__ chunk of every
     retrieved paper (captures abstract + full citation context)
  4. Return a deduplicated, ranked list of RetrievedChunk objects

Usage (standalone smoke test):
    python -m rag.retriever "What are outcomes of TORS for oropharyngeal cancer?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from openai import OpenAI

from rag.config import (
    CHROMA_DIR,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_HEADERS,
    get_model_config, get_retriever_config,
)
from rag.ingestion import make_openrouter_client, embed_batch


# ── result model ────────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    chunk_id:  str
    doi:       str
    section:   str
    text:      str
    score:     float          # cosine distance (lower = more similar)
    metadata:  dict

    @property
    def is_summary(self) -> bool:
        return self.section == "__summary__"

    @property
    def citation_label(self) -> str:
        """Short citation string for use in LLM context, e.g. '[Smith 2024]'."""
        authors_raw = self.metadata.get("authors", "")
        first_author = authors_raw.split(",")[0].split()[-1] if authors_raw else "Unknown"
        year = self.metadata.get("year", "")
        return f"[{first_author} {year}]"

    @property
    def references(self) -> list[dict]:
        """Deserialise the stored references_json metadata field."""
        raw = self.metadata.get("references_json", "[]")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []


# ── retriever class ────────────────────────────────────────────────────────────
class Retriever:
    """
    Wraps ChromaDB + OpenRouter embeddings for RAG retrieval.

    Instantiate once and reuse across multiple queries.
    """

    def __init__(
        self,
        chroma_dir: Path | None = None,
    ) -> None:
        self._model_cfg = get_model_config()
        self._ret_cfg   = get_retriever_config()
        self._client    = make_openrouter_client()

        chroma_dir = chroma_dir or CHROMA_DIR
        self._chroma = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = self._chroma.get_or_create_collection(
            name=self._ret_cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── public API ───────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for *query*.

        Args:
            query:  Natural language question.
            top_k:  Override config top_k for this call.
            where:  Optional ChromaDB metadata filter, e.g.
                    {"year": {"$gte": "2022"}} or {"section": "Results"}

        Returns:
            Deduplicated list of RetrievedChunk, best matches first.
        """
        k = top_k or self._ret_cfg.top_k

        # 1. embed query
        [query_vector] = embed_batch(self._client, [query], self._model_cfg.embedding)

        # 2. vector search
        query_kwargs: dict = dict(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            chunks.append(RetrievedChunk(
                chunk_id=cid,
                doi=meta.get("doi", ""),
                section=meta.get("section", ""),
                text=doc,
                score=dist,
                metadata=meta,
            ))

        # 3. expand with paper summary chunks
        if self._ret_cfg.include_paper_summary:
            chunks = self._expand_with_summaries(chunks, seen_ids)

        return chunks

    def retrieve_by_paper(
        self,
        doi: str,
    ) -> list[RetrievedChunk]:
        """
        Return ALL chunks for a single paper (useful for full-paper Q&A).
        """
        results = self._collection.get(
            where={"doi": doi},
            include=["documents", "metadatas"],
        )
        chunks = []
        for doc, meta, cid in zip(
            results["documents"],
            results["metadatas"],
            results["ids"],
        ):
            chunks.append(RetrievedChunk(
                chunk_id=cid,
                doi=meta.get("doi", ""),
                section=meta.get("section", ""),
                text=doc,
                score=0.0,
                metadata=meta,
            ))
        return sorted(chunks, key=lambda c: (c.metadata.get("order", 0), c.metadata.get("window", 0)))

    # ── internals ────────────────────────────────────────────────────────────────
    def _expand_with_summaries(
        self,
        chunks: list[RetrievedChunk],
        seen_ids: set[str],
    ) -> list[RetrievedChunk]:
        """
        For every unique paper in *chunks*, fetch its __summary__ chunk
        and prepend it to the result list (if not already present).
        """
        dois = {c.doi for c in chunks if not c.is_summary}
        extra: list[RetrievedChunk] = []

        for doi in dois:
            summary_id = f"{doi}::__summary__"
            if summary_id in seen_ids:
                continue
            result = self._collection.get(
                ids=[summary_id],
                include=["documents", "metadatas"],
            )
            if not result["ids"]:
                continue
            doc  = result["documents"][0]
            meta = result["metadatas"][0]
            extra.append(RetrievedChunk(
                chunk_id=summary_id,
                doi=doi,
                section="__summary__",
                text=doc,
                score=0.0,  # summary injected, not ranked
                metadata=meta,
            ))
            seen_ids.add(summary_id)

        # summaries first, then ranked section chunks
        return extra + chunks


# ── CLI smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "TORS outcomes oropharyngeal cancer"
    retriever = Retriever()
    results = retriever.retrieve(query)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} chunks\n")
    for i, chunk in enumerate(results, 1):
        flag = "[SUMMARY]" if chunk.is_summary else f"[{chunk.section}]"
        print(f"{i:>2}. {flag:<16} {chunk.citation_label:<18} score={chunk.score:.4f}")
        print(f"    {chunk.text[:160].strip()}...\n")
