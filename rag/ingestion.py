"""
rag/ingestion.py

Chunks parsed JSON papers and ingests them into a ChromaDB collection.

Chunking strategy — one chunk per logical unit, no artificial splitting:
  1. __summary__   — title + authors + journal + abstract (one per paper)
  2. <SectionName> — ONE chunk per section, exactly as parsed, regardless of length

Why no splitting?
  The parsed JSON already segments text by section (Introduction, Methods,
  Results, Discussion, etc.). Each section is a coherent semantic unit written
  by the authors. Splitting a section mid-sentence to hit an arbitrary token
  budget breaks that coherence and creates orphaned fragments that are harder
  to retrieve and harder for the LLM to use. The embedding model (Qwen3-8B)
  supports long inputs, so there is no need to impose a hard token ceiling here.
  The pipeline's context builder in pipeline.py already enforces a budget when
  assembling the final prompt sent to the LLM.

Usage:
    python -m rag.ingestion                  # ingest all of data/parsed/
    python -m rag.ingestion --limit 5        # smoke test with 5 papers
    python -m rag.ingestion --reset          # wipe collection first
"""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path
from typing import Generator

import chromadb
from openai import OpenAI

from rag.config import (
    PARSED_DIR, CHROMA_DIR,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_HEADERS,
    get_model_config, get_chunk_config, get_retriever_config,
)


# ── optional token counter (for stats only, not for splitting) ───────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _token_len(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def _token_len(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)


# ── chunker ───────────────────────────────────────────────────────────────────
def chunk_paper(
    data: dict,
    chunk_cfg,
    paper_id: str,
) -> Generator[tuple[str, dict], None, None]:
    """
    Yields (text, metadata) tuples for a single paper.

    Produces exactly:
      - 1 __summary__ chunk  (always)
      - 1 chunk per non-skipped, non-abstract section  (variable per paper)

    No sliding window, no sub-section splitting.
    """
    meta       = data.get("metadata", {})
    sections   = data.get("sections", [])
    references = data.get("references", [])

    doi = meta.get("doi", paper_id)

    base_meta = {
        "doi":             doi,
        "pmc_id":          meta.get("pmc_id", ""),
        "pmid":            str(meta.get("pmid", "")),
        "title":           meta.get("title", ""),
        "authors":         ", ".join(meta.get("authors", [])),
        "journal":         meta.get("journal", ""),
        "year":            str(meta.get("year", "")),
        "full_text":       str(meta.get("full_text_available", False)),
        # Structured references stored as JSON string — metadata only, never embedded
        "references_json": json.dumps(references, ensure_ascii=False),
    }

    # —— 1. Summary chunk (one per paper) —————————————————————————————
    abstract_text = next(
        (s["text"] for s in sections if s.get("section") == "Abstract"), ""
    )
    summary_text = (
        f"Title: {meta.get('title', '')}\n"
        f"Authors: {', '.join(meta.get('authors', []))}\n"
        f"Journal: {meta.get('journal', '')} ({meta.get('year', '')})\n"
        f"DOI: {meta.get('doi', '')}\n\n"
        f"Abstract:\n{abstract_text}"
    )
    yield summary_text, {
        **base_meta,
        "section":  "__summary__",
        "chunk_id": f"{doi}::__summary__",
        "order":    -1,
        "tokens":   _token_len(summary_text),
    }

    # —— 2. One chunk per section ——————————————————————————————————
    for sec in sections:
        sec_name       = sec.get("section", "").strip()
        sec_name_lower = sec_name.lower()

        # Skip boilerplate sections
        if sec_name_lower in chunk_cfg.skip_sections:
            continue
        # Abstract is already captured in the summary chunk above
        if sec_name_lower == "abstract":
            continue

        sec_text = sec.get("text", "").strip()
        if not sec_text:
            continue

        order = sec.get("order", 0)

        # Prefix with section label — improves embedding signal for section-level queries
        chunk_text = f"[{sec_name}]\n\n{sec_text}"

        yield chunk_text, {
            **base_meta,
            "section":  sec_name,
            "chunk_id": f"{doi}::{sec_name}::{order}",
            "order":    order,
            "tokens":   _token_len(chunk_text),
        }


# ── embedding client ───────────────────────────────────────────────────────────
def make_openrouter_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file or environment."
        )
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=OPENROUTER_HEADERS,
    )


def embed_batch(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """
    Embed a batch of texts via OpenRouter /embeddings.
    Returns one float vector per input text, in the same order.
    """
    response = client.embeddings.create(model=model, input=texts)
    return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]


# ── ingestor ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 8    # sections per embedding API call (sections can be long)


def ingest(
    parsed_dir: Path | None = None,
    chroma_dir: Path | None = None,
    reset: bool = False,
    limit: int = 0,
    batch_size: int = BATCH_SIZE,
) -> None:
    model_cfg  = get_model_config()
    chunk_cfg  = get_chunk_config()
    ret_cfg    = get_retriever_config()

    parsed_dir = parsed_dir or PARSED_DIR
    chroma_dir = chroma_dir or CHROMA_DIR
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client     = make_openrouter_client()
    chroma     = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            chroma.delete_collection(ret_cfg.collection_name)
            print(f"  Deleted existing collection '{ret_cfg.collection_name}'")
        except Exception:
            pass

    collection = chroma.get_or_create_collection(
        name=ret_cfg.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    json_files = sorted(parsed_dir.glob("*.json"))
    if limit:
        json_files = json_files[:limit]

    print(f"\nIngesting {len(json_files)} papers into '{ret_cfg.collection_name}'")
    print(f"  Embedding model  : {model_cfg.embedding}")
    print(f"  Chroma path      : {chroma_dir}")
    print(f"  Chunking strategy: 1 chunk per section (no splitting)")
    print(f"  Batch size       : {batch_size} chunks per API call")
    print()

    total_chunks = 0
    total_papers = 0

    batch_texts:  list[str]  = []
    batch_ids:    list[str]  = []
    batch_metas:  list[dict] = []

    def flush_batch() -> None:
        nonlocal total_chunks
        if not batch_texts:
            return
        vectors = embed_batch(client, batch_texts, model_cfg.embedding)
        collection.add(
            ids=batch_ids,
            embeddings=vectors,
            documents=batch_texts,
            metadatas=batch_metas,
        )
        total_chunks += len(batch_texts)
        batch_texts.clear()
        batch_ids.clear()
        batch_metas.clear()

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Skipping {json_path.name}: {exc}")
            continue

        paper_doi = data.get("metadata", {}).get("doi", "")
        if not paper_doi:
            print(f"  [SKIP] {json_path.name}: no DOI — not a paper")
            continue

        paper_chunks = 0
        paper_tokens = 0

        for text, meta in chunk_paper(data, chunk_cfg, paper_doi):
            chunk_id = meta["chunk_id"]

            # idempotent: skip already-ingested chunks
            if collection.get(ids=[chunk_id])["ids"]:
                continue

            batch_texts.append(text)
            batch_ids.append(chunk_id)
            batch_metas.append(meta)
            paper_chunks += 1
            paper_tokens += meta.get("tokens", 0)

            if len(batch_texts) >= batch_size:
                flush_batch()
                time.sleep(0.3)   # gentle rate-limit pause between batches

        flush_batch()
        total_papers += 1
        print(
            f"  [{total_papers:>4}/{len(json_files)}] "
            f"{paper_doi:<40} "
            f"+{paper_chunks} chunks  "
            f"({paper_tokens:,} tokens embedded)"
        )

    print(f"\n  Done. {total_papers} papers  |  {total_chunks} chunks ingested.")
    print(f"  Collection total : {collection.count()} vectors\n")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest parsed papers into ChromaDB")
    parser.add_argument("--reset",      action="store_true", help="Delete collection before ingesting")
    parser.add_argument("--limit",      type=int, default=0, help="Only ingest first N papers (0=all)")
    parser.add_argument("--parsed-dir", default=None,        help="Override data/parsed path")
    parser.add_argument("--chroma-dir", default=None,        help="Override data/chroma path")
    args = parser.parse_args()

    ingest(
        parsed_dir=Path(args.parsed_dir) if args.parsed_dir else None,
        chroma_dir=Path(args.chroma_dir) if args.chroma_dir else None,
        reset=args.reset,
        limit=args.limit,
    )
