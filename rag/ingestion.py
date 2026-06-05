"""
rag/ingestion.py

Chunks parsed JSON papers and ingests them into a ChromaDB collection.

Two chunk types per paper:
  1. __summary__  — title + authors + abstract (one per paper)
  2. <SectionName> — one chunk per section, split by sliding window if > max_tokens

Usage:
    python -m rag.ingestion                          # ingest all of data/parsed/
    python -m rag.ingestion --limit 5               # ingest first 5 papers (smoke test)
    python -m rag.ingestion --reset                 # wipe collection before ingesting
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


# ── token counting (mirrors corpus_stats.py) ─────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _token_len(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def _token_len(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)


# ── sliding window splitter ───────────────────────────────────────────────────
def _split_by_sentences(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Split text into windows of at most max_tokens with overlap_tokens overlap.
    Splits on sentence boundaries ('. ') where possible.
    """
    if _token_len(text) <= max_tokens:
        return [text]

    sentences = text.replace("\n", " ").split(". ")
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _token_len(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            windows.append(". ".join(current) + ".")
            # keep overlap: drop from front until we are within budget
            while current and current_tokens > overlap_tokens:
                removed = current.pop(0)
                current_tokens -= _token_len(removed)
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        windows.append(". ".join(current))

    return windows if windows else [text]


# ── chunker ────────────────────────────────────────────────────────────────────
def chunk_paper(data: dict, chunk_cfg, pmc_id: str) -> Generator[tuple[str, dict], None, None]:
    """
    Yields (text, metadata) tuples for a single paper.
    """
    meta = data.get("metadata", {})
    sections = data.get("sections", [])
    references = data.get("references", [])

    base_meta = {
        "pmc_id":    meta.get("pmc_id", pmc_id),
        "pmid":      str(meta.get("pmid", "")),
        "doi":       meta.get("doi", ""),
        "title":     meta.get("title", ""),
        "authors":   ", ".join(meta.get("authors", [])),
        "journal":   meta.get("journal", ""),
        "year":      str(meta.get("year", "")),
        "full_text": str(meta.get("full_text_available", False)),
        # Store structured references as JSON string — metadata only, never embedded
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
        f"Abstract: {abstract_text}"
    )
    yield summary_text, {
        **base_meta,
        "section": "__summary__",
        "chunk_id": f"{base_meta['pmc_id']}::__summary__::0",
        "order": -1,
        "window": 0,
    }

    # —— 2. Section chunks ————————————————————————————————————————
    for sec in sections:
        sec_name = sec.get("section", "")
        sec_name_lower = sec_name.strip().lower()

        if sec_name_lower in chunk_cfg.skip_sections:
            continue
        if sec_name == "Abstract":   # already in summary chunk
            continue

        sec_text = sec.get("text", "").strip()
        if not sec_text:
            continue

        order = sec.get("order", 0)
        # Prefix every chunk with section label for better embedding signal
        labeled = f"[{sec_name}]\n\n{sec_text}"

        windows = _split_by_sentences(labeled, chunk_cfg.max_tokens, chunk_cfg.overlap_tokens)
        for w_idx, window in enumerate(windows):
            yield window, {
                **base_meta,
                "section": sec_name,
                "chunk_id": f"{base_meta['pmc_id']}::{sec_name}::{order}::{w_idx}",
                "order": order,
                "window": w_idx,
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
    Returns a list of float vectors, one per input text.
    """
    response = client.embeddings.create(model=model, input=texts)
    # Sort by index to guarantee order matches input
    return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]


# ── ingestor ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 16   # chunks per embedding API call


def ingest(
    parsed_dir: Path | None = None,
    chroma_dir: Path | None = None,
    reset: bool = False,
    limit: int = 0,
    batch_size: int = BATCH_SIZE,
) -> None:
    model_cfg    = get_model_config()
    chunk_cfg    = get_chunk_config()
    ret_cfg      = get_retriever_config()

    parsed_dir   = parsed_dir or PARSED_DIR
    chroma_dir   = chroma_dir or CHROMA_DIR
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client       = make_openrouter_client()
    chroma       = chromadb.PersistentClient(path=str(chroma_dir))

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
    print(f"  Embedding model : {model_cfg.embedding}")
    print(f"  Chroma path     : {chroma_dir}")
    print(f"  Chunk max tokens: {chunk_cfg.max_tokens} / overlap: {chunk_cfg.overlap_tokens}")
    print()

    total_chunks = 0
    total_papers = 0

    # accumulate a batch before calling the API
    batch_texts:    list[str]  = []
    batch_ids:      list[str]  = []
    batch_metas:    list[dict] = []

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
            data   = json.loads(json_path.read_text(encoding="utf-8"))
            pmc_id = data.get("metadata", {}).get("pmc_id", json_path.stem)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Skipping {json_path.name}: {exc}")
            continue

        paper_chunks = 0
        for text, meta in chunk_paper(data, chunk_cfg, pmc_id):
            chunk_id = meta["chunk_id"]

            # skip if already ingested (idempotent re-runs)
            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                continue

            batch_texts.append(text)
            batch_ids.append(chunk_id)
            batch_metas.append(meta)
            paper_chunks += 1

            if len(batch_texts) >= batch_size:
                flush_batch()
                time.sleep(0.25)   # gentle rate-limit pause

        flush_batch()  # flush any remainder from this paper
        total_papers += 1
        print(f"  [{total_papers:>4}/{len(json_files)}] {pmc_id:<22}  +{paper_chunks} chunks")

    print(f"\n  Done. {total_papers} papers, {total_chunks} chunks ingested.")
    print(f"  Collection total: {collection.count()} vectors\n")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest parsed papers into ChromaDB")
    parser.add_argument("--reset",  action="store_true", help="Delete collection before ingesting")
    parser.add_argument("--limit",  type=int, default=0, help="Only ingest first N papers (0=all)")
    parser.add_argument("--parsed-dir", default=None,   help="Override data/parsed path")
    parser.add_argument("--chroma-dir", default=None,   help="Override data/chroma path")
    args = parser.parse_args()

    ingest(
        parsed_dir=Path(args.parsed_dir) if args.parsed_dir else None,
        chroma_dir=Path(args.chroma_dir) if args.chroma_dir else None,
        reset=args.reset,
        limit=args.limit,
    )
