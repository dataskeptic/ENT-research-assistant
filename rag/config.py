"""
rag/config.py

Single source of truth for all RAG configuration.
Override any value via environment variables or a .env file.
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field

# ── load .env if python-dotenv is available ───────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
PARSED_DIR  = ROOT_DIR / "data" / "parsed"
CHROMA_DIR  = ROOT_DIR / "data" / "chroma"


# ── OpenRouter ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# HTTP headers sent with every OpenRouter request (recommended by their docs)
OPENROUTER_HEADERS = {
    "HTTP-Referer": os.environ.get("APP_URL", "http://localhost"),
    "X-Title":      os.environ.get("APP_TITLE", "ETN Research Assistant"),
}


# ── Model identifiers ─────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # LLM used for answer synthesis
    llm: str = "nvidia/nemotron-3-ultra-550b-a55b:free"

    # Embedding model — must support OpenRouter /embeddings endpoint
    embedding: str = "qwen/qwen3-embedding-8b"

    # Max tokens the LLM should generate in a single response
    llm_max_tokens: int = 5048

    # Temperature for answer generation (0 = deterministic)
    llm_temperature: float = 0.1


# ── Chunking ──────────────────────────────────────────────────────────────────
@dataclass
class ChunkConfig:
    # Hard token limit per chunk before sliding-window split kicks in
    max_tokens: int = 512

    # Token overlap between consecutive windows of the same section
    overlap_tokens: int = 64

    # Section names to SKIP when chunking (not embedded, not stored)
    skip_sections: frozenset = field(default_factory=lambda: frozenset({
        "references",
        "reference list",
        "bibliography",
        "acknowledgements",
        "acknowledgments",
        "conflict of interest",
        "conflicts of interest",
        "funding",
        "author contributions",
        "supplementary",
        "supplementary material",
    }))


# ── Retrieval ─────────────────────────────────────────────────────────────────
@dataclass
class RetrieverConfig:
    # Number of chunks returned by vector search before re-ranking
    top_k: int = 8

    # After retrieval, always include the __summary__ chunk of each
    # retrieved paper (pulls in abstract + citation context)
    include_paper_summary: bool = True

    # Chroma collection name
    collection_name: str = "etn_papers"

    # Embedding dimension for qwen/qwen3-embedding-8b
    embedding_dim: int = 4096


# ── Generation / RAG prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an advanced ENT (Otorhinolaryngology) surgical research AI, specialized in the latest medical literature published in the last month.
Your purpose is to assist specialized surgeons with cutting-edge insights.
Answer questions using ONLY the provided context passages from peer-reviewed medical literature.
For every factual claim, cite the source paper using the format [Author Year] or [PMC ID].
If the context does not contain enough information to answer, say so explicitly — do not speculate.
Respond in clear, sophisticated clinical language appropriate for an expert specialist surgeon.
"""


# ── Convenience: build default instances ─────────────────────────────────────
def get_model_config() -> ModelConfig:
    return ModelConfig(
        llm=os.environ.get("RAG_LLM_MODEL", ModelConfig.llm),
        embedding=os.environ.get("RAG_EMBEDDING_MODEL", ModelConfig.embedding),
        llm_max_tokens=int(os.environ.get("RAG_LLM_MAX_TOKENS", ModelConfig.llm_max_tokens)),
        llm_temperature=float(os.environ.get("RAG_LLM_TEMPERATURE", ModelConfig.llm_temperature)),
    )

def get_chunk_config() -> ChunkConfig:
    return ChunkConfig(
        max_tokens=int(os.environ.get("RAG_CHUNK_MAX_TOKENS", ChunkConfig.max_tokens)),
        overlap_tokens=int(os.environ.get("RAG_CHUNK_OVERLAP", ChunkConfig.overlap_tokens)),
    )

def get_retriever_config() -> RetrieverConfig:
    return RetrieverConfig(
        top_k=int(os.environ.get("RAG_TOP_K", RetrieverConfig.top_k)),
        collection_name=os.environ.get("RAG_COLLECTION", RetrieverConfig.collection_name),
    )
