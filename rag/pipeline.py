"""
rag/pipeline.py

Orchestrates retrieval + answer synthesis.

Flow:
  1. Retriever fetches top_k chunks from ChromaDB
  2. Context is assembled (summaries first, then ranked sections)
  3. OpenRouter LLM generates a citation-backed answer
  4. Returns RAGResponse with answer text + source chunks

Usage:
    python -m rag.pipeline "What are the recurrence rates after TORS?"
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

from openai import OpenAI

from rag.config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_HEADERS,
    SYSTEM_PROMPT,
    get_model_config, get_retriever_config,
)
from rag.ingestion import make_openrouter_client
from rag.retriever import Retriever, RetrievedChunk


# ── result model ──────────────────────────────────────────────────────────────
@dataclass
class RAGResponse:
    query:        str
    answer:       str
    sources:      list[RetrievedChunk] = field(default_factory=list)
    model:        str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def format_sources(self) -> str:
        """Human-readable source list (for CLI / API display)."""
        seen: set[str] = set()
        lines = ["\nSources:"]
        for chunk in self.sources:
            if chunk.is_summary:
                continue
            label = chunk.citation_label
            if label in seen:
                continue
            seen.add(label)
            title   = chunk.metadata.get("title", "")[:70]
            journal = chunk.metadata.get("journal", "")
            doi     = chunk.metadata.get("doi", "")
            lines.append(
                f"  {label}  {title}\n"
                f"           {journal}  |  DOI:{doi}"
            )
        return "\n".join(lines)


# ── context builder ───────────────────────────────────────────────────────────
CONTEXT_HEADER = """The following passages are excerpts from peer-reviewed ENT medical \
literature. Use them to answer the question.

"""
MAX_CONTEXT_TOKENS = 6000   # conservative budget for the context block


def _build_context(chunks: list[RetrievedChunk], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """
    Assemble a context block from retrieved chunks.
    Summaries are placed first; section chunks follow in score order.
    Stops adding when the running token count approaches max_tokens.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        count_tokens = lambda t: len(enc.encode(t))  # noqa: E731
    except ImportError:
        count_tokens = lambda t: max(1, len(t) // 4)  # noqa: E731

    parts: list[str] = [CONTEXT_HEADER]
    running = count_tokens(CONTEXT_HEADER)

    for chunk in chunks:
        label = chunk.citation_label
        section = "" if chunk.is_summary else f" [{chunk.section}]"
        block = (
            f"--- {label}{section} ---\n"
            f"{chunk.text.strip()}\n\n"
        )
        block_tokens = count_tokens(block)
        if running + block_tokens > max_tokens:
            break
        parts.append(block)
        running += block_tokens

    return "".join(parts)


# ── pipeline ──────────────────────────────────────────────────────────────────
class RAGPipeline:
    """
    Full RAG pipeline: retrieve → build context → generate answer.

    Instantiate once; call ask() as many times as needed.
    """

    def __init__(self) -> None:
        self._model_cfg = get_model_config()
        self._retriever = Retriever()
        self._llm_client = make_openrouter_client()

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> RAGResponse:
        """
        Ask a question and return a citation-backed answer.

        Args:
            query:  Natural language question.
            top_k:  Override default number of retrieved chunks.
            where:  Optional ChromaDB metadata filter for scoped search,
                    e.g. {"year": {"$gte": "2020"}} or {"journal": "Laryngoscope"}

        Returns:
            RAGResponse with .answer and .sources populated.
        """
        # 1. retrieve
        chunks = self._retriever.retrieve(query, top_k=top_k, where=where)

        if not chunks:
            return RAGResponse(
                query=query,
                answer="No relevant passages found in the literature database for this query.",
                model=self._model_cfg.llm,
            )

        # 2. build context
        context = _build_context(chunks)

        # 3. generate
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": f"{context}\n\nQuestion: {query}"},
        ]

        completion = self._llm_client.chat.completions.create(
            model=self._model_cfg.llm,
            messages=messages,
            max_tokens=self._model_cfg.llm_max_tokens,
            temperature=self._model_cfg.llm_temperature,
        )

        answer = completion.choices[0].message.content or ""
        usage  = completion.usage

        return RAGResponse(
            query=query,
            answer=answer,
            sources=chunks,
            model=self._model_cfg.llm,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What are the surgical outcomes after TORS?"

    print(f"\nQuery: {query}\n")
    pipeline = RAGPipeline()
    response = pipeline.ask(query)

    print("Answer")
    print("═" * 72)
    print(textwrap.fill(response.answer, width=80))
    print(response.format_sources())
    print(f"\nModel: {response.model}  "
          f"| tokens in: {response.input_tokens}  out: {response.output_tokens}")
