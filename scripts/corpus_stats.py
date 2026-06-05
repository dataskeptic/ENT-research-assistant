#!/usr/bin/env python3
"""
corpus_stats.py

Counts papers and estimates token usage across data/parsed/*.json.

Token estimation uses the simple but accurate heuristic:
    tokens ≈ len(text) / 4
which closely matches tiktoken cl100k_base for English prose.
If tiktoken is installed, it is used instead for exact counts.

Usage:
    python scripts/corpus_stats.py
    python scripts/corpus_stats.py --parsed-dir data/parsed
    python scripts/corpus_stats.py --verbose   # show per-paper breakdown
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# ── optional exact tokeniser ──────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
    TOKEN_METHOD = "tiktoken cl100k_base (exact)"
except ImportError:
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)
    TOKEN_METHOD = "heuristic len/4 (approximate)"


# ── data model ────────────────────────────────────────────────────────────────
SKIP_SECTIONS = {"references", "reference list", "bibliography"}

@dataclass
class PaperStats:
    pmc_id: str
    title: str
    year: str
    full_text_available: bool
    section_count: int
    reference_count: int
    sections: list = field(default_factory=list)   # all section names as found
    token_abstract: int = 0
    token_body: int = 0        # sections that will be embedded (excl. abstract)
    token_abstract_only: int = 0  # same as token_abstract, alias for clarity
    token_references_text: int = 0  # raw ref-list text inside sections (skipped)
    token_ref_metadata: int = 0    # structured references[] array tokens
    token_total_embeddable: int = 0  # abstract + body  → goes to vector DB
    token_total: int = 0            # everything in the file


# ── helpers ───────────────────────────────────────────────────────────────────
def is_ref_section(name: str) -> bool:
    return name.strip().lower() in SKIP_SECTIONS


def analyse_paper(path: Path) -> Optional[PaperStats]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] Could not read {path.name}: {exc}")
        return None

    meta       = data.get("metadata", {})
    sections   = data.get("sections", [])
    references = data.get("references", [])

    pmc_id = meta.get("pmc_id", path.stem)
    title  = meta.get("title", "(no title)")[:80]
    year   = str(meta.get("year", "?"))
    fta    = meta.get("full_text_available", False)

    tok_abstract   = 0
    tok_body       = 0
    tok_ref_sec    = 0   # ref-list text inside sections[]
    section_names  = []

    for sec in sections:
        sec_text = sec.get("text", "")
        sec_name = sec.get("section", "?")
        t = count_tokens(sec_text)
        section_names.append(sec_name)

        if is_ref_section(sec_name):
            tok_ref_sec += t          # will NOT be embedded
        elif sec_name == "Abstract":
            tok_abstract += t         # will be embedded (summary chunk)
        else:
            tok_body += t             # will be embedded (section chunks)

    # structured references[] — stored as metadata only, never embedded
    tok_ref_meta = 0
    for ref in references:
        ref_text = " ".join(str(v) for v in ref.values() if isinstance(v, str))
        tok_ref_meta += count_tokens(ref_text)

    tok_embeddable = tok_abstract + tok_body
    tok_total      = tok_embeddable + tok_ref_sec + tok_ref_meta

    return PaperStats(
        pmc_id=pmc_id,
        title=title,
        year=year,
        full_text_available=fta,
        section_count=meta.get("section_count", len(sections)),
        reference_count=meta.get("reference_count", len(references)),
        sections=section_names,
        token_abstract=tok_abstract,
        token_body=tok_body,
        token_references_text=tok_ref_sec,
        token_ref_metadata=tok_ref_meta,
        token_total_embeddable=tok_embeddable,
        token_total=tok_total,
    )


def fmt(n: int) -> str:
    return f"{n:,}"

def pct(part: int, total: int) -> str:
    if total == 0:
        return "  0.0%"
    return f"{100 * part / total:5.1f}%"


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus statistics for data/parsed/")
    parser.add_argument(
        "--parsed-dir", default="data/parsed",
        help="Path to directory containing parsed JSON files (default: data/parsed)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-paper token breakdown sorted by size"
    )
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir)
    if not parsed_dir.exists():
        print(f"ERROR: directory not found: {parsed_dir}")
        return

    json_files = sorted(parsed_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {parsed_dir}")
        return

    papers: list = []
    for f in json_files:
        stats = analyse_paper(f)
        if stats:
            papers.append(stats)

    # ── aggregates ────────────────────────────────────────────────────────────
    total_papers         = len(papers)
    full_text_papers     = sum(1 for p in papers if p.full_text_available)
    abstract_only        = total_papers - full_text_papers

    tok_abstract         = sum(p.token_abstract for p in papers)
    tok_body             = sum(p.token_body for p in papers)
    tok_embeddable       = sum(p.token_total_embeddable for p in papers)
    tok_ref_sections     = sum(p.token_references_text for p in papers)
    tok_ref_metadata     = sum(p.token_ref_metadata for p in papers)
    tok_grand_total      = sum(p.token_total for p in papers)

    total_references     = sum(p.reference_count for p in papers)
    total_sections       = sum(p.section_count for p in papers)

    avg_embed  = tok_embeddable // total_papers if total_papers else 0
    max_paper  = max(papers, key=lambda p: p.token_total_embeddable)
    min_paper  = min(papers, key=lambda p: p.token_total_embeddable)

    section_freq: Counter = Counter()
    for p in papers:
        section_freq.update(p.sections)

    # ── verbose per-paper table ───────────────────────────────────────────────
    if args.verbose:
        col = "{:<22} {:>6} {:>8} {:>8} {:>9} {:>9}  {}"
        print("\n" + col.format(
            "PMC ID", "Year", "Sec-cnt", "Abstract", "Body", "Embeddable", "Title"
        ))
        print("-" * 115)
        for p in sorted(papers, key=lambda x: -x.token_total_embeddable):
            flag = "" if p.full_text_available else " ★abstract-only"
            print(col.format(
                p.pmc_id, p.year,
                p.section_count,
                fmt(p.token_abstract),
                fmt(p.token_body),
                fmt(p.token_total_embeddable),
                p.title[:55] + flag
            ))
        print()

    # ── summary ───────────────────────────────────────────────────────────────
    BAR = "═" * 60
    W   = 12   # right-align width for numbers

    print(f"\n{BAR}")
    print(f"  ETN CORPUS STATISTICS")
    print(f"  Token method : {TOKEN_METHOD}")
    print(BAR)
    print(f"  Papers total         : {fmt(total_papers)}")
    print(f"    Full text          : {fmt(full_text_papers)}")
    print(f"    Abstract only      : {fmt(abstract_only)}")
    print(f"  Sections (total)     : {fmt(total_sections)}")
    print(f"  References (total)   : {fmt(total_references)}")

    print(f"\n  ── What goes INTO the vector DB (embedded) ─────────")
    print(f"  Abstract chunks      : {fmt(tok_abstract):>{W}}   {pct(tok_abstract, tok_grand_total)}")
    print(f"  Body section chunks  : {fmt(tok_body):>{W}}   {pct(tok_body, tok_grand_total)}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  TOTAL embeddable     : {fmt(tok_embeddable):>{W}}   {pct(tok_embeddable, tok_grand_total)}")

    print(f"\n  ── What stays as METADATA only (not embedded) ──────")
    print(f"  Ref-list section txt : {fmt(tok_ref_sections):>{W}}   {pct(tok_ref_sections, tok_grand_total)}")
    print(f"  Structured refs[]    : {fmt(tok_ref_metadata):>{W}}   {pct(tok_ref_metadata, tok_grand_total)}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  TOTAL metadata       : {fmt(tok_ref_sections + tok_ref_metadata):>{W}}   {pct(tok_ref_sections + tok_ref_metadata, tok_grand_total)}")

    print(f"\n  ── Grand Total ──────────────────────────────────────")
    print(f"  All tokens in corpus : {fmt(tok_grand_total):>{W}}")
    print(f"  Avg embeddable/paper : {fmt(avg_embed):>{W}}")
    print(f"  Largest embeddable   : {max_paper.pmc_id}  ({fmt(max_paper.token_total_embeddable)} tokens)")
    print(f"  Smallest embeddable  : {min_paper.pmc_id}  ({fmt(min_paper.token_total_embeddable)} tokens)")

    print(f"\n  ── Section Name Frequency (top 15) ─────────────────")
    for sec_name, count in section_freq.most_common(15):
        flag = " ← skipped" if is_ref_section(sec_name) else ""
        bar  = "█" * min(count, 38)
        print(f"  {sec_name:<26} {count:>4}  {bar}{flag}")
    print(BAR)

    # cost estimates
    cost_small  = (tok_embeddable / 1_000_000) * 0.02   # text-embedding-3-small
    cost_large  = (tok_embeddable / 1_000_000) * 0.13   # text-embedding-3-large
    print(f"\n  Embedding cost estimate (embeddable tokens only):")
    print(f"  text-embedding-3-small  ($0.020/1M) :  ${cost_small:.4f}")
    print(f"  text-embedding-3-large  ($0.130/1M) :  ${cost_large:.4f}")
    print(BAR + "\n")


if __name__ == "__main__":
    main()
