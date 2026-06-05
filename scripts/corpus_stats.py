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
from dataclasses import dataclass
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
@dataclass
class PaperStats:
    pmc_id: str
    title: str
    year: str
    full_text_available: bool
    section_count: int
    reference_count: int
    sections: list          # section names present
    token_abstract: int = 0
    token_sections: int = 0  # all sections including abstract
    token_references: int = 0
    token_total: int = 0


# ── core logic ────────────────────────────────────────────────────────────────
def analyse_paper(path: Path) -> Optional[PaperStats]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] Could not read {path.name}: {exc}")
        return None

    meta = data.get("metadata", {})
    sections = data.get("sections", [])
    references = data.get("references", [])

    pmc_id = meta.get("pmc_id", path.stem)
    title  = meta.get("title", "(no title)")[:80]
    year   = str(meta.get("year", "?"))
    fta    = meta.get("full_text_available", False)

    tok_abstract = 0
    tok_sections = 0
    section_names = []

    for sec in sections:
        sec_text = sec.get("text", "")
        t = count_tokens(sec_text)
        tok_sections += t
        section_names.append(sec.get("section", "?"))
        if sec.get("section") == "Abstract":
            tok_abstract = t

    tok_refs = 0
    for ref in references:
        ref_text = " ".join(str(v) for v in ref.values() if isinstance(v, str))
        tok_refs += count_tokens(ref_text)

    tok_total = tok_sections + tok_refs

    return PaperStats(
        pmc_id=pmc_id,
        title=title,
        year=year,
        full_text_available=fta,
        section_count=meta.get("section_count", len(sections)),
        reference_count=meta.get("reference_count", len(references)),
        sections=section_names,
        token_abstract=tok_abstract,
        token_sections=tok_sections,
        token_references=tok_refs,
        token_total=tok_total,
    )


def fmt(n: int) -> str:
    return f"{n:,}"


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
    total_papers       = len(papers)
    full_text_papers   = sum(1 for p in papers if p.full_text_available)
    abstract_only      = total_papers - full_text_papers

    total_tokens       = sum(p.token_total for p in papers)
    total_tok_sections = sum(p.token_sections for p in papers)
    total_tok_refs     = sum(p.token_references for p in papers)
    total_references   = sum(p.reference_count for p in papers)
    total_sections     = sum(p.section_count for p in papers)

    avg_tokens = total_tokens // total_papers if total_papers else 0
    max_paper  = max(papers, key=lambda p: p.token_total)
    min_paper  = min(papers, key=lambda p: p.token_total)

    section_freq: Counter = Counter()
    for p in papers:
        section_freq.update(p.sections)

    # ── verbose per-paper table ───────────────────────────────────────────────
    if args.verbose:
        col = "{:<22} {:>6} {:>8} {:>8} {:>8} {:>8}  {}"
        print("\n" + col.format(
            "PMC ID", "Year", "Sections", "Tok-Sec", "Tok-Ref", "Total", "Title"
        ))
        print("-" * 110)
        for p in sorted(papers, key=lambda x: -x.token_total):
            fta_label = "" if p.full_text_available else " [abstract-only]"
            print(col.format(
                p.pmc_id, p.year,
                p.section_count,
                fmt(p.token_sections),
                fmt(p.token_references),
                fmt(p.token_total),
                p.title[:55] + fta_label
            ))
        print()

    # ── summary ───────────────────────────────────────────────────────────────
    BAR = "═" * 56
    print(f"\n{BAR}")
    print(f"  ETN CORPUS STATISTICS")
    print(f"  Token method : {TOKEN_METHOD}")
    print(BAR)
    print(f"  Papers total       : {fmt(total_papers)}")
    print(f"    Full text        : {fmt(full_text_papers)}")
    print(f"    Abstract only    : {fmt(abstract_only)}")
    print(f"  Sections total     : {fmt(total_sections)}")
    print(f"  References total   : {fmt(total_references)}")
    print()
    print(f"  ── Token Counts ───────────────────────────────────")
    print(f"  Sections (embeddable text) : {fmt(total_tok_sections):>10}")
    print(f"  References (metadata text) : {fmt(total_tok_refs):>10}")
    print(f"  Grand total                : {fmt(total_tokens):>10}")
    print()
    print(f"  Avg tokens / paper   : {fmt(avg_tokens)}")
    print(f"  Largest paper        : {max_paper.pmc_id}  ({fmt(max_paper.token_total)} tokens)")
    print(f"  Smallest paper       : {min_paper.pmc_id}  ({fmt(min_paper.token_total)} tokens)")
    print()
    print(f"  ── Top Section Names ───────────────────────────────")
    for sec_name, count in section_freq.most_common(12):
        bar = "█" * min(count, 40)
        print(f"  {sec_name:<22} {count:>4}  {bar}")
    print(BAR)

    # rough cost estimate (OpenAI text-embedding-3-small = $0.02 / 1M tokens)
    cost_usd = (total_tok_sections / 1_000_000) * 0.02
    print(f"\n  Estimated embedding cost (text-embedding-3-small):")
    print(f"  ${cost_usd:.4f} USD  (sections only, $0.02 / 1M tokens)")
    print(BAR + "\n")


if __name__ == "__main__":
    main()
