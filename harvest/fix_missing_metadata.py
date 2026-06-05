"""One-shot post-processor: fix already-parsed JSON files with missing metadata.

This script does NOT re-run Docling. It reads the existing JSON files in
data/parsed/ and applies the same _promote_first_header_section logic to
any file where metadata fields are empty but the data is sitting in
sections[0] (the Springer/Elsevier SECTION_HEADER-as-title pattern).

Usage:
    python -m harvest.fix_missing_metadata           # dry-run, shows what would change
    python -m harvest.fix_missing_metadata --apply   # write fixes in-place
"""

import json
import re
import sys
from pathlib import Path

OUT_DIR = Path("data/parsed")

_MIDDOT_SEP_RE = re.compile(r"\s*\u00b7\s*")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_HEADER_BODY_RE = re.compile(
    r"(?:Received:|Accepted:|\u00a9\s*The Author|\u00a9\s*Springer|Published online)",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")

KNOWN_SECTIONS = {
    "Abstract", "Introduction", "Methods", "Results",
    "Discussion", "Conclusion", "Case Report", "Body",
}


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _parse_middot_authors(text: str) -> list[str]:
    parts = _MIDDOT_SEP_RE.split(text)
    authors = []
    for part in parts:
        name = re.sub(r"[\d,\*\u2020\u2021\u00a7]+$", "", part).strip()
        if not name or _HEADER_BODY_RE.search(name):
            break
        if " " in name and 3 < len(name) < 60:
            authors.append(name)
    return authors


def fix_record(data: dict) -> tuple[dict, list[str]]:
    """Apply metadata fixes to a parsed JSON record. Returns (fixed_data, changes)."""
    changes: list[str] = []
    meta = data.get("metadata", {})
    sections = data.get("sections", [])

    if not sections:
        return data, changes

    first = sections[0]
    first_heading = first.get("section", "")
    first_text = first.get("text", "")

    # Only act on sections that look like misclassified article headers
    if first_heading in KNOWN_SECTIONS:
        return data, changes

    is_header_body = "\u00b7" in first_text or _HEADER_BODY_RE.search(first_text)
    if not is_header_body:
        return data, changes

    # Fix title
    if not meta.get("title") or meta["title"] == first_heading.replace(" ", "_") or \
            re.match(r"^[\d_/\.]+$", meta.get("title", "")):
        old = meta.get("title", "")
        meta["title"] = first_heading
        changes.append(f"title: {repr(old)} → {repr(first_heading)[:80]}")

    # Fix authors
    if not meta.get("authors") and "\u00b7" in first_text:
        authors = _parse_middot_authors(first_text)
        if authors:
            meta["authors"] = authors
            changes.append(f"authors: [] → {authors}")

    # Fix year
    if not meta.get("year"):
        pub_year_m = re.search(
            r"(?:published|accepted|received|online)[^\d]{0,20}((?:19|20)\d{2})",
            first_text, re.IGNORECASE,
        )
        if pub_year_m:
            meta["year"] = pub_year_m.group(1)
            changes.append(f"year: → {meta['year']}")

    if changes:
        # Remove the spurious first section and re-number
        new_sections = sections[1:]
        for i, sec in enumerate(new_sections):
            sec["order"] = i
        meta["section_count"] = len(new_sections)
        data["metadata"] = meta
        data["sections"] = new_sections

    return data, changes


def main(apply: bool = False) -> None:
    json_files = sorted(OUT_DIR.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("_")]

    total_fixed = 0
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ERROR reading {path.name}: {e}")
            continue

        fixed_data, changes = fix_record(data)
        if not changes:
            continue

        print(f"\n{path.name}")
        for c in changes:
            print(f"  {c}")

        if apply:
            path.write_text(json.dumps(fixed_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✓ written")
        else:
            print(f"  (dry-run — pass --apply to write)")

        total_fixed += 1

    print(f"\n{'Fixed' if apply else 'Would fix'} {total_fixed} file(s).")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
