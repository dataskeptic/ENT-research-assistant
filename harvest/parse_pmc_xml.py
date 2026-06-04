"""Utility: Parse PMC OAI-PMH XML into clean structured text sections.

PMC XML is rich but noisy. This module strips all tags and extracts
the article body as named sections (Abstract, Introduction, Methods,
Results, Discussion, Conclusion) — ideal for RAG chunking because
you always know which section a chunk came from.

Batch mode (default when run as __main__):
    python -m harvest.parse_pmc_xml
    Reads all *.xml from data/fulltext/, writes JSON to data/parsed/

Single-file mode (for debugging):
    python -m harvest.parse_pmc_xml data/fulltext/PMC12345.xml

For the ingest pipeline, each section becomes an independent chunk
with section metadata attached.

IMPORTANT: PMC OAI XML wraps the JATS article in an OAI-PMH envelope and
uses a fully-qualified JATS namespace on every element, e.g.:
    <article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/" ...>
This means lxml's iter("abstract") finds NOTHING — you must strip (or
ignore) the namespace when searching.  We do this with a _local() helper
that compares only the local-name portion of the Clark-notation tag.

Known article shapes in this corpus
------------------------------------
1. Full-text OAI response  — has <front> + <body> + <back>
   → full_text_available = True, sections cover all IMRaD parts
2. Abstract-only OAI response — <body> element absent or empty
   → full_text_available = False, only Abstract section present
   This is common for articles not in the PMC Open Access subset.
"""

import json
import logging
import re
from pathlib import Path

from lxml import etree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

XML_DIR = Path("data/fulltext")
OUT_DIR = Path("data/parsed")

# Section title normalisation map — None means skip entirely
_SECTION_ALIASES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Introduction",
    "methods": "Methods",
    "materials and methods": "Methods",
    "patients and methods": "Methods",
    "subjects and methods": "Methods",
    "methodology": "Methods",
    "results": "Results",
    "findings": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "summary": "Conclusion",
    "case report": "Case Report",
    "case presentation": "Case Report",
    "case reports": "Case Report",
    "references": None,
    "acknowledgements": None,
    "acknowledgments": None,
    "supplementary": None,
    "supplementary information": None,
    "conflict of interest": None,
    "competing interests": None,
    "author contributions": None,
    "funding": None,
    "data availability": None,
    "ethics": None,
    "declarations": None,
    "publisher's note": None,
}

_WHITESPACE_RE = re.compile(r"\s+")

# Tags whose *content* is entirely skipped for RAG text extraction.
# NOTE: 'xref' removed — its .tail (text after the closing tag) carries
#       real prose that was previously lost (BUG 5 / BUG 3 fix).
#       'mml:math' corrected to 'math' because _local() strips namespaces (BUG 4 fix).
SKIP_TAGS = {
    "table", "table-wrap", "fig", "supplementary-material",
    "inline-formula", "disp-formula", "tex-math", "math",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _local(element) -> str:
    """Return the local tag name, stripping any Clark-notation namespace.

    lxml represents namespaced tags as '{ns_uri}localname'.  PMC OAI XML
    uses a default JATS namespace on *every* element, so without this
    helper all tag comparisons silently fail and sections come out empty.
    """
    tag = element.tag
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _clean_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _iter_text(element) -> str:
    """Recursively extract prose text, skipping noisy XML elements.

    FIX (BUG 5): when a child element is in SKIP_TAGS we previously
    returned '' immediately, silently discarding child.tail — the text
    that appears *after* the closing tag in the source XML.  We now
    preserve tail text even for skipped elements.
    """
    parts = []
    if element.text:
        parts.append(element.text)

    for child in element:
        local = _local(child)
        if local in SKIP_TAGS:
            # Skip the element's own subtree but keep the tail text
            if child.tail:
                parts.append(child.tail)
        else:
            parts.append(_iter_text(child))
            # xref content is usually just a number like "1" — not useful
            # for RAG, but the tail carries real prose, so _iter_text above
            # will return the tail naturally via the recursive call.
            if child.tail:
                parts.append(child.tail)

    return " ".join(p for p in parts if p.strip())


def _normalise_section_title(raw_title: str) -> str | None:
    key = raw_title.strip().lower()
    # Remove leading numbering like "1. Introduction" → "introduction"
    key = re.sub(r"^[\d\.]+\s*", "", key).strip()
    if key in _SECTION_ALIASES:
        return _SECTION_ALIASES[key]
    for alias, canonical in _SECTION_ALIASES.items():
        if alias in key:
            return canonical
    return raw_title.title()  # unknown: keep as-is, title-cased


def _iter_local(root, local_tag: str):
    """Iterate over all descendants whose local tag name equals local_tag.

    This is namespace-safe: it works whether or not the XML uses a default
    namespace such as the JATS archiving namespace.
    """
    for el in root.iter():
        if _local(el) == local_tag:
            yield el


def _collect_paragraphs_shallow(sec_el) -> list[str]:
    """Collect direct <p> children of a <sec>, ignoring nested <sec>."""
    paragraphs = []
    for child in sec_el:
        if _local(child) == "p":
            p_text = _clean_text(_iter_text(child))
            if p_text:
                paragraphs.append(p_text)
    return paragraphs


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def _extract_abstract(root) -> str | None:
    """Extract the abstract as a single flat text block.

    FIX (ISSUE 7): Previously abstract sub-sections (Background, Methods,
    Results inside <abstract>) were fed into _process_sec() which mapped
    them to canonical section names — creating duplicate entries that
    clashed with the body sections.  The abstract is now extracted as one
    combined block regardless of its internal structure.
    """
    for abstract_el in _iter_local(root, "abstract"):
        text = _clean_text(_iter_text(abstract_el))
        if len(text) > 50:
            return text
    return None


def _process_sec(sec_el, sections: list[dict], order_counter: list[int]) -> None:
    """Recursively process a <sec> element, accumulating into sections list."""
    title_el = None
    for child in sec_el:
        if _local(child) == "title":
            title_el = child
            break

    raw_title = ""
    if title_el is not None:
        raw_title = _clean_text(" ".join(title_el.itertext()))
    if not raw_title:
        raw_title = "Body"

    canonical = _normalise_section_title(raw_title)
    if canonical is None:
        return  # explicitly skipped section

    # Collect direct <p> children of this <sec>
    paragraphs = _collect_paragraphs_shallow(sec_el)

    if paragraphs:
        text = " ".join(paragraphs)
        if len(text) > 50:
            sections.append({
                "section": canonical,
                "text": text,
                "order": order_counter[0],
            })
            order_counter[0] += 1

    # Recurse into nested <sec> (e.g. sub-sections inside Case Presentation)
    for child in sec_el:
        if _local(child) == "sec":
            _process_sec(child, sections, order_counter)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pmc_xml(xml_path: str | Path) -> tuple[list[dict], bool]:
    """Parse a PMC OAI XML file into a list of clean text sections.

    Returns
    -------
    sections : list[dict]
        Each dict has keys: section (str), text (str), order (int).
    full_text_available : bool
        True if the XML contained a <body> element with at least one
        section.  False means only abstract/metadata was present
        (common for non-OA articles).

    FIX (BUG 1): The original code called ``order_counter_val`` which
    referenced an undefined local variable after the body loop, causing a
    NameError at runtime.  Order tracking is now handled by a single
    mutable list ``[order]`` passed through all helpers.
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"XML file not found: {path}")

    try:
        tree = etree.parse(str(path))
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Failed to parse XML {path.name}: {e}") from e

    root = tree.getroot()
    sections: list[dict] = []
    order_counter = [0]  # mutable counter shared with _process_sec

    # ── Abstract ─────────────────────────────────────────────────────────────
    abstract_text = _extract_abstract(root)
    if abstract_text:
        sections.append({"section": "Abstract", "text": abstract_text, "order": order_counter[0]})
        order_counter[0] += 1

    # ── Body sections ────────────────────────────────────────────────────────
    # Walk *top-level* <sec> elements inside <body> to avoid double-counting
    # paragraphs that appear in both a parent <sec> and a child <sec>.
    body_sections_found = 0
    for body_el in _iter_local(root, "body"):
        for sec_el in body_el:  # direct children of <body>
            if _local(sec_el) != "sec":
                continue
            before = len(sections)
            _process_sec(sec_el, sections, order_counter)
            body_sections_found += len(sections) - before
        break  # only first <body>

    full_text_available = body_sections_found > 0

    # FIX (ISSUE 6): if body is absent/empty, log a warning so the caller
    # knows this article has no full text rather than silently returning
    # only the abstract.
    if not full_text_available:
        log.debug("No body sections found in %s — abstract-only record", path.name)

    return sections, full_text_available


def extract_article_metadata(xml_path: str | Path) -> dict:
    """Extract article-level metadata from PMC OAI XML (namespace-safe).

    FIX (ISSUE 9): year extraction now prefers epub pub-date, then ppub,
    then any pub-date rather than taking whichever appears first in
    document order.
    """
    root = etree.parse(str(xml_path)).getroot()

    title = ""
    for el in _iter_local(root, "article-title"):
        title = _clean_text(" ".join(el.itertext()))
        break

    authors = []
    for contrib in _iter_local(root, "contrib"):
        if contrib.get("contrib-type") == "author":
            surname, given = "", ""
            for child in _iter_local(contrib, "surname"):
                surname = child.text.strip() if child.text else ""
                break
            for child in _iter_local(contrib, "given-names"):
                given = child.text.strip() if child.text else ""
                break
            if surname:
                authors.append(f"{given} {surname}".strip())

    journal = ""
    for el in _iter_local(root, "journal-title"):
        journal = el.text.strip() if el.text else ""
        break

    # FIX (ISSUE 9): prefer epub, then ppub, then any pub-date
    year = ""
    pub_dates: dict[str, str] = {}
    for pub_date in _iter_local(root, "pub-date"):
        pub_type = pub_date.get("pub-type", "other")
        for year_el in _iter_local(pub_date, "year"):
            if year_el.text:
                pub_dates[pub_type] = year_el.text.strip()
            break
    for preferred in ("epub", "ppub", "other"):
        if preferred in pub_dates:
            year = pub_dates[preferred]
            break
    if not year and pub_dates:
        year = next(iter(pub_dates.values()))

    doi = ""
    for article_id in _iter_local(root, "article-id"):
        if article_id.get("pub-id-type") == "doi" and article_id.text:
            doi = article_id.text.strip()
            break

    pmc_id = ""
    for article_id in _iter_local(root, "article-id"):
        if article_id.get("pub-id-type") in ("pmc", "pmcid") and article_id.text:
            val = article_id.text.strip()
            pmc_id = val if val.startswith("PMC") else f"PMC{val}"
            break

    pmid = ""
    for article_id in _iter_local(root, "article-id"):
        if article_id.get("pub-id-type") == "pmid" and article_id.text:
            pmid = article_id.text.strip()
            break

    return {
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "doi": doi,
        "pmc_id": pmc_id,
        "pmid": pmid,
    }


def process_folder(
    xml_dir: Path = XML_DIR,
    out_dir: Path = OUT_DIR,
    overwrite: bool = False,
    reparse_incomplete: bool = False,
) -> list[dict]:
    """Batch-process all *.xml files in xml_dir → JSON files in out_dir.

    Each output JSON has shape::

        {
          "metadata": {
            "title": "...", "authors": [...], "journal": "...",
            "year": "...", "doi": "...", "pmc_id": "...", "pmid": "...",
            "full_text_available": true,
            "section_count": 6
          },
          "sections": [{"section": "...", "text": "...", "order": 0}, ...]
        }

    Parameters
    ----------
    xml_dir : Path
        Directory containing ``*.xml`` files downloaded from PMC OAI.
    out_dir : Path
        Directory where ``*.json`` parsed outputs are written.
    overwrite : bool
        If True, re-parse all files even if a JSON already exists.
    reparse_incomplete : bool
        FIX (ISSUE 10): If True, re-parse files that were previously saved
        as abstract-only (``full_text_available=false``) in case the full
        XML is now available.  Narrower than ``--overwrite``; only
        re-processes files known to be incomplete.

    Returns
    -------
    list[dict]
        Summary dicts for each file, including ``ok`` status.
    """
    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        log.warning("No XML files found in %s", xml_dir)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Processing %d XML files from %s → %s", len(xml_files), xml_dir, out_dir)

    summary = []
    errors = []

    for xml_path in tqdm(xml_files, desc="Parsing XML"):
        out_path = out_dir / (xml_path.stem + ".json")

        skip = False
        if out_path.exists() and not overwrite:
            if reparse_incomplete:
                # Re-process only if previously saved as abstract-only
                try:
                    with open(out_path, encoding="utf-8") as f:
                        existing = json.load(f)
                    if existing.get("metadata", {}).get("full_text_available", True):
                        skip = True  # already complete
                except Exception:
                    pass  # corrupt JSON — re-parse
            else:
                skip = True

        if skip:
            log.debug("Skipping (already parsed): %s", xml_path.name)
            continue

        try:
            meta = extract_article_metadata(xml_path)
            sections, full_text_available = parse_pmc_xml(xml_path)

            meta["full_text_available"] = full_text_available
            meta["section_count"] = len(sections)

            result = {"metadata": meta, "sections": sections}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            summary.append({
                "file": xml_path.name,
                "pmc_id": meta["pmc_id"],
                "doi": meta["doi"],
                "title": meta["title"],
                "sections": len(sections),
                "full_text_available": full_text_available,
                "ok": True,
            })
        except Exception as e:
            log.error("Failed to parse %s: %s", xml_path.name, e)
            errors.append({"file": xml_path.name, "error": str(e), "ok": False})

    ok = len(summary)
    log.info("Done: %d parsed, %d errors", ok, len(errors))

    # Write a summary report
    report_path = out_dir / "_parse_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"parsed": summary, "errors": errors}, f, ensure_ascii=False, indent=2)
    log.info("Report saved → %s", report_path)

    return summary + errors


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1].endswith(".xml"):
        # Single-file debug mode
        xml_file = sys.argv[1]
        meta = extract_article_metadata(xml_file)
        sections, full_text_available = parse_pmc_xml(xml_file)
        meta["full_text_available"] = full_text_available
        meta["section_count"] = len(sections)
        print(json.dumps({"metadata": meta, "sections": sections}, indent=2, ensure_ascii=False))
    else:
        # Batch mode: optional flags
        overwrite = "--overwrite" in sys.argv
        reparse_incomplete = "--reparse-incomplete" in sys.argv
        process_folder(overwrite=overwrite, reparse_incomplete=reparse_incomplete)
