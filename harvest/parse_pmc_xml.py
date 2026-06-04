"""Utility: Parse PMC OAI-PMH XML into clean structured text sections.

PMC XML is rich but noisy. This module strips all tags and extracts
the article body as named sections (Abstract, Introduction, Methods,
Results, Discussion, Conclusion) - ideal for RAG chunking because
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
ignore) the namespace when searching.  We do this with a _tag() helper
that compares only the local-name portion of the Clark-notation tag.
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
    "conflict of interest": None,
    "competing interests": None,
    "author contributions": None,
    "funding": None,
    "data availability": None,
    "ethics": None,
}

_WHITESPACE_RE = re.compile(r"\s+")

SKIP_TAGS = {
    "table", "table-wrap", "fig", "supplementary-material",
    "inline-formula", "disp-formula", "tex-math", "mml:math",
    "xref",  # citation markers like [1], [2] — pure noise in RAG
}


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
    """Recursively extract prose text, skipping noisy XML elements."""
    if _local(element) in SKIP_TAGS:
        return ""
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_iter_text(child))
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


def parse_pmc_xml(xml_path: str | Path) -> list[dict]:
    """Parse a PMC OAI XML file into a list of clean text sections.

    Returns list of dicts: {section, text, order}
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
    order = 0

    # ── Abstract ─────────────────────────────────────────────────────────────
    for abstract_el in _iter_local(root, "abstract"):
        text = _clean_text(_iter_text(abstract_el))
        if len(text) > 50:
            sections.append({"section": "Abstract", "text": text, "order": order})
            order += 1
        break  # only first abstract

    # ── Body sections ────────────────────────────────────────────────────────
    # We walk *top-level* <sec> elements inside <body> to avoid double-counting
    # paragraphs that appear in both a parent <sec> and a child <sec>.
    for body_el in _iter_local(root, "body"):
        for sec_el in body_el:  # direct children of <body>
            if _local(sec_el) != "sec":
                continue
            _process_sec(sec_el, sections, order_counter=[order])
        order = order_counter_val(sections)

    return sections


def order_counter_val(sections: list[dict]) -> int:
    return sections[-1]["order"] + 1 if sections else 0


def _process_sec(sec_el, sections: list[dict], order_counter: list[int]) -> None:
    """Recursively process a <sec> element, accumulating into sections list."""
    title_el = None
    for child in sec_el:
        if _local(child) == "title":
            title_el = child
            break

    raw_title = ""
    if title_el is not None:
        # title_el.itertext() spans mixed content (e.g. <italic> inside title)
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


def extract_article_metadata(xml_path: str | Path) -> dict:
    """Extract article-level metadata from PMC OAI XML (namespace-safe)."""
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

    year = ""
    for pub_date in _iter_local(root, "pub-date"):
        for year_el in _iter_local(pub_date, "year"):
            year = year_el.text.strip() if year_el.text else ""
            break
        if year:
            break

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

    return {
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "doi": doi,
        "pmc_id": pmc_id,
    }


def process_folder(
    xml_dir: Path = XML_DIR,
    out_dir: Path = OUT_DIR,
    overwrite: bool = False,
) -> list[dict]:
    """Batch-process all *.xml files in xml_dir → JSON files in out_dir.

    Each output JSON has shape:
        {"metadata": {...}, "sections": [{"section", "text", "order"}, ...]}

    Returns list of summary dicts for logging/reporting.
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
        if out_path.exists() and not overwrite:
            log.debug("Skipping (already parsed): %s", xml_path.name)
            continue

        try:
            meta = extract_article_metadata(xml_path)
            sections = parse_pmc_xml(xml_path)

            result = {"metadata": meta, "sections": sections}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            summary.append({
                "file": xml_path.name,
                "pmc_id": meta["pmc_id"],
                "doi": meta["doi"],
                "title": meta["title"],
                "sections": len(sections),
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
        sections = parse_pmc_xml(xml_file)
        print(json.dumps({"metadata": meta, "sections": sections}, indent=2, ensure_ascii=False))
    else:
        # Batch mode: optional --overwrite flag
        overwrite = "--overwrite" in sys.argv
        process_folder(overwrite=overwrite)
