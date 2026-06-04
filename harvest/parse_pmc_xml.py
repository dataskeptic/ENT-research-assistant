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


def _clean_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _iter_text(element) -> str:
    """Recursively extract prose text, skipping noisy XML elements."""
    tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
    if tag in SKIP_TAGS:
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
    if key in _SECTION_ALIASES:
        return _SECTION_ALIASES[key]
    for alias, canonical in _SECTION_ALIASES.items():
        if alias in key:
            return canonical
    return raw_title.title()  # unknown: keep as-is, title-cased


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

    # Abstract
    for abstract_el in root.iter("abstract"):
        text = _clean_text(_iter_text(abstract_el))
        if len(text) > 50:
            sections.append({"section": "Abstract", "text": text, "order": order})
            order += 1
        break  # only first abstract

    # Body sections
    for body_el in root.iter("body"):
        for sec_el in body_el.iter("sec"):
            title_el = sec_el.find("title")
            raw_title = title_el.text.strip() if title_el is not None and title_el.text else "Body"
            canonical = _normalise_section_title(raw_title)

            if canonical is None:
                continue

            paragraphs = []
            for child in sec_el:
                child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if child_tag == "p":
                    p_text = _clean_text(_iter_text(child))
                    if p_text:
                        paragraphs.append(p_text)

            text = " ".join(paragraphs)
            if len(text) > 50:
                sections.append({"section": canonical, "text": text, "order": order})
                order += 1

    return sections


def extract_article_metadata(xml_path: str | Path) -> dict:
    """Extract article-level metadata from PMC XML."""
    root = etree.parse(str(xml_path)).getroot()

    title_el = root.find(".//article-title")
    title = title_el.text.strip() if title_el is not None and title_el.text else ""

    authors = []
    for contrib in root.iter("contrib"):
        if contrib.get("contrib-type") == "author":
            s_el = contrib.find(".//surname")
            g_el = contrib.find(".//given-names")
            surname = s_el.text.strip() if s_el is not None and s_el.text else ""
            given = g_el.text.strip() if g_el is not None and g_el.text else ""
            if surname:
                authors.append(f"{given} {surname}".strip())

    journal_el = root.find(".//journal-title")
    journal = journal_el.text.strip() if journal_el is not None and journal_el.text else ""

    year = ""
    for pub_date in root.iter("pub-date"):
        year_el = pub_date.find("year")
        if year_el is not None and year_el.text:
            year = year_el.text.strip()
            break

    doi = ""
    for article_id in root.iter("article-id"):
        if article_id.get("pub-id-type") == "doi" and article_id.text:
            doi = article_id.text.strip()
            break

    pmc_id = ""
    for article_id in root.iter("article-id"):
        if article_id.get("pub-id-type") == "pmc" and article_id.text:
            pmc_id = f"PMC{article_id.text.strip()}"
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
