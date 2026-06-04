"""Utility: Parse PMC OAI-PMH XML into clean structured text sections.

PMC XML is rich but noisy. This module strips all tags and extracts
the article body as named sections (Abstract, Introduction, Methods,
Results, Discussion, Conclusion) — ideal for RAG chunking because
you always know which section a chunk came from.

Usage:
    from harvest.parse_pmc_xml import parse_pmc_xml
    sections = parse_pmc_xml("data/fulltext/PMC12345.xml")
    # returns: [{"section": "Abstract", "text": "..."}, ...]

For the ingest pipeline, each section becomes an independent chunk
with section metadata attached.
"""

import re
from pathlib import Path
from lxml import etree


# Section title normalisation map
_SECTION_ALIASES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Introduction",
    "methods": "Methods",
    "materials and methods": "Methods",
    "methodology": "Methods",
    "results": "Results",
    "findings": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "summary": "Conclusion",
    "references": None,   # None = skip this section entirely
    "acknowledgements": None,
    "acknowledgments": None,
    "supplementary": None,
    "conflict of interest": None,
    "competing interests": None,
    "author contributions": None,
    "funding": None,
}

_WHITESPACE_RE = re.compile(r'\s+')


def _clean_text(text: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _iter_text(element) -> str:
    """Recursively extract all text content from an XML element,
    skipping table cells, figure captions, and formula elements
    which add noise without semantic value for RAG."""
    SKIP_TAGS = {
        "table", "table-wrap", "fig", "supplementary-material",
        "inline-formula", "disp-formula", "tex-math", "mml:math",
    }
    parts = []
    tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
    if tag in SKIP_TAGS:
        return ""
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_iter_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(p for p in parts if p.strip())


def _normalise_section_title(raw_title: str) -> str | None:
    """Map a raw section title to a canonical name, or None to skip."""
    key = raw_title.strip().lower()
    # exact match
    if key in _SECTION_ALIASES:
        return _SECTION_ALIASES[key]
    # partial match (e.g. 'patients and methods', 'results and discussion')
    for alias, canonical in _SECTION_ALIASES.items():
        if alias in key:
            return canonical
    return raw_title.title()  # unknown section: keep as-is, title-cased


def parse_pmc_xml(xml_path: str | Path) -> list[dict]:
    """Parse a PMC OAI XML file into a list of clean text sections.

    Returns:
        List of dicts with keys:
            section  (str)  : canonical section name
            text     (str)  : clean prose text, tags stripped
            order    (int)  : section order in the article
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"XML file not found: {path}")

    try:
        tree = etree.parse(str(path))
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Failed to parse XML {path.name}: {e}") from e

    root = tree.getroot()
    # Strip namespace prefixes for easier xpath
    # PMC OAI wraps content in <OAI-PMH><GetRecord><record><metadata><article>...
    ns = {"oai": "http://www.openarchives.org/OAI/2.0/"}

    sections: list[dict] = []
    order = 0

    # --- Abstract ---
    # Can appear as <abstract> directly under <article-meta> or wrapped
    for abstract_el in root.iter("abstract"):
        text = _clean_text(_iter_text(abstract_el))
        if len(text) > 50:  # ignore empty/minimal abstracts
            sections.append({"section": "Abstract", "text": text, "order": order})
            order += 1
        break  # only first abstract

    # --- Body sections ---
    # PMC body structure: <body><sec><title>Introduction</title><p>...</p></sec>
    for body_el in root.iter("body"):
        for sec_el in body_el.iter("sec"):
            # Get section title from first <title> child
            title_el = sec_el.find("title")
            raw_title = title_el.text.strip() if title_el is not None and title_el.text else "Body"
            canonical = _normalise_section_title(raw_title)

            if canonical is None:
                continue  # skip noise sections (references, acks, etc.)

            # Collect text from all <p> children (not nested <sec>, those recurse separately)
            paragraphs = []
            for p_el in sec_el:
                child_tag = p_el.tag.split("}")[-1] if "}" in p_el.tag else p_el.tag
                if child_tag == "p":
                    p_text = _clean_text(_iter_text(p_el))
                    if p_text:
                        paragraphs.append(p_text)

            text = " ".join(paragraphs)
            if len(text) > 50:
                sections.append({"section": canonical, "text": text, "order": order})
                order += 1

    return sections


def extract_article_metadata(xml_path: str | Path) -> dict:
    """Extract article-level metadata from PMC XML.

    Returns dict with: title, authors, journal, year, doi, pmc_id
    """
    path = Path(xml_path)
    tree = etree.parse(str(path))
    root = tree.getroot()

    def find_text(xpath: str, default: str = "") -> str:
        el = root.find(xpath)
        return el.text.strip() if el is not None and el.text else default

    # Title
    title = find_text(".//article-title")

    # Authors
    authors = []
    for contrib in root.iter("contrib"):
        if contrib.get("contrib-type") == "author":
            surname = find_text(".//surname") if contrib.find(".//surname") is not None else ""
            given = find_text(".//given-names") if contrib.find(".//given-names") is not None else ""
            # Use element-local find
            s_el = contrib.find(".//surname")
            g_el = contrib.find(".//given-names")
            surname = s_el.text.strip() if s_el is not None and s_el.text else ""
            given = g_el.text.strip() if g_el is not None and g_el.text else ""
            if surname:
                authors.append(f"{given} {surname}".strip())

    # Journal
    journal_el = root.find(".//journal-title")
    journal = journal_el.text.strip() if journal_el is not None and journal_el.text else ""

    # Year
    year = ""
    for pub_date in root.iter("pub-date"):
        year_el = pub_date.find("year")
        if year_el is not None and year_el.text:
            year = year_el.text.strip()
            break

    # DOI
    doi = ""
    for article_id in root.iter("article-id"):
        if article_id.get("pub-id-type") == "doi" and article_id.text:
            doi = article_id.text.strip()
            break

    # PMC ID
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


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m harvest.parse_pmc_xml <path_to_xml>")
        sys.exit(1)

    xml_file = sys.argv[1]
    meta = extract_article_metadata(xml_file)
    sections = parse_pmc_xml(xml_file)

    print(json.dumps({"metadata": meta, "sections": sections}, indent=2, ensure_ascii=False))
