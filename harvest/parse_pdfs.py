"""Utility: Parse PDF research papers into the same JSON schema as parse_pmc_xml.py.

Uses Docling (IBM Research) which understands two-column journal layouts,
section hierarchy, tables, and reference blocks natively.

Batch mode (default when run as __main__):
    python -m harvest.parse_pdfs
    Reads all *.pdf from data/pdfs/, writes JSON to data/parsed/

Single-file mode (for debugging):
    python -m harvest.parse_pdfs data/pdfs/some_paper.pdf

Behaviour:
  - Skips any PDF whose output JSON already exists in data/parsed/
    (same skip-if-exists policy as parse_pmc_xml.py)
  - Use --overwrite to force re-parse everything
  - Use --reparse-incomplete to re-parse PDFs previously saved as
    abstract-only (full_text_available=false)

Output JSON shape (identical to parse_pmc_xml.py output)::

    {
      "metadata": {
        "title": "...", "authors": [...], "journal": "...",
        "year": "...", "doi": "...", "pmc_id": "...", "pmid": "...",
        "full_text_available": true,
        "section_count": 6,
        "reference_count": 28
      },
      "sections": [...],
      "references": [...]
    }

Docling dependency:
    pip install docling>=2.0.0
"""

import json
import logging
import re
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/parsed")

# ---------------------------------------------------------------------------
# Section title normalisation
# ---------------------------------------------------------------------------

_SECTION_ALIASES: dict[str, str | None] = {
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
    "abbreviations": None,
}

_WHITESPACE_RE = re.compile(r"\s+")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s,;\"\'>\]]+", re.IGNORECASE)
_PMC_RE = re.compile(r"\bPMC\d+\b", re.IGNORECASE)
_PMID_RE = re.compile(r"\bPMID[:\s]+(\d{7,9})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MIDDOT_SEP_RE = re.compile(r"\s*\u00b7\s*")
_AUTHOR_SEP_RE = re.compile(r"[,;]\s*")
_AUTHOR_TOKEN_RE = re.compile(
    r"^[A-Z\u00c0-\u00d6][a-zA-Z\u00c0-\u00f6\u00d8-\u00ff'\-]{1,25}"
    r"(?:\s+[A-Z]{1,4}\.?|\s+[A-Z\u00c0-\u00d6][a-zA-Z\u00c0-\u00f6\u00d8-\u00ff'\-]{1,20})?$"
)
_JOURNAL_KEYWORDS = re.compile(
    r"\b(journal|otolaryngol|head\s+neck|laryngoscope|rhinolog|allerg|audiol|"
    r"otolog|surgery|medicine|annals|archives|review|lancet|jama|bmj|nejm|"
    r"plos|nature|science|cell|cochrane)\b",
    re.IGNORECASE,
)
_HEADER_BODY_RE = re.compile(
    r"(?:Received:|Accepted:|\u00a9\s*The Author|\u00a9\s*Springer|Published online)",
    re.IGNORECASE,
)

# Regex that matches a PDF stem that encodes a DOI:
#   10_1007_s00405-026-10300-1  →  10.1007/s00405-026-10300-1
_DOI_STEM_RE = re.compile(r"^(10)_(\d{4,9})_(.+)$")

_FAKE_TITLE_WORDS = {
    "laryngology", "otology", "rhinology", "audiology", "oncology",
    "neurology", "cardiology", "radiology", "pathology", "immunology",
    "original article", "original research", "research article",
    "review article", "systematic review", "meta-analysis",
    "case report", "case series", "letter", "editorial", "commentary",
    "correspondence", "brief communication", "short communication",
    "clinical study", "clinical trial", "retrospective study",
    "prospective study", "cohort study", "pilot study",
    "head and neck", "head & neck",
}


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _is_fake_title(text: str) -> bool:
    stripped = text.strip()
    if stripped.lower() in _FAKE_TITLE_WORDS:
        return True
    words = stripped.split()
    if len(words) <= 3 and len(stripped) < 40:
        if stripped == stripped.upper() and stripped.replace(" ", "").isalpha():
            return True
        if stripped.lower() in _FAKE_TITLE_WORDS:
            return True
    return False


def _normalise_section_title(raw: str) -> str | None:
    key = raw.strip().lower()
    key = re.sub(r"^[\d\.]+\s*", "", key).strip()
    if key in _SECTION_ALIASES:
        return _SECTION_ALIASES[key]
    for alias, canonical in _SECTION_ALIASES.items():
        if alias in key:
            return canonical
    return raw.strip().title()


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


def _looks_like_author_list(text: str) -> list[str]:
    if len(text) > 400 or len(text) < 5:
        return []
    tokens = [t.strip() for t in _AUTHOR_SEP_RE.split(text) if t.strip()]
    if len(tokens) < 2:
        return []
    clean_tokens = [re.sub(r"[\d,\*\u2020\u2021\u00a7]+$", "", t).strip() for t in tokens]
    if all(_AUTHOR_TOKEN_RE.match(t) for t in clean_tokens if t):
        return [t for t in clean_tokens if t]
    return []


# ---------------------------------------------------------------------------
# DOI-from-filename
# ---------------------------------------------------------------------------

def _doi_from_stem(stem: str) -> str:
    """Convert a DOI-encoded filename stem to a real DOI string.

    Springer/Elsevier downloads encode DOIs as:
        10_1007_s00405-026-10300-1  →  10.1007/s00405-026-10300-1
    """
    m = _DOI_STEM_RE.match(stem)
    if not m:
        return ""
    return f"{m.group(1)}.{m.group(2)}/{m.group(3)}"


# ---------------------------------------------------------------------------
# CrossRef lookup
# ---------------------------------------------------------------------------

def _crossref_lookup(doi: str) -> dict:
    """Query CrossRef REST API for title/journal/year/authors.

    Silent fail on network errors. Returns empty dict if nothing found.
    """
    result: dict = {}
    if not doi:
        return result
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ETN-RAG/1.0 (research tool)"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        item = data.get("message", {})

        titles = item.get("title", [])
        if titles:
            result["title"] = _clean(titles[0])

        container = item.get("container-title", [])
        if container:
            result["journal"] = _clean(container[0])

        for date_key in ("published-print", "published-online", "created"):
            date_parts = item.get(date_key, {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                result["year"] = str(date_parts[0][0])
                break

        authors_raw = item.get("author", [])
        if authors_raw:
            names = []
            for a in authors_raw:
                given = a.get("given", "")
                family = a.get("family", "")
                if family:
                    names.append(f"{given} {family}".strip() if given else family)
            if names:
                result["authors"] = names

    except Exception as exc:
        log.debug("CrossRef lookup failed for %s: %s", doi, exc)

    return result


# ---------------------------------------------------------------------------
# PubMed lookup  (title + PMID + PMC ID via DOI)
# ---------------------------------------------------------------------------

def _pubmed_lookup(doi: str) -> dict:
    """Query NCBI E-utilities to retrieve title, PMID, and PMC ID by DOI.

    Uses two calls:
      1. esearch: DOI  →  PMID list
      2. esummary: PMID →  title, source, pmc_id

    Silent fail on network errors. Returns empty dict if nothing found.
    """
    result: dict = {}
    if not doi:
        return result

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    headers = {"User-Agent": "ETN-RAG/1.0 (research tool)"}

    try:
        # Step 1: esearch – resolve DOI to PMID
        esearch_params = urllib.parse.urlencode({
            "db": "pubmed",
            "term": f"{doi}[doi]",
            "retmode": "json",
            "retmax": "1",
        })
        req = urllib.request.Request(f"{base}/esearch.fcgi?{esearch_params}", headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            esearch = json.loads(resp.read().decode("utf-8"))

        ids = esearch.get("esearchresult", {}).get("idlist", [])
        if not ids:
            log.debug("PubMed: no PMID for DOI %s", doi)
            return result

        pmid = ids[0]
        result["pmid"] = pmid
        log.debug("PubMed: DOI %s → PMID %s", doi, pmid)

        # Step 2: esummary – fetch title + PMC ID
        esummary_params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
        })
        req = urllib.request.Request(f"{base}/esummary.fcgi?{esummary_params}", headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            esummary = json.loads(resp.read().decode("utf-8"))

        summary = esummary.get("result", {}).get(pmid, {})

        title = summary.get("title", "").strip()
        if title:
            result["title"] = _clean(title)

        # PMC ID is stored under "articleids" as type "pmc"
        for aid in summary.get("articleids", []):
            if aid.get("idtype") == "pmc":
                pmc_val = str(aid.get("value", "")).strip()
                if pmc_val:
                    result["pmc_id"] = pmc_val if pmc_val.upper().startswith("PMC") else f"PMC{pmc_val}"
                break

    except Exception as exc:
        log.debug("PubMed lookup failed for %s: %s", doi, exc)

    return result


# ---------------------------------------------------------------------------
# Abstract title fallback
# ---------------------------------------------------------------------------

def _title_from_abstract(sections: list[dict]) -> str:
    """Extract a surrogate title from the Abstract when all API lookups fail.

    For structured abstracts (Background / Objective / ... labels), the text
    after the first label is used.  For unstructured abstracts, the first
    meaningful sentence is returned.

    Returns empty string when nothing usable is found.
    """
    _LABEL_RE = re.compile(
        r"^(Background|Objective|Purpose|Aim|Introduction|Context|Summary)[s]?"
        r"[\s:.]+ ?",
        re.IGNORECASE,
    )
    for sec in sections:
        if sec.get("section") != "Abstract":
            continue
        text = sec.get("text", "").strip()
        if not text:
            continue

        # For structured abstracts: extract the content after the first label
        m = _LABEL_RE.match(text)
        if m:
            remainder = text[m.end():].strip()
            # Take only up to the next label boundary
            next_label = _LABEL_RE.search(remainder)
            if next_label:
                remainder = remainder[:next_label.start()].strip()
            # Take the first sentence of the remainder
            sentences = re.split(r"(?<=[.?!])\s+", remainder)
            for sent in sentences:
                sent = sent.strip()
                if 30 < len(sent) < 300:
                    return sent

        # For unstructured abstracts: first sentence that isn't a label
        sentences = re.split(r"(?<=[.?!])\s+", text)
        for sent in sentences:
            sent = sent.strip()
            if _LABEL_RE.match(sent):
                continue
            if 30 < len(sent) < 300:
                return sent

    return ""


# ---------------------------------------------------------------------------
# Docling converter
# ---------------------------------------------------------------------------

def _build_converter():
    try:
        from docling.document_converter import DocumentConverter
        return DocumentConverter()
    except Exception as e:
        raise ImportError(
            "Docling is required for PDF parsing. Install it with:\n"
            "    pip install docling>=2.0.0"
        ) from e


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(doc) -> dict:
    meta: dict = {
        "title": "",
        "authors": [],
        "journal": "",
        "year": "",
        "doi": "",
        "pmc_id": "",
        "pmid": "",
    }

    if hasattr(doc, "metadata") and doc.metadata:
        md = doc.metadata
        if hasattr(md, "title") and md.title:
            candidate = _clean(str(md.title))
            if not _is_fake_title(candidate):
                meta["title"] = candidate
        if hasattr(md, "authors") and md.authors:
            raw_authors = md.authors
            if isinstance(raw_authors, list):
                meta["authors"] = [_clean(str(a)) for a in raw_authors if str(a).strip()]
            elif isinstance(raw_authors, str):
                meta["authors"] = [a.strip() for a in raw_authors.split(";") if a.strip()]

    early_title_items: list[str] = []
    early_heading_items: list[str] = []
    early_text_items: list[str] = []
    full_head = ""
    past_abstract = False

    try:
        for item, _ in list(doc.iterate_items())[:30]:
            if not hasattr(item, "text") or not item.text:
                continue
            item_text = _clean(item.text)
            if not item_text:
                continue

            label_str = str(getattr(item, "label", "")).upper().split(".")[-1]

            if ("SECTION_HEADER" in label_str or "HEADING" in label_str) and \
                    "ABSTRACT" in item_text.upper():
                past_abstract = True

            full_head += " " + item_text

            if "TITLE" in label_str:
                early_title_items.append(item_text)
            elif not past_abstract and (
                "SECTION_HEADER" in label_str or "HEADING" in label_str
            ):
                early_heading_items.append(item_text)
            elif not past_abstract and (
                "TEXT" in label_str
                or "PARAGRAPH" in label_str
                or "LIST_ITEM" in label_str
            ):
                early_text_items.append(item_text)

            if len(full_head) > 4000:
                break
    except Exception:
        pass

    if not meta["title"]:
        for candidate in early_title_items:
            if not _is_fake_title(candidate):
                meta["title"] = candidate
                break

    if not meta["title"]:
        if hasattr(doc, "name") and doc.name:
            candidate = _clean(doc.name)
            if "/" not in candidate \
                    and not candidate.lower().endswith(".pdf") \
                    and not _is_fake_title(candidate):
                meta["title"] = candidate

    if not meta["title"]:
        for candidate in early_heading_items:
            if not _is_fake_title(candidate):
                meta["title"] = candidate
                break

    if not meta["authors"]:
        for text in early_text_items:
            if "\u00b7" in text:
                authors = _parse_middot_authors(text)
                if authors:
                    meta["authors"] = authors
                    break

    if not meta["authors"]:
        for text in early_text_items:
            authors = _looks_like_author_list(text)
            if authors:
                meta["authors"] = authors
                break

    if not meta["journal"]:
        for text in early_text_items:
            if _JOURNAL_KEYWORDS.search(text) and len(text) < 150:
                candidate = re.sub(r"\s*[\(\[]\s*(19|20)\d{2}.*$", "", text).strip()
                candidate = re.sub(r"\s*\d+\s*[:\(].*$", "", candidate).strip()
                if candidate:
                    meta["journal"] = candidate
                    break

    if not meta["year"]:
        pub_year_m = re.search(
            r"(?:published|accepted|received|online)[^\d]{0,20}((?:19|20)\d{2})",
            full_head, re.IGNORECASE,
        )
        if pub_year_m:
            meta["year"] = pub_year_m.group(1)
        else:
            y_m = _YEAR_RE.search(full_head)
            if y_m:
                meta["year"] = y_m.group(0)

    if not meta["doi"]:
        m = _DOI_RE.search(full_head)
        if m:
            meta["doi"] = m.group(0).rstrip(".")

    if not meta["pmc_id"]:
        m = _PMC_RE.search(full_head)
        if m:
            meta["pmc_id"] = m.group(0).upper()

    if not meta["pmid"]:
        m = _PMID_RE.search(full_head)
        if m:
            meta["pmid"] = m.group(1)

    return meta


# ---------------------------------------------------------------------------
# Post-processing: promote spurious first section to metadata
# ---------------------------------------------------------------------------

def _promote_first_header_section(meta: dict, sections: list[dict]) -> tuple[dict, list[dict]]:
    if not sections:
        return meta, sections

    first = sections[0]
    first_heading = first.get("section", "")
    first_text = first.get("text", "")

    canonical = _normalise_section_title(first_heading)
    known_sections = {
        "Abstract", "Introduction", "Methods", "Results",
        "Discussion", "Conclusion", "Case Report", "Body",
    }
    if canonical in known_sections:
        return meta, sections

    if _is_fake_title(first_heading):
        return meta, sections

    is_header_body = (
        "\u00b7" in first_text
        or _HEADER_BODY_RE.search(first_text)
    )
    if not is_header_body:
        return meta, sections

    if not meta.get("title"):
        meta["title"] = first_heading

    if not meta.get("authors") and "\u00b7" in first_text:
        authors = _parse_middot_authors(first_text)
        if authors:
            meta["authors"] = authors

    if not meta.get("year"):
        pub_year_m = re.search(
            r"(?:published|accepted|received|online)[^\d]{0,20}((?:19|20)\d{2})",
            first_text, re.IGNORECASE,
        )
        if pub_year_m:
            meta["year"] = pub_year_m.group(1)

    sections = sections[1:]
    for i, sec in enumerate(sections):
        sec["order"] = i

    return meta, sections


# ---------------------------------------------------------------------------
# Metadata enrichment: DOI-from-stem → CrossRef → PubMed → Abstract fallback
# ---------------------------------------------------------------------------

def _enrich_metadata(meta: dict, pdf_path: Path, sections: list[dict]) -> dict:
    """Fill empty fields after all Docling extraction and header promotion.

    Step 1 – DOI from filename stem (Springer/Elsevier pattern).
    Step 2 – CrossRef: fills title / journal / year / authors.
    Step 3 – PubMed: fills title (if still empty) + pmid + pmc_id.
              Called whenever doi is known and (title or pmid or pmc_id) is empty.
    Step 4 – Abstract text: last-resort title only, never overwrites API results.

    Never overwrites a field that already has a value.
    """
    # Step 1: DOI from filename
    if not meta.get("doi"):
        doi_candidate = _doi_from_stem(pdf_path.stem)
        if doi_candidate:
            meta["doi"] = doi_candidate
            log.debug("DOI from filename: %s", doi_candidate)

    doi = meta.get("doi", "")

    # Step 2: CrossRef
    if doi and not all([meta.get("title"), meta.get("journal"), meta.get("year")]):
        cr = _crossref_lookup(doi)
        if cr:
            log.debug("CrossRef filled: %s", list(cr.keys()))
        if not meta.get("title") and cr.get("title"):
            meta["title"] = cr["title"]
        if not meta.get("journal") and cr.get("journal"):
            meta["journal"] = cr["journal"]
        if not meta.get("year") and cr.get("year"):
            meta["year"] = cr["year"]
        if not meta.get("authors") and cr.get("authors"):
            meta["authors"] = cr["authors"]

    # Step 3: PubMed (title + pmid + pmc_id)
    needs_pubmed = doi and (
        not meta.get("title")
        or not meta.get("pmid")
        or not meta.get("pmc_id")
    )
    if needs_pubmed:
        pm = _pubmed_lookup(doi)
        if pm:
            log.debug("PubMed filled: %s", list(pm.keys()))
        if not meta.get("title") and pm.get("title"):
            meta["title"] = pm["title"]
        if not meta.get("pmid") and pm.get("pmid"):
            meta["pmid"] = pm["pmid"]
        if not meta.get("pmc_id") and pm.get("pmc_id"):
            meta["pmc_id"] = pm["pmc_id"]

    # Step 4: Abstract text – last resort title
    if not meta.get("title"):
        abs_title = _title_from_abstract(sections)
        if abs_title:
            meta["title"] = abs_title
            log.debug("Title from Abstract: %s", abs_title[:80])

    return meta


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def _extract_sections(doc) -> tuple[list[dict], bool]:
    sections: list[dict] = []
    current_heading = "Body"
    current_paragraphs: list[str] = []
    order_counter = [0]

    _SKIP_LABELS = {
        "TABLE", "FIGURE", "PICTURE", "FORMULA",
        "PAGE_HEADER", "PAGE_FOOTER", "FOOTNOTE", "CAPTION",
    }

    def _flush(heading: str, paragraphs: list[str]) -> None:
        if not paragraphs:
            return
        text = _clean(" ".join(paragraphs))
        if len(text) < 50:
            return
        canonical = _normalise_section_title(heading)
        if canonical is None:
            return
        sections.append({"section": canonical, "text": text, "order": order_counter[0]})
        order_counter[0] += 1

    try:
        items = list(doc.iterate_items())
    except Exception as e:
        log.warning("Could not iterate document items: %s", e)
        return [], False

    for item, _level in items:
        label_str = ""
        if hasattr(item, "label"):
            label_str = str(item.label).upper().split(".")[-1]

        if not label_str:
            continue
        if any(skip in label_str for skip in _SKIP_LABELS):
            continue

        item_text = _clean(item.text) if hasattr(item, "text") and item.text else ""
        if not item_text:
            continue

        if "SECTION_HEADER" in label_str or "HEADING" in label_str:
            _flush(current_heading, current_paragraphs)
            current_heading = item_text
            current_paragraphs = []
        elif "TITLE" in label_str:
            continue
        elif "TEXT" in label_str or "LIST_ITEM" in label_str or "PARAGRAPH" in label_str:
            current_paragraphs.append(item_text)

    _flush(current_heading, current_paragraphs)

    full_text_available = (
        len(sections) > 1
        and any(s["section"] != "Abstract" for s in sections)
    )
    return sections, full_text_available


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

def _extract_references(doc) -> list[dict]:
    refs: list[dict] = []

    if hasattr(doc, "references") and doc.references:
        for i, ref in enumerate(doc.references, start=1):
            entry: dict = {"ref_id": f"R{i}", "label": f"{i}."}
            for field in ("title", "journal", "doi", "pmid", "year", "volume", "issue"):
                val = getattr(ref, field, None)
                if val:
                    entry[field] = _clean(str(val))
            if hasattr(ref, "pages") and ref.pages:
                pages = str(ref.pages).strip()
                if "-" in pages:
                    fp, lp = pages.split("-", 1)
                    entry["fpage"] = fp.strip()
                    entry["lpage"] = lp.strip()
                else:
                    entry["fpage"] = pages
            if hasattr(ref, "pmc_id") and ref.pmc_id:
                val = str(ref.pmc_id).strip()
                entry["pmc_id"] = val if val.startswith("PMC") else f"PMC{val}"
            if hasattr(ref, "authors") and ref.authors:
                raw = ref.authors
                entry["authors"] = (
                    [_clean(str(a)) for a in raw if str(a).strip()]
                    if isinstance(raw, list)
                    else [a.strip() for a in raw.split(";") if a.strip()]
                )
            if len(entry) > 2:
                refs.append(entry)
        if refs:
            return refs

    in_references = False
    ref_lines: list[str] = []

    try:
        items = list(doc.iterate_items())
    except Exception:
        return []

    for item, _level in items:
        label_str = str(getattr(item, "label", "")).upper().split(".")[-1]
        item_text = _clean(item.text) if hasattr(item, "text") and item.text else ""
        if not item_text:
            continue
        if "SECTION_HEADER" in label_str or "HEADING" in label_str:
            key = re.sub(r"^[\d\.]+\s*", "", item_text.lower()).strip()
            in_references = key in ("references", "bibliography", "works cited")
            continue
        if in_references and ("TEXT" in label_str or "LIST_ITEM" in label_str):
            ref_lines.append(item_text)

    for i, line in enumerate(ref_lines, start=1):
        if len(line) < 20:
            continue
        entry: dict = {"ref_id": f"R{i}", "label": f"{i}.", "raw_citation": line}
        m = _DOI_RE.search(line)
        if m:
            entry["doi"] = m.group(0).rstrip(".")
        m = re.search(r"\b(19|20)\d{2}\b", line)
        if m:
            entry["year"] = m.group(0)
        refs.append(entry)

    return refs


# ---------------------------------------------------------------------------
# Output stem resolution
# ---------------------------------------------------------------------------

def _output_stem(pdf_path: Path, doc=None) -> str:
    if doc is not None:
        try:
            head = ""
            for item, _ in list(doc.iterate_items())[:30]:
                if hasattr(item, "text") and item.text:
                    head += " " + item.text
                if len(head) > 2000:
                    break
            m = _PMC_RE.search(head)
            if m:
                return m.group(0).upper()
        except Exception:
            pass
    return pdf_path.stem


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str | Path) -> dict:
    """Parse a single PDF → structured JSON dict."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    converter = _build_converter()
    log.info("Converting %s with Docling...", pdf_path.name)
    result = converter.convert(str(pdf_path))
    doc = result.document

    meta = _extract_metadata(doc)
    sections, full_text_available = _extract_sections(doc)
    references = _extract_references(doc)

    meta, sections = _promote_first_header_section(meta, sections)
    meta = _enrich_metadata(meta, pdf_path, sections)

    if not meta["title"]:
        meta["title"] = pdf_path.stem

    meta["full_text_available"] = full_text_available
    meta["section_count"] = len(sections)
    meta["reference_count"] = len(references)

    return {"metadata": meta, "sections": sections, "references": references}


def process_folder(
    pdf_dir: Path = PDF_DIR,
    out_dir: Path = OUT_DIR,
    overwrite: bool = False,
    reparse_incomplete: bool = False,
) -> list[dict]:
    """Batch-process all *.pdf files in pdf_dir → JSON files in out_dir."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDF files found in %s", pdf_dir)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Processing %d PDF files from %s → %s", len(pdf_files), pdf_dir, out_dir)

    summary: list[dict] = []
    errors: list[dict] = []
    converter = _build_converter()

    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        candidate_stem = pdf_path.stem
        m = _PMC_RE.match(pdf_path.stem.upper())
        if m:
            candidate_stem = m.group(0).upper()

        candidate_out = out_dir / (candidate_stem + ".json")

        skip = False
        if candidate_out.exists() and not overwrite:
            if reparse_incomplete:
                try:
                    existing = json.loads(candidate_out.read_text(encoding="utf-8"))
                    if existing.get("metadata", {}).get("full_text_available", True):
                        skip = True
                except Exception:
                    pass
            else:
                skip = True

        if skip:
            log.debug("Skipping (already parsed): %s", pdf_path.name)
            continue

        try:
            result_obj = converter.convert(str(pdf_path))
            doc = result_obj.document

            stem = _output_stem(pdf_path, doc)
            out_path = out_dir / (stem + ".json")

            if out_path.exists() and not overwrite:
                if reparse_incomplete:
                    try:
                        existing = json.loads(out_path.read_text(encoding="utf-8"))
                        if existing.get("metadata", {}).get("full_text_available", True):
                            log.debug("Skipping (already parsed by PMC ID): %s", pdf_path.name)
                            continue
                    except Exception:
                        pass
                else:
                    log.debug("Skipping (already parsed by PMC ID): %s", pdf_path.name)
                    continue

            meta = _extract_metadata(doc)
            sections, full_text_available = _extract_sections(doc)
            references = _extract_references(doc)

            meta, sections = _promote_first_header_section(meta, sections)
            meta = _enrich_metadata(meta, pdf_path, sections)

            if not meta["title"]:
                meta["title"] = pdf_path.stem

            meta["full_text_available"] = full_text_available
            meta["section_count"] = len(sections)
            meta["reference_count"] = len(references)

            output = {"metadata": meta, "sections": sections, "references": references}
            out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

            log.info(
                "Parsed %s → %s  (%d sections, %d refs, full_text=%s)",
                pdf_path.name, out_path.name, len(sections), len(references), full_text_available,
            )
            summary.append({
                "file": pdf_path.name, "output": out_path.name,
                "pmc_id": meta["pmc_id"], "doi": meta["doi"], "title": meta["title"],
                "sections": len(sections), "references": len(references),
                "full_text_available": full_text_available, "ok": True,
            })

        except Exception as e:
            log.error("Failed to parse %s: %s", pdf_path.name, e)
            errors.append({"file": pdf_path.name, "error": str(e), "ok": False})

    log.info("Done: %d parsed, %d errors", len(summary), len(errors))
    report_path = out_dir / "_pdf_parse_report.json"
    report_path.write_text(
        json.dumps({"parsed": summary, "errors": errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Report saved → %s", report_path)
    return summary + errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1].lower().endswith(".pdf"):
        result = parse_pdf(Path(sys.argv[1]))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        process_folder(
            overwrite="--overwrite" in sys.argv,
            reparse_incomplete="--reparse-incomplete" in sys.argv,
        )
