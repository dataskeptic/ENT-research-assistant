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
      "sections": [
        {"section": "Abstract",      "text": "...", "order": 0},
        {"section": "Introduction",  "text": "...", "order": 1},
        {"section": "Methods",       "text": "...", "order": 2},
        {"section": "Results",       "text": "...", "order": 3},
        {"section": "Discussion",    "text": "...", "order": 4},
        {"section": "Conclusion",    "text": "...", "order": 5},
      ],
      "references": [
        {
          "ref_id": "R1",
          "label": "1.",
          "authors": ["Smith J", "Jones A"],
          "title": "A great paper",
          "journal": "Lancet",
          "year": "2023",
          "volume": "42",
          "issue": "3",
          "fpage": "100",
          "lpage": "108",
          "doi": "10.1016/...",
          "pmid": "12345678",
          "pmc_id": "PMC9876543"
        },
        ...
      ]
    }

Docling dependency:
    pip install docling>=2.0.0
"""

import json
import logging
import re
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/parsed")

# ---------------------------------------------------------------------------
# Section title normalisation — same aliases as parse_pmc_xml.py
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
    # Explicitly skipped — not useful for RAG
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

# Regex to extract DOI from raw text (e.g. from a reference line)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s,;\"\'>\]]+", re.IGNORECASE)

# Regex to find PMC IDs embedded in text
_PMC_RE = re.compile(r"\bPMC\d+\b", re.IGNORECASE)

# Regex to find PubMed IDs (bare 8-digit numbers preceded by "PMID")
_PMID_RE = re.compile(r"\bPMID[:\s]+(\d{7,9})\b", re.IGNORECASE)


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalise_section_title(raw: str) -> str | None:
    key = raw.strip().lower()
    # Strip leading numbering like "1. Introduction" → "introduction"
    key = re.sub(r"^[\d\.]+\s*", "", key).strip()
    if key in _SECTION_ALIASES:
        return _SECTION_ALIASES[key]
    for alias, canonical in _SECTION_ALIASES.items():
        if alias in key:
            return canonical
    # Unknown section: keep as-is, title-cased
    return raw.strip().title()


# ---------------------------------------------------------------------------
# Docling-based PDF → structured document
# ---------------------------------------------------------------------------

def _load_docling():
    """Import Docling lazily so the module can be imported even if Docling
    is not installed (e.g. during unit tests that mock this function)."""
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        return DocumentConverter, InputFormat, PdfPipelineOptions
    except ImportError as e:
        raise ImportError(
            "Docling is required for PDF parsing. Install it with:\n"
            "    pip install docling>=2.0.0"
        ) from e


def _build_converter():
    """Build a Docling DocumentConverter tuned for scientific PDFs."""
    DocumentConverter, InputFormat, PdfPipelineOptions = _load_docling()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False            # OCR only if text layer absent
    pipeline_options.do_table_structure = False  # skip table parsing (not needed for RAG)

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: pipeline_options,  # type: ignore[index]
        },
    )
    return converter


# ---------------------------------------------------------------------------
# Metadata extraction from a Docling DoclingDocument
# ---------------------------------------------------------------------------

def _extract_metadata(doc) -> dict:
    """Extract article-level metadata from a Docling document.

    Docling exposes metadata via doc.metadata (if available from embedded
    PDF XMP/Info fields) and via the document body text for fields like DOI.
    We attempt best-effort extraction; missing fields are left as empty strings.
    """
    meta: dict = {
        "title": "",
        "authors": [],
        "journal": "",
        "year": "",
        "doi": "",
        "pmc_id": "",
        "pmid": "",
    }

    # ── Title ────────────────────────────────────────────────────────────────
    # Docling exposes the document title via doc.name or the first heading
    if hasattr(doc, "name") and doc.name:
        meta["title"] = _clean(doc.name)

    # Try PDF metadata fields if available
    if hasattr(doc, "metadata") and doc.metadata:
        md = doc.metadata
        if hasattr(md, "title") and md.title:
            meta["title"] = _clean(str(md.title))
        if hasattr(md, "authors") and md.authors:
            raw_authors = md.authors
            if isinstance(raw_authors, list):
                meta["authors"] = [_clean(str(a)) for a in raw_authors if str(a).strip()]
            elif isinstance(raw_authors, str):
                meta["authors"] = [a.strip() for a in raw_authors.split(";") if a.strip()]

    # ── DOI / PMID / PMC from full document text ──────────────────────────
    # Iterate over the first ~2000 chars of document text to find identifiers.
    # These are almost always in the header/footer of a journal PDF.
    full_head = ""
    try:
        items = list(doc.iterate_items())
        # Grab text from first 20 items max — the header region
        for item, _ in items[:20]:
            if hasattr(item, "text") and item.text:
                full_head += " " + item.text
            if len(full_head) > 3000:
                break
    except Exception:
        pass

    if not meta["doi"]:
        doi_match = _DOI_RE.search(full_head)
        if doi_match:
            meta["doi"] = doi_match.group(0).rstrip(".")

    if not meta["pmc_id"]:
        pmc_match = _PMC_RE.search(full_head)
        if pmc_match:
            meta["pmc_id"] = pmc_match.group(0).upper()

    if not meta["pmid"]:
        pmid_match = _PMID_RE.search(full_head)
        if pmid_match:
            meta["pmid"] = pmid_match.group(1)

    return meta


# ---------------------------------------------------------------------------
# Section extraction from a Docling DoclingDocument
# ---------------------------------------------------------------------------

def _extract_sections(doc) -> tuple[list[dict], bool]:
    """Walk Docling document items and group text under section headings.

    Docling labels items with DocItemLabel values:
      SECTION_HEADER  → heading / section title
      TEXT            → regular paragraph
      LIST_ITEM       → bullet / numbered list item
      TITLE           → document title (skip — already in metadata)
      TABLE / FIGURE  → skipped (same policy as parse_pmc_xml.py)

    Returns
    -------
    sections : list[dict]
        Each dict has keys: section (str), text (str), order (int).
    full_text_available : bool
        True if more than just an abstract was found.
    """
    try:
        from docling.datamodel.document import DocItemLabel
    except ImportError:
        # Older Docling versions use a different import path
        try:
            from docling_core.types.doc import DocItemLabel
        except ImportError:
            DocItemLabel = None

    sections: list[dict] = []
    current_heading = "Body"        # default if no section header found first
    current_paragraphs: list[str] = []
    order_counter = [0]

    _SKIP_LABELS = {"TABLE", "FIGURE", "PICTURE", "FORMULA", "PAGE_HEADER",
                    "PAGE_FOOTER", "FOOTNOTE", "CAPTION"}

    def _flush(heading: str, paragraphs: list[str]) -> None:
        """Save accumulated paragraphs under a heading if non-trivial."""
        if not paragraphs:
            return
        text = _clean(" ".join(paragraphs))
        if len(text) < 50:
            return
        canonical = _normalise_section_title(heading)
        if canonical is None:
            return  # explicitly skipped
        sections.append({
            "section": canonical,
            "text": text,
            "order": order_counter[0],
        })
        order_counter[0] += 1

    try:
        items = list(doc.iterate_items())
    except Exception as e:
        log.warning("Could not iterate document items: %s", e)
        return [], False

    for item, _level in items:
        # Get the label string robustly across Docling versions
        label_str = ""
        if hasattr(item, "label"):
            label_str = str(item.label).upper()
            # Docling uses enum values like DocItemLabel.TEXT → "TEXT"
            # or the full path "DocItemLabel.TEXT" — normalise both
            label_str = label_str.split(".")[-1]

        if not label_str:
            continue

        # Skip non-prose items
        if any(skip in label_str for skip in _SKIP_LABELS):
            continue

        item_text = ""
        if hasattr(item, "text") and item.text:
            item_text = _clean(item.text)

        if not item_text:
            continue

        if "SECTION_HEADER" in label_str or "HEADING" in label_str:
            # Flush current section before starting a new one
            _flush(current_heading, current_paragraphs)
            current_heading = item_text
            current_paragraphs = []

        elif "TITLE" in label_str:
            # Document title — skip for sections (already captured in metadata)
            continue

        elif "TEXT" in label_str or "LIST_ITEM" in label_str or "PARAGRAPH" in label_str:
            current_paragraphs.append(item_text)

    # Flush the final section
    _flush(current_heading, current_paragraphs)

    full_text_available = any(
        s["section"] not in ("Abstract",) for s in sections
    ) and len(sections) > 1

    return sections, full_text_available


# ---------------------------------------------------------------------------
# Reference extraction from a Docling DoclingDocument
# ---------------------------------------------------------------------------

def _extract_references(doc) -> list[dict]:
    """Extract references from the document.

    Docling exposes parsed references via doc.references when available.
    For PDFs where Docling does not fully structure the bibliography,
    we fall back to raw text extraction from the reference section,
    producing minimal entries with a raw_citation field — same fallback
    strategy as parse_pmc_xml.py's mixed-citation handling.

    Returns a list of dicts matching the parse_pmc_xml.py reference schema.
    """
    refs: list[dict] = []

    # ── Strategy 1: Docling native reference objects ──────────────────────
    if hasattr(doc, "references") and doc.references:
        for i, ref in enumerate(doc.references, start=1):
            entry: dict = {"ref_id": f"R{i}", "label": f"{i}."}
            if hasattr(ref, "title") and ref.title:
                entry["title"] = _clean(str(ref.title))
            if hasattr(ref, "authors") and ref.authors:
                raw = ref.authors
                if isinstance(raw, list):
                    entry["authors"] = [_clean(str(a)) for a in raw if str(a).strip()]
                elif isinstance(raw, str):
                    entry["authors"] = [a.strip() for a in raw.split(";") if a.strip()]
            if hasattr(ref, "journal") and ref.journal:
                entry["journal"] = _clean(str(ref.journal))
            if hasattr(ref, "year") and ref.year:
                entry["year"] = str(ref.year).strip()
            if hasattr(ref, "volume") and ref.volume:
                entry["volume"] = str(ref.volume).strip()
            if hasattr(ref, "issue") and ref.issue:
                entry["issue"] = str(ref.issue).strip()
            if hasattr(ref, "pages") and ref.pages:
                pages = str(ref.pages).strip()
                if "-" in pages:
                    parts = pages.split("-", 1)
                    entry["fpage"] = parts[0].strip()
                    entry["lpage"] = parts[1].strip()
                else:
                    entry["fpage"] = pages
            if hasattr(ref, "doi") and ref.doi:
                entry["doi"] = str(ref.doi).strip()
            if hasattr(ref, "pmid") and ref.pmid:
                entry["pmid"] = str(ref.pmid).strip()
            if hasattr(ref, "pmc_id") and ref.pmc_id:
                val = str(ref.pmc_id).strip()
                entry["pmc_id"] = val if val.startswith("PMC") else f"PMC{val}"
            if len(entry) > 2:  # more than just ref_id + label
                refs.append(entry)
        if refs:
            return refs

    # ── Strategy 2: Scrape reference text items from the back of the doc ──
    # If Docling didn't parse structured references, find the reference
    # section by label and collect raw text lines.
    in_references = False
    ref_lines: list[str] = []

    try:
        items = list(doc.iterate_items())
    except Exception:
        return []

    for item, _level in items:
        label_str = ""
        if hasattr(item, "label"):
            label_str = str(item.label).upper().split(".")[-1]

        item_text = ""
        if hasattr(item, "text") and item.text:
            item_text = _clean(item.text)

        if not item_text:
            continue

        if "SECTION_HEADER" in label_str or "HEADING" in label_str:
            key = item_text.strip().lower()
            key = re.sub(r"^[\d\.]+\s*", "", key).strip()
            in_references = key in ("references", "bibliography", "works cited")
            continue

        if in_references and ("TEXT" in label_str or "LIST_ITEM" in label_str):
            ref_lines.append(item_text)

    # Build minimal reference entries from raw lines
    for i, line in enumerate(ref_lines, start=1):
        if len(line) < 20:
            continue
        entry: dict = {
            "ref_id": f"R{i}",
            "label": f"{i}.",
            "raw_citation": line,
        }
        # Try to extract DOI from the raw line
        doi_match = _DOI_RE.search(line)
        if doi_match:
            entry["doi"] = doi_match.group(0).rstrip(".")
        # Try to extract year (4-digit number between 1900–2099)
        year_match = re.search(r"\b(19|20)\d{2}\b", line)
        if year_match:
            entry["year"] = year_match.group(0)
        if entry:
            refs.append(entry)

    return refs


# ---------------------------------------------------------------------------
# Stem → output path resolution
# ---------------------------------------------------------------------------

def _output_stem(pdf_path: Path, doc=None) -> str:
    """Determine the output filename stem for a PDF.

    Priority:
    1. PMC ID found in the PDF header text (e.g. "PMC12353658")
    2. Original PDF filename stem (e.g. "my_paper" → "my_paper")

    This keeps output filenames consistent with the PMC-based naming used
    by parse_pmc_xml.py wherever a PMC ID is recoverable.
    """
    # Try to extract PMC ID from early document text
    if doc is not None:
        try:
            head = ""
            for item, _ in list(doc.iterate_items())[:20]:
                if hasattr(item, "text") and item.text:
                    head += " " + item.text
                if len(head) > 2000:
                    break
            pmc_match = _PMC_RE.search(head)
            if pmc_match:
                return pmc_match.group(0).upper()
        except Exception:
            pass

    return pdf_path.stem


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str | Path) -> dict:
    """Parse a single PDF and return the structured JSON dict.

    Returns a dict with keys: metadata, sections, references —
    identical schema to parse_pmc_xml.py output.
    """
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

    # If title still empty, use filename stem as fallback
    if not meta["title"]:
        meta["title"] = pdf_path.stem.replace("_", " ").replace("-", " ").title()

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
    """Batch-process all *.pdf files in pdf_dir → JSON files in out_dir.

    Parameters
    ----------
    pdf_dir : Path
        Directory containing ``*.pdf`` files.
    out_dir : Path
        Directory where ``*.json`` parsed outputs are written.
        Shared with parse_pmc_xml.py — both write to data/parsed/.
    overwrite : bool
        If True, re-parse all files even if a JSON already exists.
    reparse_incomplete : bool
        If True, re-parse PDFs previously saved as abstract-only
        (``full_text_available=false``) in case extraction improves.
    """
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDF files found in %s", pdf_dir)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Processing %d PDF files from %s → %s", len(pdf_files), pdf_dir, out_dir)

    summary = []
    errors = []

    # Build converter once — Docling startup is expensive
    converter = _build_converter()

    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        # Peek at PMC ID from filename first (fast path)
        candidate_stem = pdf_path.stem
        pmc_in_name = _PMC_RE.match(pdf_path.stem.upper())
        if pmc_in_name:
            candidate_stem = pmc_in_name.group(0).upper()

        candidate_out = out_dir / (candidate_stem + ".json")

        # Quick skip based on filename stem — avoids loading Docling for every file
        skip = False
        if candidate_out.exists() and not overwrite:
            if reparse_incomplete:
                try:
                    with open(candidate_out, encoding="utf-8") as f:
                        existing = json.load(f)
                    if existing.get("metadata", {}).get("full_text_available", True):
                        skip = True
                except Exception:
                    pass  # corrupt JSON → re-parse
            else:
                skip = True

        if skip:
            log.debug("Skipping (already parsed): %s", pdf_path.name)
            continue

        try:
            result_obj = converter.convert(str(pdf_path))
            doc = result_obj.document

            # Resolve final output stem (may find PMC ID inside the PDF)
            stem = _output_stem(pdf_path, doc)
            out_path = out_dir / (stem + ".json")

            # Re-check skip after resolving stem from doc content
            if out_path.exists() and not overwrite:
                if reparse_incomplete:
                    try:
                        with open(out_path, encoding="utf-8") as f:
                            existing = json.load(f)
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

            if not meta["title"]:
                meta["title"] = pdf_path.stem.replace("_", " ").replace("-", " ").title()

            meta["full_text_available"] = full_text_available
            meta["section_count"] = len(sections)
            meta["reference_count"] = len(references)

            result = {"metadata": meta, "sections": sections, "references": references}

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            log.info(
                "Parsed %s → %s  (%d sections, %d refs, full_text=%s)",
                pdf_path.name, out_path.name,
                len(sections), len(references), full_text_available,
            )

            summary.append({
                "file": pdf_path.name,
                "output": out_path.name,
                "pmc_id": meta["pmc_id"],
                "doi": meta["doi"],
                "title": meta["title"],
                "sections": len(sections),
                "references": len(references),
                "full_text_available": full_text_available,
                "ok": True,
            })

        except Exception as e:
            log.error("Failed to parse %s: %s", pdf_path.name, e)
            errors.append({"file": pdf_path.name, "error": str(e), "ok": False})

    ok_count = len(summary)
    log.info("Done: %d parsed, %d errors", ok_count, len(errors))

    report_path = out_dir / "_pdf_parse_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"parsed": summary, "errors": errors}, f, ensure_ascii=False, indent=2)
    log.info("Report saved → %s", report_path)

    return summary + errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1].lower().endswith(".pdf"):
        # Single-file debug mode
        pdf_file = Path(sys.argv[1])
        result = parse_pdf(pdf_file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        overwrite = "--overwrite" in sys.argv
        reparse_incomplete = "--reparse-incomplete" in sys.argv
        process_folder(overwrite=overwrite, reparse_incomplete=reparse_incomplete)
