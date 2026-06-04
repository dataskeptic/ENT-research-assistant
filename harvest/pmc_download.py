"""Step 2a: Fetch full-text from PubMed Central via OAI-PMH API.

For each record (from ANY source file) that has a ``pmc_id``, this
downloads the full JATS XML via the PMC OAI endpoint.  Optionally
(``--pdf`` flag) it also downloads the PDF from the PMC FTP service.

Source files
------------
Reads ALL ``*_results.json`` files in ``data/metadata/`` — currently
``pubmed_results.json`` and ``rss_results.json``.  After downloading,
each source file is updated independently.

OAI unavailability
------------------
Very recent articles (published in the last 2–4 weeks) may not yet be
indexed by the PMC OAI export pipeline even though they exist on the PMC
website.  When an OAI fetch returns a 400/404, the record is marked
``oai_unavailable=True`` so ``unpaywall_resolve.py`` can pick it up as a
fallback via Unpaywall instead of leaving it stuck in “PMC pending”
forever.

Usage
-----
::

    python -m harvest.pmc_download                    # XML only (default)
    python -m harvest.pmc_download --pdf              # XML + PDF
    python -m harvest.pmc_download --overwrite        # re-download everything
    python -m harvest.pmc_download --pdf --overwrite

Writes: ``data/fulltext/<PMCID>.xml``
        ``data/pdfs/<PMCID>.pdf``  (only with ``--pdf`` flag)
Updates ``fulltext_path``, ``pmc_downloaded``, ``oai_unavailable``,
``pdf_path`` fields in each source metadata file.
"""

import json
import logging
import time
from pathlib import Path

import requests
from lxml import etree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

METADATA_DIR = Path("data/metadata")
FULLTEXT_DIR = Path("data/fulltext")
PDF_DIR = Path("data/pdfs")

_SOURCE_GLOB = "*_results.json"

OAI_BASE = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"


# ---------------------------------------------------------------------------
# I/O helpers (same pattern as unpaywall_resolve.py)
# ---------------------------------------------------------------------------

def load_all_records() -> tuple[list[dict], dict[str, list[dict]]]:
    """Load and merge records from all ``*_results.json`` source files.

    Returns
    -------
    merged : list[dict]
        Deduplicated flat list.  Each record carries ``_source_file`` so
        it can be written back to the correct file.
    by_file : dict[str, list[dict]]
        Original per-file lists keyed by file path string.
    """
    source_files = sorted(METADATA_DIR.glob(_SOURCE_GLOB))
    if not source_files:
        log.warning("No *_results.json files found in %s", METADATA_DIR)
        return [], {}

    by_file: dict[str, list[dict]] = {}
    all_records: list[dict] = []

    for path in source_files:
        try:
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            log.info("Loaded %d records from %s", len(records), path.name)
        except Exception as e:
            log.error("Could not load %s: %s", path.name, e)
            continue
        for rec in records:
            rec["_source_file"] = str(path)
        by_file[str(path)] = records
        all_records.extend(records)

    # Deduplicate by pmc_id — keep first occurrence
    merged: list[dict] = []
    seen_pmc: set[str] = set()
    for rec in all_records:
        pmc_id = (rec.get("pmc_id") or "").strip()
        if pmc_id:
            if pmc_id in seen_pmc:
                continue
            seen_pmc.add(pmc_id)
        merged.append(rec)

    dupes = len(all_records) - len(merged)
    log.info(
        "Merged pool: %d records from %d files (%d duplicates removed)",
        len(merged), len(source_files), dupes,
    )
    return merged, by_file


def save_all_records(by_file: dict[str, list[dict]]) -> None:
    """Write each source file back with updated metadata, stripping _source_file."""
    for path_str, records in by_file.items():
        clean = [{k: v for k, v in r.items() if k != "_source_file"} for r in records]
        with open(path_str, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        log.debug("Saved %d records \u2192 %s", len(clean), Path(path_str).name)


# ---------------------------------------------------------------------------
# OAI-PMH fetch
# ---------------------------------------------------------------------------

def fetch_pmc_xml(pmc_id: str) -> tuple[str | None, bool]:
    """Fetch OAI-PMH XML for a PMC article.

    Returns
    -------
    xml_text : str | None
        Raw XML string on success, None on failure.
    oai_unavailable : bool
        True when the server returned 400/404 — meaning the article exists
        in PMC but is not yet indexed by OAI (typical for articles <4 weeks
        old).  Callers should mark the record ``oai_unavailable=True`` so
        Unpaywall can serve as a fallback.
    """
    identifier = f"oai:pubmedcentral.nih.gov:{pmc_id.replace('PMC', '')}"
    params = {
        "verb": "GetRecord",
        "identifier": identifier,
        "metadataPrefix": "pmc",
    }
    try:
        time.sleep(0.35)  # ~3 req/s — polite for NCBI
        resp = requests.get(OAI_BASE, params=params, timeout=30)
        if resp.status_code in (400, 404):
            log.warning(
                "OAI fetch failed for %s: %s %s (article not yet in OAI index — "
                "marking oai_unavailable, Unpaywall will be tried as fallback)",
                pmc_id, resp.status_code, resp.reason,
            )
            return None, True  # oai_unavailable
        resp.raise_for_status()
        return resp.text, False
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        unavailable = status in (400, 404)
        log.warning("OAI fetch failed for %s: %s", pmc_id, e)
        return None, unavailable
    except requests.RequestException as e:
        log.warning("OAI fetch failed for %s: %s", pmc_id, e)
        return None, False  # network error, not an OAI unavailability


# ---------------------------------------------------------------------------
# PDF fetch
# ---------------------------------------------------------------------------

def _get_pdf_filename_from_xml(xml_text: str) -> str | None:
    """Extract the PDF filename from the ``<self-uri content-type='pmc-pdf'>`` element."""
    try:
        root = etree.fromstring(xml_text.encode())
        for el in root.iter():
            if el.tag.split("}")[-1] == "self-uri":
                if el.get("content-type") == "pmc-pdf":
                    href = el.get("{http://www.w3.org/1999/xlink}href") or el.get("href", "")
                    if href.endswith(".pdf"):
                        return href
    except Exception:
        pass
    return None


def fetch_pmc_pdf(pmc_id: str, pdf_filename: str | None = None) -> bytes | None:
    """Download the PDF for a PMC article."""
    urls_to_try: list[str] = []

    if pdf_filename:
        urls_to_try.append(
            f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/{pdf_filename}"
        )
    urls_to_try.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/")

    headers = {"User-Agent": "ENTResearchHarvester/1.0 (research use; contact via GitHub)"}

    for url in urls_to_try:
        try:
            time.sleep(0.5)
            resp = requests.get(url, timeout=60, allow_redirects=True, headers=headers)
            if resp.status_code == 200 and len(resp.content) > 5000:
                content_type = resp.headers.get("content-type", "")
                if "pdf" in content_type or resp.content[:4] == b"%PDF":
                    return resp.content
        except requests.RequestException as e:
            log.warning("PDF fetch failed from %s: %s", url, e)

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    fulltext_dir: Path = FULLTEXT_DIR,
    pdf_dir: Path = PDF_DIR,
    download_pdf: bool = False,
    overwrite: bool = False,
) -> None:
    """Fetch XML (and optionally PDF) for all PMC-linked records across all sources."""
    records, by_file = load_all_records()
    if not records:
        log.error("No records loaded. Run pubmed_harvest.py or journal_rss.py first.")
        return

    fulltext_dir.mkdir(parents=True, exist_ok=True)
    if download_pdf:
        pdf_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        r for r in records
        if r.get("pmc_id") and (
            overwrite
            or not r.get("pmc_downloaded")
            or not (fulltext_dir / f"{r['pmc_id']}.xml").exists()
        )
    ]
    log.info("%d records to process (download_pdf=%s)", len(candidates), download_pdf)

    # Break down by source so the user knows where records are coming from
    by_source: dict[str, int] = {}
    for r in candidates:
        src = r.get("source", "pubmed")
        by_source[src] = by_source.get(src, 0) + 1
    for src, cnt in sorted(by_source.items()):
        log.info("  %s: %d records with pmc_id", src, cnt)

    xml_ok = 0
    xml_fail = 0
    xml_unavailable = 0
    pdf_ok = 0
    pdf_fail = 0

    for record in tqdm(candidates, desc="PMC download"):
        pmc_id = record["pmc_id"]
        xml_path = fulltext_dir / f"{pmc_id}.xml"

        # ── XML ────────────────────────────────────────────────────────────
        if overwrite or not xml_path.exists():
            xml_text, oai_unavailable = fetch_pmc_xml(pmc_id)
            if xml_text:
                xml_path.write_text(xml_text, encoding="utf-8")
                record["fulltext_path"] = str(xml_path)
                record["xml_path"] = str(xml_path)
                record["pmc_downloaded"] = True
                record.pop("oai_unavailable", None)  # clear any previous failure flag
                xml_ok += 1
            else:
                if oai_unavailable:
                    # Article not yet in OAI index — flag it so Unpaywall picks it up
                    record["oai_unavailable"] = True
                    xml_unavailable += 1
                else:
                    xml_fail += 1
                continue  # no XML → skip PDF attempt
        else:
            xml_text = xml_path.read_text(encoding="utf-8")
            record["fulltext_path"] = str(xml_path)
            record["xml_path"] = str(xml_path)
            record["pmc_downloaded"] = True

        # ── PDF (optional) ─────────────────────────────────────────────────────
        if download_pdf:
            pdf_path = pdf_dir / f"{pmc_id}.pdf"
            if overwrite or not pdf_path.exists():
                pdf_filename = _get_pdf_filename_from_xml(xml_text)
                pdf_bytes = fetch_pmc_pdf(pmc_id, pdf_filename)
                if pdf_bytes:
                    pdf_path.write_bytes(pdf_bytes)
                    record["pdf_path"] = str(pdf_path)
                    record["fulltext_downloaded"] = True
                    pdf_ok += 1
                else:
                    pdf_fail += 1
                    log.warning("PDF not available for %s", pmc_id)
            else:
                record["pdf_path"] = str(pdf_path)
                record["fulltext_downloaded"] = True

    save_all_records(by_file)
    log.info("XML: %d ok, %d OAI-unavailable (will fall back to Unpaywall), %d other errors",
             xml_ok, xml_unavailable, xml_fail)
    if download_pdf:
        log.info("PDF: %d ok, %d failed/unavailable", pdf_ok, pdf_fail)
    if xml_unavailable:
        log.info(
            "Tip: run 'python -m harvest.unpaywall_resolve --download' to fetch PDFs "
            "for the %d OAI-unavailable articles.",
            xml_unavailable,
        )


if __name__ == "__main__":
    import sys
    _download_pdf = "--pdf" in sys.argv
    _overwrite = "--overwrite" in sys.argv
    run(download_pdf=_download_pdf, overwrite=_overwrite)
