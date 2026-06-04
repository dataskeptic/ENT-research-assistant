"""Step 2a: Fetch full-text from PubMed Central via OAI-PMH API.

For each PubMed record that has a PMC ID, this downloads the full JATS XML
via the PMC OAI endpoint.  Optionally (--pdf flag) it also downloads the
PDF from the PMC FTP service.

Usage:
    python -m harvest.pmc_download          # XML only (default)
    python -m harvest.pmc_download --pdf    # XML + PDF download
    python -m harvest.pmc_download --pdf --overwrite  # re-download everything

Reads:  data/metadata/pubmed_results.json
Writes: data/fulltext/<PMCID>.xml
        data/pdfs/<PMCID>.pdf       (only with --pdf flag)
Updates fulltext_path, pmc_downloaded, pdf_path fields in the metadata.
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from lxml import etree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

METADATA_PATH = Path("data/metadata/pubmed_results.json")
FULLTEXT_DIR = Path("data/fulltext")
PDF_DIR = Path("data/pdfs")

OAI_BASE = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
# PMC FTP / HTTPS PDF endpoint pattern
PMC_PDF_BASE = "https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/"


def load_metadata() -> list[dict]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_metadata(records: list[dict]) -> None:
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def fetch_pmc_xml(pmc_id: str) -> str | None:
    """Fetch OAI-PMH XML for a PMC article. Returns raw XML string or None."""
    identifier = f"oai:pubmedcentral.nih.gov:{pmc_id.replace('PMC', '')}"
    params = {
        "verb": "GetRecord",
        "identifier": identifier,
        "metadataPrefix": "pmc",
    }
    try:
        time.sleep(0.35)  # ~3 req/s — polite for NCBI
        resp = requests.get(OAI_BASE, params=params, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        log.warning("OAI fetch failed for %s: %s", pmc_id, e)
        return None


def _get_pdf_filename_from_xml(xml_text: str) -> str | None:
    """Extract the PDF filename from the <self-uri content-type='pmc-pdf'> element.

    PMC XML embeds the canonical PDF filename, e.g.:
        <self-uri content-type="pmc-pdf" xlink:href="pathogens-15-00523.pdf"/>
    We use this to build the correct PDF download URL.
    """
    try:
        root = etree.fromstring(xml_text.encode())
        ns = {"xlink": "http://www.w3.org/1999/xlink"}
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
    """Download the PDF for a PMC article.

    Strategy:
      1. If we have the pdf_filename from the XML, use the direct PMC article URL.
      2. Otherwise fall back to the generic /articles/{pmc_id}/pdf/ redirect.
    """
    urls_to_try: list[str] = []
    bare_id = pmc_id.replace("PMC", "")

    if pdf_filename:
        # Direct URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC13209169/pdf/pathogens-15-00523.pdf
        urls_to_try.append(
            f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/{pdf_filename}"
        )
    # Fallback redirect URL
    urls_to_try.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/")

    headers = {
        "User-Agent": "ENTResearchHarvester/1.0 (research use; contact via GitHub)"
    }

    for url in urls_to_try:
        try:
            time.sleep(0.5)
            resp = requests.get(url, timeout=60, allow_redirects=True, headers=headers)
            if resp.status_code == 200 and len(resp.content) > 5000:
                content_type = resp.headers.get("content-type", "")
                if "pdf" in content_type or resp.content[:4] == b"%PDF":
                    return resp.content
                log.debug("Not a PDF response from %s (content-type: %s)", url, content_type)
        except requests.RequestException as e:
            log.warning("PDF fetch failed from %s: %s", url, e)

    return None


def run(
    metadata_path: Path = METADATA_PATH,
    fulltext_dir: Path = FULLTEXT_DIR,
    pdf_dir: Path = PDF_DIR,
    download_pdf: bool = False,
    overwrite: bool = False,
) -> None:
    """Main entry point: fetch XML (and optionally PDF) for all PMC-linked records."""
    records = load_metadata()

    fulltext_dir.mkdir(parents=True, exist_ok=True)
    if download_pdf:
        pdf_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        r for r in records
        if r.get("pmc_id") and (overwrite or not r.get("pmc_downloaded"))
    ]
    log.info("%d records to process (download_pdf=%s)", len(candidates), download_pdf)

    xml_ok = 0
    xml_fail = 0
    pdf_ok = 0
    pdf_fail = 0

    for record in tqdm(candidates, desc="PMC download"):
        pmc_id = record["pmc_id"]
        xml_path = fulltext_dir / f"{pmc_id}.xml"

        # ── XML ──────────────────────────────────────────────────────────────
        if overwrite or not xml_path.exists():
            xml_text = fetch_pmc_xml(pmc_id)
            if xml_text:
                xml_path.write_text(xml_text, encoding="utf-8")
                record["fulltext_path"] = str(xml_path)
                record["pmc_downloaded"] = True
                xml_ok += 1
            else:
                xml_fail += 1
                continue
        else:
            xml_text = xml_path.read_text(encoding="utf-8")
            record["fulltext_path"] = str(xml_path)
            record["pmc_downloaded"] = True

        # ── PDF (optional) ───────────────────────────────────────────────────
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
                    log.debug("PDF saved: %s", pdf_path.name)
                else:
                    pdf_fail += 1
                    log.warning("PDF not available for %s", pmc_id)
            else:
                record["pdf_path"] = str(pdf_path)
                record["fulltext_downloaded"] = True

    save_metadata(records)
    log.info("XML: %d ok, %d failed", xml_ok, xml_fail)
    if download_pdf:
        log.info("PDF: %d ok, %d failed/unavailable", pdf_ok, pdf_fail)


if __name__ == "__main__":
    import sys
    _download_pdf = "--pdf" in sys.argv
    _overwrite = "--overwrite" in sys.argv
    run(download_pdf=_download_pdf, overwrite=_overwrite)
