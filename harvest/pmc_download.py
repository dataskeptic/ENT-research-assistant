"""Step 2a: Download full text (XML + PDF) for papers available in PMC Open Access.

Reads:  data/metadata/pubmed_results.json
Writes: data/fulltext/<pmcid>.xml
        data/pdfs/<pmcid>.pdf
Updates fulltext_downloaded, xml_path, pdf_path fields in metadata.
"""

import os
import json
import time
import logging
from pathlib import Path

import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
PMC_PDF_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles"
API_KEY = os.getenv("NCBI_API_KEY", "")

METADATA_PATH = "data/metadata/pubmed_results.json"
FULLTEXT_DIR = Path("data/fulltext")
PDF_DIR = Path("data/pdfs")


def load_metadata() -> list[dict]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_metadata(records: list[dict]) -> None:
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def is_pmc_oa(pmc_id: str) -> bool:
    """Check if a PMC article is in the Open Access subset."""
    if not pmc_id:
        return False
    # PMC OA API: request record — if it returns content, it's OA
    params = {
        "verb": "GetRecord",
        "identifier": f"oai:pubmedcentral.nih.gov:{pmc_id.replace('PMC', '')}",
        "metadataPrefix": "pmc",
    }
    try:
        time.sleep(0.35)
        resp = requests.get(PMC_OA_BASE, params=params, timeout=30)
        return "<error" not in resp.text and "idDoesNotExist" not in resp.text
    except requests.RequestException:
        return False


def download_pmc_xml(pmc_id: str) -> str | None:
    """Download full-text XML for a PMC OA article. Returns saved path or None."""
    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    clean_id = pmc_id.replace("PMC", "")
    out_path = FULLTEXT_DIR / f"{pmc_id}.xml"

    if out_path.exists():
        return str(out_path)

    params = {
        "verb": "GetRecord",
        "identifier": f"oai:pubmedcentral.nih.gov:{clean_id}",
        "metadataPrefix": "pmc",
    }
    try:
        time.sleep(0.35)
        resp = requests.get(PMC_OA_BASE, params=params, timeout=60)
        if resp.status_code == 200 and "<error" not in resp.text:
            out_path.write_bytes(resp.content)
            log.debug("XML saved: %s", out_path)
            return str(out_path)
    except requests.RequestException as e:
        log.warning("XML download failed for %s: %s", pmc_id, e)
    return None


def download_pmc_pdf(pmc_id: str) -> str | None:
    """Attempt to download PDF from PMC for an OA article. Returns saved path or None."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PDF_DIR / f"{pmc_id}.pdf"

    if out_path.exists():
        return str(out_path)

    # PMC PDF URL pattern
    url = f"{PMC_PDF_BASE}/{pmc_id}/pdf/"
    try:
        time.sleep(0.5)
        resp = requests.get(url, timeout=60, allow_redirects=True,
                            headers={"User-Agent": "ENTResearchHarvester/1.0 (research use; contact via GitHub)"})
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
            out_path.write_bytes(resp.content)
            log.debug("PDF saved: %s", out_path)
            return str(out_path)
    except requests.RequestException as e:
        log.warning("PDF download failed for %s: %s", pmc_id, e)
    return None


def run() -> None:
    records = load_metadata()
    pmc_candidates = [r for r in records if r.get("pmc_id") and not r["fulltext_downloaded"]]
    log.info("%d papers have a PMC ID to check", len(pmc_candidates))

    updated = 0
    for record in tqdm(pmc_candidates, desc="PMC OA download"):
        pmc_id = record["pmc_id"]

        if not is_pmc_oa(pmc_id):
            log.debug("%s not in PMC OA subset, skipping", pmc_id)
            continue

        xml_path = download_pmc_xml(pmc_id)
        pdf_path = download_pmc_pdf(pmc_id)

        if xml_path or pdf_path:
            record["fulltext_downloaded"] = True
            record["xml_path"] = xml_path
            record["pdf_path"] = pdf_path
            updated += 1

    save_metadata(records)
    log.info("PMC OA: %d full texts downloaded", updated)


if __name__ == "__main__":
    run()
