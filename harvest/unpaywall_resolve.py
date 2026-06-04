"""Step 2b: For papers not in PMC OA, resolve OA PDF URLs via Unpaywall API.

Unpaywall is free and legal — it only returns open-access PDFs.
Requires an email address (no registration, just identification).

Reads:  data/metadata/pubmed_results.json  (papers where fulltext_downloaded=False)
Writes: data/pdfs/<doi_safe>.pdf
Updates pdf_path, fulltext_downloaded fields in metadata.
"""

import os
import re
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

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
EMAIL = os.getenv("UNPAYWALL_EMAIL", "")
METADATA_PATH = "data/metadata/pubmed_results.json"
PDF_DIR = Path("data/pdfs")


def doi_to_filename(doi: str) -> str:
    """Convert DOI to a safe filename."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", doi) + ".pdf"


def load_metadata() -> list[dict]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_metadata(records: list[dict]) -> None:
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def resolve_unpaywall(doi: str) -> str | None:
    """Query Unpaywall for a best OA PDF URL. Returns URL or None."""
    if not EMAIL:
        raise ValueError("UNPAYWALL_EMAIL not set in .env")
    url = f"{UNPAYWALL_BASE}/{doi}"
    try:
        time.sleep(0.2)  # be polite
        resp = requests.get(url, params={"email": EMAIL}, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        # best_oa_location is Unpaywall's recommended OA version
        best = data.get("best_oa_location")
        if best and best.get("url_for_pdf"):
            return best["url_for_pdf"]
        # fallback: scan all OA locations
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
    except (requests.RequestException, ValueError) as e:
        log.warning("Unpaywall failed for DOI %s: %s", doi, e)
    return None


def download_pdf(url: str, out_path: Path) -> bool:
    """Download a PDF from a URL. Returns True on success."""
    try:
        time.sleep(0.5)
        resp = requests.get(
            url, timeout=60, allow_redirects=True,
            headers={"User-Agent": "ENTResearchHarvester/1.0 (research use; contact via GitHub)"}
        )
        if resp.status_code == 200 and len(resp.content) > 1000:
            out_path.write_bytes(resp.content)
            return True
    except requests.RequestException as e:
        log.warning("PDF download failed from %s: %s", url, e)
    return False


def run() -> None:
    if not EMAIL:
        log.error("Set UNPAYWALL_EMAIL in your .env file before running.")
        return

    records = load_metadata()
    candidates = [
        r for r in records
        if not r["fulltext_downloaded"] and r.get("doi")
    ]
    log.info("%d papers without full text to resolve via Unpaywall", len(candidates))

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    resolved = 0

    for record in tqdm(candidates, desc="Unpaywall resolve"):
        doi = record["doi"]
        pdf_url = resolve_unpaywall(doi)

        if not pdf_url:
            log.debug("No OA PDF found for DOI: %s", doi)
            continue

        out_path = PDF_DIR / doi_to_filename(doi)
        if download_pdf(pdf_url, out_path):
            record["fulltext_downloaded"] = True
            record["pdf_path"] = str(out_path)
            record["oa_pdf_url"] = pdf_url
            resolved += 1
            log.debug("Downloaded: %s → %s", doi, out_path.name)

    save_metadata(records)
    log.info("Unpaywall: %d PDFs downloaded (%d had no OA version)",
             resolved, len(candidates) - resolved)


if __name__ == "__main__":
    run()
