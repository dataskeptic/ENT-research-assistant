"""Step 2b: Resolve OA PDF URLs via Unpaywall and optionally download them.

Two modes:

  AUDIT mode (default, --audit or no flag):
      Checks all DOIs against Unpaywall and prints a summary of how many
      have a freely available PDF. Does NOT download anything.
      Use this first to estimate coverage before committing to downloads.

  DOWNLOAD mode (--download):
      Resolves + downloads PDFs for all papers without full text.

Usage:
    python -m harvest.unpaywall_resolve            # audit only
    python -m harvest.unpaywall_resolve --audit    # same
    python -m harvest.unpaywall_resolve --download # resolve + download

Requires UNPAYWALL_EMAIL in .env (no registration needed, just identification).

Reads:  data/metadata/pubmed_results.json
Writes: data/pdfs/<doi_safe>.pdf          (download mode only)
Updates pdf_path, fulltext_downloaded, oa_pdf_url in metadata (download only).
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
METADATA_PATH = Path("data/metadata/pubmed_results.json")
PDF_DIR = Path("data/pdfs")


def doi_to_filename(doi: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", doi) + ".pdf"


def load_metadata() -> list[dict]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_metadata(records: list[dict]) -> None:
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def resolve_unpaywall(doi: str) -> dict | None:
    """Query Unpaywall for a DOI. Returns a dict with OA info or None on failure.

    Dict shape:
        {
            "pdf_url": str | None,
            "oa_status": str,          # 'gold', 'hybrid', 'green', 'bronze', 'closed'
            "host_type": str | None,   # 'publisher', 'repository'
            "version": str | None,     # 'publishedVersion', 'acceptedVersion', etc.
        }
    """
    if not EMAIL:
        raise ValueError("UNPAYWALL_EMAIL not set in .env")
    try:
        time.sleep(0.2)
        resp = requests.get(
            f"{UNPAYWALL_BASE}/{doi}",
            params={"email": EMAIL},
            timeout=20,
        )
        if resp.status_code == 404:
            return {"pdf_url": None, "oa_status": "not_found", "host_type": None, "version": None}
        resp.raise_for_status()
        data = resp.json()

        oa_status = data.get("oa_status", "unknown")
        best = data.get("best_oa_location")
        pdf_url = None
        host_type = None
        version = None

        if best:
            pdf_url = best.get("url_for_pdf")
            host_type = best.get("host_type")
            version = best.get("version")

        # Fallback: scan all locations for any PDF
        if not pdf_url:
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    pdf_url = loc["url_for_pdf"]
                    host_type = loc.get("host_type")
                    version = loc.get("version")
                    break

        return {
            "pdf_url": pdf_url,
            "oa_status": oa_status,
            "host_type": host_type,
            "version": version,
        }
    except (requests.RequestException, ValueError) as e:
        log.warning("Unpaywall error for DOI %s: %s", doi, e)
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


def audit(records: list[dict]) -> None:
    """Audit mode: check Unpaywall coverage without downloading anything."""
    candidates = [r for r in records if not r.get("fulltext_downloaded") and r.get("doi")]
    already_done = sum(1 for r in records if r.get("fulltext_downloaded"))
    no_doi = sum(1 for r in records if not r.get("doi"))

    log.info("=== Unpaywall Audit Mode ===")
    log.info("Total records    : %d", len(records))
    log.info("Already have PDF : %d", already_done)
    log.info("No DOI (skip)    : %d", no_doi)
    log.info("To check         : %d", len(candidates))

    if not candidates:
        log.info("Nothing to check.")
        return

    stats = {
        "has_pdf": 0,
        "oa_no_pdf": 0,       # OA but no direct PDF URL
        "closed": 0,
        "not_found": 0,
        "error": 0,
    }
    oa_status_counts: dict[str, int] = {}
    available: list[dict] = []

    for record in tqdm(candidates, desc="Auditing Unpaywall"):
        doi = record["doi"]
        result = resolve_unpaywall(doi)

        if result is None:
            stats["error"] += 1
            continue

        oa_status = result["oa_status"]
        oa_status_counts[oa_status] = oa_status_counts.get(oa_status, 0) + 1

        if result["pdf_url"]:
            stats["has_pdf"] += 1
            available.append({
                "doi": doi,
                "title": record.get("title", ""),
                "pdf_url": result["pdf_url"],
                "oa_status": oa_status,
                "host_type": result["host_type"],
                "version": result["version"],
            })
        elif oa_status in ("gold", "hybrid", "green", "bronze"):
            stats["oa_no_pdf"] += 1
        elif oa_status == "not_found":
            stats["not_found"] += 1
        else:
            stats["closed"] += 1

    # Print summary
    print("\n" + "=" * 50)
    print("UNPAYWALL AUDIT SUMMARY")
    print("=" * 50)
    print(f"  Checked          : {len(candidates)}")
    print(f"  Have PDF URL     : {stats['has_pdf']}  ({stats['has_pdf']/len(candidates)*100:.1f}%)")
    print(f"  OA but no PDF    : {stats['oa_no_pdf']}")
    print(f"  Closed access    : {stats['closed']}")
    print(f"  Not in Unpaywall : {stats['not_found']}")
    print(f"  Errors           : {stats['error']}")
    print("\n  OA status breakdown:")
    for status, count in sorted(oa_status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status:<12}: {count}")
    print("=" * 50)
    print(f"\nRun with --download to fetch the {stats['has_pdf']} available PDFs.\n")

    # Save the list of available PDFs for reference
    audit_path = Path("data/metadata/unpaywall_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(available, f, ensure_ascii=False, indent=2)
    log.info("Audit results saved → %s", audit_path)


def download(records: list[dict]) -> None:
    """Download mode: resolve and download all available OA PDFs."""
    candidates = [r for r in records if not r.get("fulltext_downloaded") and r.get("doi")]
    log.info("%d papers to resolve via Unpaywall", len(candidates))

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    resolved = 0

    for record in tqdm(candidates, desc="Unpaywall download"):
        doi = record["doi"]
        result = resolve_unpaywall(doi)

        if not result or not result["pdf_url"]:
            log.debug("No OA PDF for: %s", doi)
            continue

        out_path = PDF_DIR / doi_to_filename(doi)
        if out_path.exists():
            record["fulltext_downloaded"] = True
            record["pdf_path"] = str(out_path)
            record["oa_pdf_url"] = result["pdf_url"]
            resolved += 1
            continue

        if download_pdf(result["pdf_url"], out_path):
            record["fulltext_downloaded"] = True
            record["pdf_path"] = str(out_path)
            record["oa_pdf_url"] = result["pdf_url"]
            record["oa_status"] = result["oa_status"]
            resolved += 1
            log.debug("Downloaded: %s", out_path.name)

    save_metadata(records)
    log.info("Downloaded %d PDFs (%d had no OA version)",
             resolved, len(candidates) - resolved)


def run(mode: str = "audit") -> None:
    if not EMAIL:
        log.error("Set UNPAYWALL_EMAIL in your .env file before running.")
        return

    records = load_metadata()

    if mode == "download":
        download(records)
    else:
        audit(records)


if __name__ == "__main__":
    import sys
    mode = "download" if "--download" in sys.argv else "audit"
    run(mode=mode)
