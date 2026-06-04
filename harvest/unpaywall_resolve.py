"""Step 2b: Resolve OA PDF URLs via Unpaywall and optionally download them.

Modes:

  PRECHECK mode (--precheck):
      Scans ALL DOIs in the metadata file and reports which ones have a
      downloadable PDF URL on Unpaywall — BEFORE committing to any download.
      This is useful when you want to see coverage across your entire corpus,
      not just the subset that hasn't been downloaded yet.
      Saves a report to data/metadata/unpaywall_precheck.json.

  AUDIT mode (default, --audit or no flag):
      Checks DOIs that don't already have full text (neither PDF downloaded
      nor PMC XML retrieved) and prints a coverage summary.
      Saves a report to data/metadata/unpaywall_audit.json.

  DOWNLOAD mode (--download):
      Resolves + downloads PDFs for all papers without full text.

Usage:
    python -m harvest.unpaywall_resolve               # audit only
    python -m harvest.unpaywall_resolve --audit       # same
    python -m harvest.unpaywall_resolve --precheck    # check ALL DOIs for PDF links
    python -m harvest.unpaywall_resolve --download    # resolve + download

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


def _has_fulltext(record: dict) -> bool:
    """Return True if this record already has any form of full text.

    Covers:
      - PDF downloaded via Unpaywall    → fulltext_downloaded = True  AND  pdf_path set
      - PMC XML fetched via OAI         → xml_path set  (actual field name in metadata)
      - PMC PDF downloaded              → fulltext_downloaded = True  (set by pmc_download --pdf)
      - Legacy: pmc_downloaded flag     → pmc_downloaded = True

    Note: fulltext_downloaded alone is not enough — we check that the file
    actually exists to avoid counting stale metadata from failed runs.
    """
    # PDF on disk (Unpaywall or PMC PDF)
    if record.get("fulltext_downloaded") and record.get("pdf_path"):
        if Path(record["pdf_path"]).exists():
            return True

    # PMC XML on disk (field name used in actual metadata schema)
    if record.get("xml_path") and Path(record["xml_path"]).exists():
        return True

    # Legacy fulltext_path field
    if record.get("fulltext_path") and Path(record["fulltext_path"]).exists():
        return True

    # pmc_downloaded flag + pmc_id → check for xml file
    if record.get("pmc_downloaded") or record.get("pmc_id"):
        pmc_id = record.get("pmc_id", "")
        if pmc_id:
            xml_path = Path("data/fulltext") / f"{pmc_id}.xml"
            if xml_path.exists():
                return True

    return False


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


def precheck(records: list[dict]) -> None:
    """Pre-check mode: scan ALL DOIs (regardless of download status) for Unpaywall PDF links."""
    with_doi = [r for r in records if r.get("doi")]
    no_doi = len(records) - len(with_doi)

    log.info("=== Unpaywall Pre-Check Mode ===")
    log.info("Total records : %d", len(records))
    log.info("Have DOI      : %d", len(with_doi))
    log.info("No DOI (skip) : %d", no_doi)

    stats = {"has_pdf": 0, "oa_no_pdf": 0, "closed": 0, "not_found": 0, "error": 0}
    oa_status_counts: dict[str, int] = {}
    available: list[dict] = []
    unavailable: list[dict] = []

    for record in tqdm(with_doi, desc="Pre-checking Unpaywall"):
        doi = record["doi"]
        result = resolve_unpaywall(doi)

        if result is None:
            stats["error"] += 1
            continue

        oa_status = result["oa_status"]
        oa_status_counts[oa_status] = oa_status_counts.get(oa_status, 0) + 1
        already = _has_fulltext(record)

        entry = {
            "doi": doi,
            "title": record.get("title", ""),
            "pmc_id": record.get("pmc_id", ""),
            "already_have_fulltext": already,
            "pdf_url": result["pdf_url"],
            "oa_status": oa_status,
            "host_type": result["host_type"],
            "version": result["version"],
        }

        if result["pdf_url"]:
            stats["has_pdf"] += 1
            available.append(entry)
        elif oa_status in ("gold", "hybrid", "green", "bronze"):
            stats["oa_no_pdf"] += 1
            unavailable.append(entry)
        elif oa_status == "not_found":
            stats["not_found"] += 1
            unavailable.append(entry)
        else:
            stats["closed"] += 1
            unavailable.append(entry)

    already_covered = sum(1 for e in available if e["already_have_fulltext"])
    new_downloadable = stats["has_pdf"] - already_covered

    print("\n" + "=" * 55)
    print("UNPAYWALL PRE-CHECK SUMMARY (all DOIs)")
    print("=" * 55)
    print(f"  Total with DOI         : {len(with_doi)}")
    print(f"  Have PDF URL           : {stats['has_pdf']}  ({stats['has_pdf']/len(with_doi)*100:.1f}%)")
    print(f"    → already have text  : {already_covered}")
    print(f"    → NEW to download    : {new_downloadable}")
    print(f"  OA but no PDF          : {stats['oa_no_pdf']}")
    print(f"  Closed access          : {stats['closed']}")
    print(f"  Not in Unpaywall       : {stats['not_found']}")
    print(f"  Errors                 : {stats['error']}")
    print("\n  OA status breakdown:")
    for status, count in sorted(oa_status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status:<14}: {count}")
    print("=" * 55)
    print(f"\nRun with --download to fetch the {new_downloadable} new PDFs.\n")

    out_path = Path("data/metadata/unpaywall_precheck.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"available": available, "unavailable": unavailable}, f, ensure_ascii=False, indent=2)
    log.info("Pre-check results saved → %s", out_path)


def audit(records: list[dict]) -> None:
    """Audit mode: check Unpaywall coverage for papers WITHOUT full text."""
    candidates = [r for r in records if not _has_fulltext(r) and r.get("doi")]
    already_done = sum(1 for r in records if _has_fulltext(r))
    no_doi = sum(1 for r in records if not r.get("doi"))

    log.info("=== Unpaywall Audit Mode ===")
    log.info("Total records    : %d", len(records))
    log.info("Already have text: %d  (PDF + PMC XML)", already_done)
    log.info("No DOI (skip)    : %d", no_doi)
    log.info("To check         : %d", len(candidates))

    if not candidates:
        log.info("Nothing to check — all records already have full text.")
        log.info("Tip: run with --precheck to survey ALL DOIs for downloadable links.")
        return

    stats = {"has_pdf": 0, "oa_no_pdf": 0, "closed": 0, "not_found": 0, "error": 0}
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

    print("\n" + "=" * 50)
    print("UNPAYWALL AUDIT SUMMARY")
    print("=" * 50)
    print(f"  Checked          : {len(candidates)}")
    print(f"  Have PDF URL     : {stats['has_pdf']}  ({stats['has_pdf']/max(len(candidates),1)*100:.1f}%)")
    print(f"  OA but no PDF    : {stats['oa_no_pdf']}")
    print(f"  Closed access    : {stats['closed']}")
    print(f"  Not in Unpaywall : {stats['not_found']}")
    print(f"  Errors           : {stats['error']}")
    print("\n  OA status breakdown:")
    for status, count in sorted(oa_status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status:<12}: {count}")
    print("=" * 50)
    print(f"\nRun with --download to fetch the {stats['has_pdf']} available PDFs.\n")

    audit_path = Path("data/metadata/unpaywall_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(available, f, ensure_ascii=False, indent=2)
    log.info("Audit results saved → %s", audit_path)


def download(records: list[dict]) -> None:
    """Download mode: resolve and download all available OA PDFs (skips existing full text)."""
    candidates = [r for r in records if not _has_fulltext(r) and r.get("doi")]
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

    if mode == "precheck":
        precheck(records)
    elif mode == "download":
        download(records)
    else:
        audit(records)


if __name__ == "__main__":
    import sys
    if "--precheck" in sys.argv:
        mode = "precheck"
    elif "--download" in sys.argv:
        mode = "download"
    else:
        mode = "audit"
    run(mode=mode)
