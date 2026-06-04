"""Step 2b: Resolve OA PDF URLs via Unpaywall and optionally download them.

Source files
------------
Reads ALL ``*_results.json`` files in ``data/metadata/`` (currently
``pubmed_results.json`` and ``rss_results.json``) and merges them into a
single deduplicated pool before resolving.  After downloading, every source
file is updated independently so no records are lost.

PMC-first guarantee
-------------------
Records that have a ``pmc_id`` but whose XML is **not** yet on disk are
skipped — Unpaywall would give you a PDF of the same paper you could get
for free (and in structured XML form) from PMC.  Run ``pmc_download.py``
first, then re-run this script; those records will be skipped by
``_has_fulltext()`` because their XML will now exist.

Modes
-----

  PRECHECK mode (--precheck):
      Scans ALL DOIs in every source file and reports which ones have a
      downloadable PDF URL on Unpaywall.  Saves a report to
      ``data/metadata/unpaywall_precheck.json``.

  AUDIT mode (default, --audit or no flag):
      Checks DOIs that don't already have full text (neither PDF downloaded
      nor PMC XML retrieved) and prints a coverage summary.
      Saves a report to ``data/metadata/unpaywall_audit.json``.

  DOWNLOAD mode (--download):
      Resolves + downloads PDFs for all papers without full text that
      are not better served by PMC.

Usage
-----
::

    python -m harvest.unpaywall_resolve               # audit only
    python -m harvest.unpaywall_resolve --audit       # same
    python -m harvest.unpaywall_resolve --precheck    # check ALL DOIs
    python -m harvest.unpaywall_resolve --download    # resolve + download

Requires ``UNPAYWALL_EMAIL`` in ``.env`` (no registration needed, just
identification).

Writes: ``data/pdfs/<doi_safe>.pdf``  (download mode only)
Updates ``pdf_path``, ``fulltext_downloaded``, ``oa_pdf_url`` in each
source metadata file (download mode only).
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
METADATA_DIR = Path("data/metadata")
PDF_DIR = Path("data/pdfs")
FULLTEXT_DIR = Path("data/fulltext")

# Source files to merge.  New harvester outputs just need to follow the
# "*_results.json" naming convention to be picked up automatically.
_SOURCE_GLOB = "*_results.json"


# ---------------------------------------------------------------------------
# I/O helpers — Fix A & Fix C
# ---------------------------------------------------------------------------

def load_all_records() -> tuple[list[dict], dict[str, list[dict]]]:
    """Load and merge records from all ``*_results.json`` source files.

    Returns
    -------
    merged : list[dict]
        Deduplicated flat list of all records across all source files.
        Each record carries a private ``_source_file`` key so it can be
        written back to the correct file by ``save_all_records()``.
    by_file : dict[str, list[dict]]
        Original per-file lists (un-deduplicated), keyed by file path
        string — used by ``save_all_records()`` to write clean updates.
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

    # Deduplicate by DOI — keep the first occurrence (PubMed preferred over
    # RSS because PubMed records usually have richer metadata: pmid, authors).
    # Records without a DOI are always kept (can't deduplicate them).
    merged: list[dict] = []
    seen_doi: set[str] = set()
    for rec in all_records:
        doi = (rec.get("doi") or "").strip().lower()
        if doi:
            if doi in seen_doi:
                continue
            seen_doi.add(doi)
        merged.append(rec)

    dupes = len(all_records) - len(merged)
    log.info(
        "Merged pool: %d records from %d files (%d duplicates removed)",
        len(merged), len(source_files), dupes,
    )
    return merged, by_file


def save_all_records(by_file: dict[str, list[dict]]) -> None:
    """Write each source file back with updated metadata.

    The ``_source_file`` key that was injected during loading is stripped
    before writing so the output files stay clean.
    """
    for path_str, records in by_file.items():
        clean = [{k: v for k, v in r.items() if k != "_source_file"} for r in records]
        with open(path_str, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        log.debug("Saved %d records → %s", len(clean), Path(path_str).name)


# ---------------------------------------------------------------------------
# Full-text detection helpers
# ---------------------------------------------------------------------------

def _has_fulltext(record: dict) -> bool:
    """Return True if this record already has any form of full text on disk.

    Covers:
    - PDF downloaded via Unpaywall    → fulltext_downloaded=True AND pdf_path set
    - PMC XML fetched via OAI         → xml_path set
    - PMC PDF downloaded              → fulltext_downloaded=True
    - pmc_id present + XML on disk    → implicit PMC success
    """
    if record.get("fulltext_downloaded") and record.get("pdf_path"):
        if Path(record["pdf_path"]).exists():
            return True

    if record.get("xml_path") and Path(record["xml_path"]).exists():
        return True

    if record.get("fulltext_path") and Path(record["fulltext_path"]).exists():
        return True

    pmc_id = record.get("pmc_id", "")
    if pmc_id:
        xml_path = FULLTEXT_DIR / f"{pmc_id}.xml"
        if xml_path.exists():
            return True

    return False


def _pmc_pending(record: dict) -> bool:
    """Fix B — PMC-first guard.

    Return True if this record has a ``pmc_id`` but the XML has NOT yet
    been downloaded.  Such records should be processed by ``pmc_download.py``
    first; sending them to Unpaywall would waste an API call and download a
    PDF of a paper that is available in superior structured-XML form.
    """
    pmc_id = record.get("pmc_id", "")
    if not pmc_id:
        return False  # no PMC ID — Unpaywall is the right path
    xml_path = FULLTEXT_DIR / f"{pmc_id}.xml"
    return not xml_path.exists()  # has pmc_id but XML missing → PMC pending


# ---------------------------------------------------------------------------
# Unpaywall API
# ---------------------------------------------------------------------------

def doi_to_filename(doi: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", doi) + ".pdf"


def resolve_unpaywall(doi: str) -> dict | None:
    """Query Unpaywall for a DOI.  Returns a dict with OA info or None on failure.

    Dict shape::

        {
            "pdf_url":   str | None,
            "oa_status": str,           # 'gold', 'hybrid', 'green', 'bronze', 'closed'
            "host_type": str | None,    # 'publisher', 'repository'
            "version":   str | None,    # 'publishedVersion', 'acceptedVersion', …
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
    """Download a PDF from a URL.  Returns True on success."""
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


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def precheck(records: list[dict]) -> None:
    """Pre-check mode: scan ALL DOIs (regardless of download status)."""
    with_doi = [r for r in records if r.get("doi")]
    no_doi = len(records) - len(with_doi)

    pmc_pending_count = sum(1 for r in with_doi if _pmc_pending(r))

    log.info("=== Unpaywall Pre-Check Mode ===")
    log.info("Total records        : %d", len(records))
    log.info("Have DOI             : %d", len(with_doi))
    log.info("No DOI (skip)        : %d", no_doi)
    log.info("PMC pending (skip)   : %d  ← run pmc_download.py first", pmc_pending_count)

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
        pending = _pmc_pending(record)

        entry = {
            "doi": doi,
            "title": record.get("title", ""),
            "source": record.get("source", "pubmed"),
            "pmc_id": record.get("pmc_id", ""),
            "pmc_pending": pending,
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
    new_downloadable = stats["has_pdf"] - already_covered - pmc_pending_count

    by_source: dict[str, int] = {}
    for e in available:
        s = e.get("source", "pubmed")
        by_source[s] = by_source.get(s, 0) + 1

    print("\n" + "=" * 58)
    print("UNPAYWALL PRE-CHECK SUMMARY (all DOIs, all sources)")
    print("=" * 58)
    print(f"  Total with DOI          : {len(with_doi)}")
    print(f"  Have PDF URL            : {stats['has_pdf']}  ({stats['has_pdf']/max(len(with_doi),1)*100:.1f}%)")
    print(f"    → already have text   : {already_covered}")
    print(f"    → PMC pending (skip)  : {pmc_pending_count}")
    print(f"    → NEW to download     : {max(new_downloadable, 0)}")
    print(f"  OA but no PDF           : {stats['oa_no_pdf']}")
    print(f"  Closed access           : {stats['closed']}")
    print(f"  Not in Unpaywall        : {stats['not_found']}")
    print(f"  Errors                  : {stats['error']}")
    print("\n  PDF availability by source:")
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"    {src:<14}: {cnt}")
    print("\n  OA status breakdown:")
    for status, count in sorted(oa_status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status:<14}: {count}")
    print("=" * 58)
    print(f"\nRun with --download to fetch the {max(new_downloadable, 0)} new PDFs.\n")

    out_path = METADATA_DIR / "unpaywall_precheck.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"available": available, "unavailable": unavailable}, f, ensure_ascii=False, indent=2)
    log.info("Pre-check results saved \u2192 %s", out_path)


def audit(records: list[dict]) -> None:
    """Audit mode: check Unpaywall coverage for papers WITHOUT full text."""
    # Fix B — exclude PMC-pending records from the audit pool
    candidates = [
        r for r in records
        if not _has_fulltext(r) and r.get("doi") and not _pmc_pending(r)
    ]
    already_done = sum(1 for r in records if _has_fulltext(r))
    no_doi = sum(1 for r in records if not r.get("doi"))
    pmc_pending_count = sum(1 for r in records if _pmc_pending(r))

    log.info("=== Unpaywall Audit Mode ===")
    log.info("Total records      : %d", len(records))
    log.info("Already have text  : %d  (PDF + PMC XML)", already_done)
    log.info("No DOI (skip)      : %d", no_doi)
    log.info("PMC pending (skip) : %d  ← run pmc_download.py first", pmc_pending_count)
    log.info("To check           : %d", len(candidates))

    if not candidates:
        log.info("Nothing to check — all records already have full text or are PMC-pending.")
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
                "source": record.get("source", "pubmed"),
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

    print("\n" + "=" * 52)
    print("UNPAYWALL AUDIT SUMMARY")
    print("=" * 52)
    print(f"  Checked            : {len(candidates)}")
    print(f"  PMC-pending (skip) : {pmc_pending_count}")
    print(f"  Have PDF URL       : {stats['has_pdf']}  ({stats['has_pdf']/max(len(candidates),1)*100:.1f}%)")
    print(f"  OA but no PDF      : {stats['oa_no_pdf']}")
    print(f"  Closed access      : {stats['closed']}")
    print(f"  Not in Unpaywall   : {stats['not_found']}")
    print(f"  Errors             : {stats['error']}")
    print("\n  OA status breakdown:")
    for status, count in sorted(oa_status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status:<12}: {count}")
    print("=" * 52)
    print(f"\nRun with --download to fetch the {stats['has_pdf']} available PDFs.\n")

    audit_path = METADATA_DIR / "unpaywall_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(available, f, ensure_ascii=False, indent=2)
    log.info("Audit results saved \u2192 %s", audit_path)


def download(records: list[dict], by_file: dict[str, list[dict]]) -> None:
    """Download mode: resolve + download OA PDFs, skipping PMC-pending records."""
    # Fix B — only process records with no PMC path available
    candidates = [
        r for r in records
        if not _has_fulltext(r) and r.get("doi") and not _pmc_pending(r)
    ]

    pmc_pending_count = sum(1 for r in records if _pmc_pending(r))
    if pmc_pending_count:
        log.warning(
            "Skipping %d records that have a pmc_id but no XML on disk. "
            "Run pmc_download.py first to get the structured XML for those.",
            pmc_pending_count,
        )

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

    # Fix C — write back to each source file separately
    save_all_records(by_file)
    log.info(
        "Downloaded %d PDFs (%d had no OA version, %d skipped for PMC)",
        resolved, len(candidates) - resolved, pmc_pending_count,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(mode: str = "audit") -> None:
    if not EMAIL:
        log.error("Set UNPAYWALL_EMAIL in your .env file before running.")
        return

    records, by_file = load_all_records()
    if not records:
        log.error("No records loaded. Run pubmed_harvest.py or journal_rss.py first.")
        return

    if mode == "precheck":
        precheck(records)
    elif mode == "download":
        download(records, by_file)
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
