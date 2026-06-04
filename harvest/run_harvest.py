"""Orchestrator: runs the full ENT literature harvest pipeline.

Order:
  1. pubmed_harvest.py  — collect DOIs + metadata from PubMed
  2. journal_rss.py     — collect DOIs from top ENT journal RSS feeds
  3. merge              — merge PubMed + RSS results, deduplicate by DOI
  4. pmc_download.py    — download full text for papers in PMC OA
  5. unpaywall_resolve.py — resolve remaining papers via Unpaywall

Run this script on a monthly cron job:
  0 6 1 * * /usr/bin/python /path/to/harvest/run_harvest.py
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MERGED_PATH = "data/metadata/pubmed_results.json"


def merge_rss_into_metadata(rss_records: list[dict], pubmed_records: list[dict]) -> list[dict]:
    """Merge RSS results into PubMed results, deduplicating by DOI."""
    existing_dois = {r["doi"] for r in pubmed_records if r.get("doi")}
    existing_pmids = {r["pmid"] for r in pubmed_records if r.get("pmid")}

    added = 0
    for record in rss_records:
        doi = record.get("doi")
        if doi and doi in existing_dois:
            continue
        pubmed_records.append(record)
        if doi:
            existing_dois.add(doi)
        added += 1

    log.info("Merged %d new articles from RSS (not in PubMed results)", added)
    return pubmed_records


def print_summary(records: list[dict]) -> None:
    total = len(records)
    with_doi = sum(1 for r in records if r.get("doi"))
    with_pmc = sum(1 for r in records if r.get("pmc_id"))
    downloaded = sum(1 for r in records if r["fulltext_downloaded"])
    log.info("=" * 50)
    log.info("HARVEST SUMMARY")
    log.info("  Total papers:        %d", total)
    log.info("  With DOI:            %d", with_doi)
    log.info("  With PMC ID:         %d", with_pmc)
    log.info("  Full text obtained:  %d (%.0f%%)", downloaded, 100 * downloaded / total if total else 0)
    log.info("  Abstract only:       %d", total - downloaded)
    log.info("=" * 50)


def run() -> None:
    log.info("Starting ENT literature harvest pipeline")

    # Step 1: PubMed
    log.info("--- Step 1: PubMed harvest ---")
    from harvest.pubmed_harvest import search_pmids, fetch_metadata, save_metadata
    pmids = search_pmids()
    pubmed_records = fetch_metadata(pmids)

    # Step 2: RSS feeds
    log.info("--- Step 2: Journal RSS feeds ---")
    from harvest.journal_rss import run as rss_run
    rss_records = rss_run()

    # Step 3: Merge
    log.info("--- Step 3: Merging results ---")
    merged = merge_rss_into_metadata(rss_records, pubmed_records)
    save_metadata(merged, MERGED_PATH)
    log.info("Total merged records: %d", len(merged))

    # Step 4: PMC OA download
    log.info("--- Step 4: PMC Open Access download ---")
    from harvest.pmc_download import run as pmc_run
    pmc_run()

    # Step 5: Unpaywall
    log.info("--- Step 5: Unpaywall PDF resolution ---")
    from harvest.unpaywall_resolve import run as unpaywall_run
    unpaywall_run()

    # Final summary
    with open(MERGED_PATH, encoding="utf-8") as f:
        final_records = json.load(f)
    print_summary(final_records)
    log.info("Harvest complete. Run ingest pipeline next to embed into ChromaDB.")


if __name__ == "__main__":
    run()
