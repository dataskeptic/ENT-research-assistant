"""Step 1: Query PubMed for recent ENT papers and collect DOIs + metadata.

Uses NCBI E-utilities (free, no registration required).
With NCBI_API_KEY: 10 req/s. Without: 3 req/s.

Outputs: data/metadata/pubmed_results.json
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("NCBI_API_KEY", "")
HARVEST_DAYS = int(os.getenv("HARVEST_DAYS", 30))
MAX_PAPERS = int(os.getenv("MAX_PAPERS", 500))

# MeSH terms covering ENT / Otolaryngology subspecialties
ENT_MESH_QUERY = (
    '("Otolaryngology"[MeSH] OR "Ear Diseases"[MeSH] OR "Nose Diseases"[MeSH] '
    'OR "Throat Diseases"[MeSH] OR "Laryngeal Diseases"[MeSH] '
    'OR "Hearing Loss"[MeSH] OR "Tonsil"[MeSH] OR "Sinusitis"[MeSH] '
    'OR "Otitis"[MeSH] OR "Cochlear Implants"[MeSH] '
    'OR "Head and Neck Neoplasms"[MeSH] OR "Rhinoplasty"[MeSH])'
)


def _build_date_filter() -> str:
    end = datetime.today()
    start = end - timedelta(days=HARVEST_DAYS)
    return f"{start.strftime('%Y/%m/%d')}:{end.strftime('%Y/%m/%d')}[pdat]"


def _ncbi_get(endpoint: str, params: dict) -> dict:
    """GET wrapper with rate-limit handling and optional API key."""
    if API_KEY:
        params["api_key"] = API_KEY
    params["retmode"] = "json"
    delay = 0.11 if API_KEY else 0.34  # stay under rate limits
    time.sleep(delay)
    resp = requests.get(f"{NCBI_BASE}/{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def search_pmids() -> list[str]:
    """Return list of PMIDs matching ENT query within date range."""
    query = f"{ENT_MESH_QUERY} AND {_build_date_filter()}"
    log.info("Searching PubMed: %s", query)

    # First call: get total count
    data = _ncbi_get("esearch.fcgi", {
        "db": "pubmed",
        "term": query,
        "retmax": 0,
        "usehistory": "y",
    })
    total = int(data["esearchresult"]["count"])
    web_env = data["esearchresult"]["webenv"]
    query_key = data["esearchresult"]["querykey"]
    log.info("Total ENT papers found: %d", total)

    retmax = min(total, MAX_PAPERS) if MAX_PAPERS > 0 else total
    pmids = []
    batch = 200

    for start in tqdm(range(0, retmax, batch), desc="Fetching PMIDs"):
        data = _ncbi_get("esearch.fcgi", {
            "db": "pubmed",
            "term": query,
            "retstart": start,
            "retmax": min(batch, retmax - start),
            "webenv": web_env,
            "query_key": query_key,
        })
        pmids.extend(data["esearchresult"].get("idlist", []))

    log.info("Collected %d PMIDs", len(pmids))
    return pmids


def fetch_metadata(pmids: list[str]) -> list[dict]:
    """Fetch metadata (title, authors, DOI, journal, abstract) for each PMID."""
    records = []
    batch = 100

    for i in tqdm(range(0, len(pmids), batch), desc="Fetching metadata"):
        chunk = pmids[i : i + batch]
        params = {
            "db": "pubmed",
            "id": ",".join(chunk),
            "rettype": "abstract",
        }
        if API_KEY:
            params["api_key"] = API_KEY
        time.sleep(0.34)

        resp = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params=params,
            timeout=30,
            headers={"Accept": "application/json"},
        )
        # efetch returns XML for pubmed; parse with summary endpoint instead
        summary_resp = _ncbi_get("esummary.fcgi", {
            "db": "pubmed",
            "id": ",".join(chunk),
        })
        result = summary_resp.get("result", {})
        for pmid in chunk:
            if pmid not in result:
                continue
            item = result[pmid]
            # Extract DOI from articleids list
            doi = None
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value")
                    break
            # Extract PMC id
            pmc_id = None
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "pmc":
                    pmc_id = aid.get("value")
                    break

            records.append({
                "pmid": pmid,
                "doi": doi,
                "pmc_id": pmc_id,
                "title": item.get("title", ""),
                "authors": [a.get("name") for a in item.get("authors", [])],
                "journal": item.get("fulljournalname", ""),
                "pub_date": item.get("pubdate", ""),
                "source": "pubmed",
                "fulltext_downloaded": False,
                "pdf_path": None,
                "xml_path": None,
            })

    log.info("Metadata collected for %d papers", len(records))
    return records


def save_metadata(records: list[dict], out_path: str = "data/metadata/pubmed_results.json") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    log.info("Saved %d records → %s", len(records), out_path)


if __name__ == "__main__":
    pmids = search_pmids()
    records = fetch_metadata(pmids)
    save_metadata(records)
