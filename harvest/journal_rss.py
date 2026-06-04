"""Step 3: Harvest new article DOIs directly from top ENT journal RSS feeds.

This catches papers that may not yet be indexed in PubMed
or that are behind Unpaywall but published in OA journals.

Outputs: data/metadata/rss_results.json
         (merged into pubmed_results.json by run_harvest.py)
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import os

import feedparser
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HARVEST_DAYS = int(os.getenv("HARVEST_DAYS", 30))
EMAIL = os.getenv("UNPAYWALL_EMAIL", "")

# Top ENT / Otolaryngology journals with RSS feeds
ENT_JOURNAL_FEEDS = [
    {
        "name": "Laryngoscope",
        "rss": "https://onlinelibrary.wiley.com/feed/15314995/most-recent",
        "issn": "1531-4995",
    },
    {
        "name": "Otolaryngology–Head and Neck Surgery",
        "rss": "https://journals.sagepub.com/action/showFeed?type=etoc&feed=rss&jc=otohns",
        "issn": "1097-6817",
    },
    {
        "name": "JAMA Otolaryngology–Head & Neck Surgery",
        "rss": "https://jamanetwork.com/rss/site_3/67.xml",
        "issn": "2168-619X",
    },
    {
        "name": "Ear and Hearing",
        "rss": "https://journals.lww.com/ear-hearing/rss",
        "issn": "1538-4667",
    },
    {
        "name": "International Journal of Audiology",
        "rss": "https://www.tandfonline.com/feed/rss/iija20",
        "issn": "1708-8186",
    },
    {
        "name": "Clinical Otolaryngology",
        "rss": "https://onlinelibrary.wiley.com/feed/17494486/most-recent",
        "issn": "1749-4486",
    },
    {
        "name": "European Archives of Oto-Rhino-Laryngology",
        "rss": "https://link.springer.com/search.rss?facet-journal-id=405&query=",
        "issn": "1434-4726",
    },
    {
        "name": "American Journal of Rhinology & Allergy",
        "rss": "https://journals.sagepub.com/action/showFeed?type=etoc&feed=rss&jc=ajra",
        "issn": "1945-8932",
    },
    {
        "name": "Head & Neck",
        "rss": "https://onlinelibrary.wiley.com/feed/10970347/most-recent",
        "issn": "1097-0347",
    },
    {
        "name": "Otology & Neurotology",
        "rss": "https://journals.lww.com/otology-neurotology/rss",
        "issn": "1537-4505",
    },
]


def extract_doi_from_entry(entry: feedparser.FeedParserDict) -> str | None:
    """Try multiple fields to find a DOI in an RSS entry."""
    # Some feeds put DOI in dc_identifier or prism_doi
    for field in ["prism_doi", "dc_identifier"]:
        val = getattr(entry, field, None)
        if val and val.startswith("10."):
            return val

    # Try the link URL — DOIs often appear as https://doi.org/10.xxxx
    link = entry.get("link", "")
    if "doi.org/" in link:
        return link.split("doi.org/")[-1].strip()

    # Try tags
    for tag in entry.get("tags", []):
        term = tag.get("term", "")
        if term.startswith("10."):
            return term

    return None


def parse_feed_date(entry: feedparser.FeedParserDict) -> datetime | None:
    """Parse publication date from feed entry."""
    for field in ["published_parsed", "updated_parsed", "created_parsed"]:
        parsed = getattr(entry, field, None)
        if parsed:
            try:
                return datetime(*parsed[:6])
            except (TypeError, ValueError):
                pass
    return None


def harvest_feed(journal: dict, cutoff: datetime) -> list[dict]:
    """Parse one RSS feed and return articles newer than cutoff."""
    articles = []
    try:
        time.sleep(0.5)
        feed = feedparser.parse(journal["rss"])
        if feed.bozo and not feed.entries:
            log.warning("Feed parse error for %s: %s", journal["name"], feed.bozo_exception)
            return []

        for entry in feed.entries:
            pub_date = parse_feed_date(entry)
            if pub_date and pub_date < cutoff:
                continue  # older than our window

            doi = extract_doi_from_entry(entry)
            articles.append({
                "doi": doi,
                "title": entry.get("title", "").strip(),
                "journal": journal["name"],
                "issn": journal["issn"],
                "pub_date": pub_date.isoformat() if pub_date else "",
                "link": entry.get("link", ""),
                "source": "rss",
                "pmid": None,
                "pmc_id": None,
                "authors": [],
                "fulltext_downloaded": False,
                "pdf_path": None,
                "xml_path": None,
            })
    except Exception as e:
        log.error("Failed to parse feed %s: %s", journal["name"], e)

    return articles


def run() -> list[dict]:
    cutoff = datetime.now() - timedelta(days=HARVEST_DAYS)
    log.info("Harvesting RSS feeds (articles since %s)", cutoff.date())

    all_articles = []
    for journal in tqdm(ENT_JOURNAL_FEEDS, desc="RSS feeds"):
        articles = harvest_feed(journal, cutoff)
        log.info("%s: %d recent articles", journal["name"], len(articles))
        all_articles.extend(articles)

    # Deduplicate by DOI
    seen_dois = set()
    unique = []
    for a in all_articles:
        key = a.get("doi") or a["link"]
        if key and key not in seen_dois:
            seen_dois.add(key)
            unique.append(a)

    log.info("Total unique articles from RSS: %d", len(unique))

    out_path = Path("data/metadata/rss_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    log.info("Saved RSS results → %s", out_path)

    return unique


if __name__ == "__main__":
    run()
