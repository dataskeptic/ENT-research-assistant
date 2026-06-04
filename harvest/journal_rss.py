"""Step 3: Harvest new article DOIs directly from top ENT journal RSS feeds.

Fixes:
- JAMA Otolaryngology feed corrected to journal-specific RSS (not general JAMA)
- Springer DOI extracted from article URL path (e.g. /article/10.1007/...)
- Generic URL-path DOI fallback for any journal not putting DOI in feed metadata

Outputs: data/metadata/rss_results.json
         (merged into pubmed_results.json by run_harvest.py)
"""

import json
import logging
import re
import time
from pathlib import Path
from datetime import datetime, timedelta
import os

import feedparser
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HARVEST_DAYS = int(os.getenv("HARVEST_DAYS", 30))

# Regex to find a DOI anywhere in a string
_DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s"<>]+)')

# Top ENT / Otolaryngology journals with RSS feeds
# JAMA OTO: corrected to jamanetwork.com/rss/site_19/68.xml (JAMA Otolaryngology-HNS specific)
ENT_JOURNAL_FEEDS = [
    {
        "name": "Laryngoscope",
        "rss": "https://onlinelibrary.wiley.com/feed/15314995/most-recent",
        "issn": "1531-4995",
        "doi_from_link": False,
    },
    {
        "name": "Otolaryngology\u2013Head and Neck Surgery",
        "rss": "https://journals.sagepub.com/action/showFeed?type=etoc&feed=rss&jc=otohns",
        "issn": "1097-6817",
        "doi_from_link": False,
    },
    {
        # Corrected: JAMA Otolaryngology-Head & Neck Surgery specific feed
        # site_19 = JAMA Network Otolaryngology journals group, feed 68 = OTO-HNS
        "name": "JAMA Otolaryngology\u2013Head & Neck Surgery",
        "rss": "https://jamanetwork.com/rss/site_19/68.xml",
        "issn": "2168-619X",
        "doi_from_link": True,  # JAMA links are https://jamanetwork.com/journals/jamaotolaryngology/...
    },
    {
        "name": "Ear and Hearing",
        "rss": "https://journals.lww.com/ear-hearing/rss",
        "issn": "1538-4667",
        "doi_from_link": False,
    },
    {
        "name": "International Journal of Audiology",
        "rss": "https://www.tandfonline.com/feed/rss/iija20",
        "issn": "1708-8186",
        "doi_from_link": False,
    },
    {
        "name": "Clinical Otolaryngology",
        "rss": "https://onlinelibrary.wiley.com/feed/17494486/most-recent",
        "issn": "1749-4486",
        "doi_from_link": False,
    },
    {
        # Springer: DOI lives in the article URL path as /article/10.xxxx/yyyyy
        "name": "European Archives of Oto-Rhino-Laryngology",
        "rss": "https://link.springer.com/search.rss?facet-journal-id=405&query=",
        "issn": "1434-4726",
        "doi_from_link": True,  # extract from https://link.springer.com/article/10.1007/...
    },
    {
        "name": "American Journal of Rhinology & Allergy",
        "rss": "https://journals.sagepub.com/action/showFeed?type=etoc&feed=rss&jc=ajra",
        "issn": "1945-8932",
        "doi_from_link": False,
    },
    {
        "name": "Head & Neck",
        "rss": "https://onlinelibrary.wiley.com/feed/10970347/most-recent",
        "issn": "1097-0347",
        "doi_from_link": False,
    },
    {
        "name": "Otology & Neurotology",
        "rss": "https://journals.lww.com/otology-neurotology/rss",
        "issn": "1537-4505",
        "doi_from_link": False,
    },
    {
        # Laryngoscope Investigative Otolaryngology (OA sister journal)
        "name": "Laryngoscope Investigative Otolaryngology",
        "rss": "https://onlinelibrary.wiley.com/feed/23787880/most-recent",
        "issn": "2378-7880",
        "doi_from_link": False,
    },
]


def extract_doi(entry: feedparser.FeedParserDict, doi_from_link: bool = False) -> str | None:
    """Extract DOI from a feed entry using multiple strategies.

    Strategy order:
    1. Structured feed fields (prism:doi, dc:identifier)
    2. doi.org URL in the link field
    3. Regex scan of link URL path (catches Springer, JAMA patterns)
    4. Regex scan of entry summary/content (last resort)
    """
    # 1. Structured metadata fields
    for field in ["prism_doi", "dc_identifier"]:
        val = getattr(entry, field, None)
        if val:
            m = _DOI_RE.search(val)
            if m:
                return m.group(1).rstrip(".")

    link = entry.get("link", "")

    # 2. doi.org resolver URL
    if "doi.org/" in link:
        doi = link.split("doi.org/")[-1].strip().rstrip("/")
        if doi.startswith("10."):
            return doi

    # 3. DOI embedded in URL path (Springer: /article/10.1007/s00405-026-xxxxx)
    if doi_from_link or "springer.com" in link or "jamanetwork.com" in link:
        m = _DOI_RE.search(link)
        if m:
            return m.group(1).rstrip(".")

    # 4. Scan entry tags
    for tag in entry.get("tags", []):
        term = tag.get("term", "")
        if term.startswith("10."):
            return term.rstrip(".")

    # 5. Last resort: regex scan summary text
    summary = entry.get("summary", "")
    m = _DOI_RE.search(summary)
    if m:
        return m.group(1).rstrip(".")

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
    doi_from_link = journal.get("doi_from_link", False)
    try:
        time.sleep(0.5)
        feed = feedparser.parse(journal["rss"])
        if feed.bozo and not feed.entries:
            log.warning("Feed parse error for %s: %s", journal["name"], feed.bozo_exception)
            return []

        no_doi_count = 0
        for entry in feed.entries:
            pub_date = parse_feed_date(entry)
            if pub_date and pub_date < cutoff:
                continue

            doi = extract_doi(entry, doi_from_link=doi_from_link)
            if not doi:
                no_doi_count += 1

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

        if no_doi_count:
            log.warning("%s: %d/%d entries had no extractable DOI",
                        journal["name"], no_doi_count, len(articles))
    except Exception as e:
        log.error("Failed to parse feed %s: %s", journal["name"], e)

    return articles


def run() -> list[dict]:
    cutoff = datetime.now() - timedelta(days=HARVEST_DAYS)
    log.info("Harvesting RSS feeds (articles since %s)", cutoff.date())

    all_articles = []
    for journal in tqdm(ENT_JOURNAL_FEEDS, desc="RSS feeds"):
        articles = harvest_feed(journal, cutoff)
        with_doi = sum(1 for a in articles if a["doi"])
        log.info("%s: %d articles, %d with DOI", journal["name"], len(articles), with_doi)
        all_articles.extend(articles)

    # Deduplicate: prefer DOI key, fallback to link
    seen: set[str] = set()
    unique = []
    for a in all_articles:
        key = a.get("doi") or a["link"]
        if key and key not in seen:
            seen.add(key)
            unique.append(a)

    with_doi_total = sum(1 for a in unique if a["doi"])
    log.info("Total unique articles: %d | with DOI: %d | no DOI: %d",
             len(unique), with_doi_total, len(unique) - with_doi_total)

    out_path = Path("data/metadata/rss_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    log.info("Saved RSS results \u2192 %s", out_path)

    return unique


if __name__ == "__main__":
    run()
