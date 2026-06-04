# ENT Research Assistant

RAG-based knowledge base for ENT (Otolaryngology) medical literature.

## Project Structure

```
ENT-research-assistant/
├── harvest/
│   ├── pubmed_harvest.py       # Collect DOIs + metadata from PubMed
│   ├── pmc_download.py         # Download full text from PMC Open Access
│   ├── unpaywall_resolve.py    # Resolve remaining DOIs via Unpaywall
│   ├── journal_rss.py          # Harvest DOIs from top ENT journal RSS feeds
│   └── run_harvest.py          # Orchestrator: runs full pipeline
├── data/
│   ├── metadata/               # JSON metadata per paper
│   ├── pdfs/                   # Downloaded PDFs
│   └── fulltext/               # PMC XML full texts
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your email for Unpaywall (required)
```

## Running the Harvester

```bash
# Full pipeline (last 30 days of ENT literature)
python harvest/run_harvest.py

# Individual steps
python harvest/pubmed_harvest.py      # Step 1: collect DOIs
python harvest/pmc_download.py        # Step 2a: PMC OA full text
python harvest/unpaywall_resolve.py   # Step 2b: resolve remaining
python harvest/journal_rss.py         # Step 3: top ENT journals RSS
```

## Data Sources

| Source | What it provides |
|---|---|
| PubMed E-utilities | DOIs + metadata for all indexed ENT papers |
| PMC Open Access | Full text XML/PDF for OA papers |
| Unpaywall | Legal OA PDF URLs for non-PMC papers |
| Journal RSS feeds | Real-time DOIs from top ENT journals |
