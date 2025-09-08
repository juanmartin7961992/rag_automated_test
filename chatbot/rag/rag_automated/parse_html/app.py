from __future__ import annotations
from datetime import datetime
import json
import os
from urllib.parse import urlparse
import uuid
import boto3
from bs4 import BeautifulSoup
import pandas as pd
from trafilatura import fetch_url
from unidecode import unidecode
import io
import logging
from dataclasses import dataclass, asdict
from typing import Callable

# ------------------ Setup ------------------ #
s3 = boto3.client("s3")
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------ Helpers ------------------ #
class ScraperError(Exception):
    """Custom exception for scraper errors."""


def normalize_hostname(url: str) -> str:
    """Return a hostname without scheme, port, or leading www."""
    hostname = urlparse(url).hostname or ""
    return hostname.lower().removeprefix("www.")


def fetch_and_select(url: str, selector: str) -> str:
    """Fetch HTML, parse with BeautifulSoup, and return text from selector."""
    downloaded = fetch_url(url)
    if not downloaded:
        raise ScraperError(f"No content fetched for {url}")

    soup = BeautifulSoup(downloaded, "html.parser")
    element = soup.select_one(selector)
    if not element:
        raise ScraperError(f"No matching block found for {url} with {selector}")

    return element.get_text(separator="\n", strip=True)


# ------------------ Scrapers ------------------ #
def scrape_allinforinclusiveed(url: str) -> str:
    return fetch_and_select(url, "div.sqs-html-content[data-sqsp-text-block-content]")


def scrape_neuhaus(url: str) -> str:
    return fetch_and_select(url, "article")


SCRAPER_MAP: dict[str, Callable[[str], str]] = {
    "allinforinclusiveed.org": scrape_allinforinclusiveed,
    "neuhaus.org": scrape_neuhaus,
}


def extract_text(url: str) -> dict | None:
    """Route a URL to the proper scraper based on hostname."""
    hostname = normalize_hostname(url)
    scraper = SCRAPER_MAP.get(hostname)

    if not scraper:
        logger.error(f"Hostname not allowed: {hostname}. Add to SCRAPER_MAP.")
        raise Exception(f"Hostname not allowed: {hostname}. Add to SCRAPER_MAP.")

    try:
        clean_text = scraper(url)
    except ScraperError as e:
        logger.error(str(e))
        raise

    return {"cleantext": clean_text}


# ------------------ Document Model ------------------ #
@dataclass
class Document:
    uuid: str
    text: str
    source: int
    metadata: dict
    csvInputKey: str

    @classmethod
    def from_raw(cls, cleantext: str, title: str, url: str, csv_key: str, source: int) -> "Document":
        return cls(
            uuid=str(uuid.uuid4()),
            text=unidecode(cleantext),
            source=source,
            metadata={
                "title": unidecode(title),
                "url": url,
                "type": "html",
            },
            csvInputKey=csv_key,
        )


# ------------------ Output Writers ------------------ #
def write_jsonl(docs: list[Document]) -> str:
    buffer = io.StringIO()
    for doc in docs:
        json.dump(asdict(doc), buffer)
        buffer.write("\n")
    return buffer.getvalue()


def write_csv(docs: list[Document]) -> str:
    records = [{"id": d.source, "url": d.metadata["url"], "title": d.metadata["title"]} for d in docs]
    buffer = io.StringIO()
    pd.DataFrame(records).drop_duplicates().to_csv(buffer, index=False)
    return buffer.getvalue()


def upload_to_s3(base_name: str, jsonl_data: str) -> None:
    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=f"{base_name}.jsonl",
        Body=jsonl_data,
        ContentType="application/json",
    )


# ------------------ Lambda Handler ------------------ #
def lambda_handler(event, context):
    logger.info("Start HTML parse function")

    # Parse SQS records into DataFrame
    sqs_records = event.get("Records", [])
    records_list = [
        {
            "CsvInputKey": body.get("csv_input_key", ""),
            "Name": body.get("title", ""),
            "URL": body.get("url", ""),
        }
        for record in sqs_records
        for body in [json.loads(record["body"])]
    ]
    df = pd.DataFrame(records_list)

    # Extract documents
    docs: list[Document] = []
    for _, row in df.iterrows():
        extracted = extract_text(row["URL"])
        if not extracted:
            continue
        docs.append(
            Document.from_raw(
                extracted["cleantext"], row["Name"], row["URL"], row["CsvInputKey"], len(docs)
            )
        )

    if not docs:
        return {"statusCode": 204, "body": json.dumps({"message": "No documents processed"})}

    # Build base filename
    now = datetime.now()
    time_str = f"{now.strftime('%H_%M_%S')}.{now.microsecond}"
    base_name = f"{docs[0].csvInputKey}/{now.date()}/{time_str}_html"

    # Write + upload
    upload_to_s3(base_name, write_jsonl(docs))

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "Scraping completed",
                "jsonl_file": f"{base_name}.jsonl",
                "csv_file": f"{base_name}_records.csv",
                "processed": len(docs),
            }
        ),
    }


# ------------------ Local Debug ------------------ #
if __name__ == "__main__":
    # test_url = "https://neuhaus.org/addressing-dyslexia-in-the-classroom/"
    test_url = "https://www.allinforinclusiveed.org/podcastarchive/episode-1-inclusive-education-in-new-jersey"
    result = extract_text(test_url)
    print(result and result["cleantext"][:300])
