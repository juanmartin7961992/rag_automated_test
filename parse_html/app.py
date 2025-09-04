from datetime import datetime
import json
import os
import uuid
import boto3
import pandas as pd
from trafilatura import fetch_url, extract
from unidecode import unidecode
import io
import logging

s3 = boto3.client('s3')
# s3 = boto3.client('s3', endpoint_url='http://host.docker.internal:4566')
output_bucket = os.getenv("OUTPUT_BUCKET") 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_one(url):
    """Fetches and extracts text from a URL using trafilatura."""
    try:
        downloaded = fetch_url(url)
        if not downloaded:
            return None
        result = json.loads(extract(downloaded, output_format="json", include_links=True))
        result["html"] = downloaded
        result["cleantext"] = extract(downloaded)  # plain text
        return result
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        raise  # propagate to Lambda so it fails

def lambda_handler(event, context):
    # Extract records from SQS event
    logger.info(f"Start HTML parse function")
    sqs_records = event['Records']
    logger.info(f"Received records: {len(sqs_records)}")
    logger.info(f"Records: {sqs_records}")
    
    # Convert SQS messages into a DataFrame-like structure
    records_list = []
    for record in sqs_records:
        body = json.loads(record['body'])
        # Expect body to contain "title" and "url"
        records_list.append({
            "CsvInputKey": body.get("csv_input_key", ""),
            "Name": body.get("title", ""),
            "URL": body.get("url", "")
        })
    
    df = pd.DataFrame(records_list)

    OFFSET = 0  # default, could be passed via event if needed

    # Fetch and extract URLs
    outs1 = []
    for url in df["URL"]:
        out = get_one(url)
        outs1.append(out)

    # Attach metadata
    for o, url, title, csvInputKey in zip(outs1, df["URL"], df["Name"], df["CsvInputKey"]):
        if o:
            o["url"] = url
            o["title"] = title
            o["csvInputKey"] = csvInputKey

    # Build structured docs
    docs = []
    for idx, o in enumerate(outs1):
        if not o:
            continue
        csvInputKey = o.get("csvInputKey", "")
        tmp = {
            "uuid": str(uuid.uuid4()),
            "text": unidecode(o.get("cleantext", "")),
            "source": len(docs) + OFFSET,
            "metadata": {
                "title": unidecode(o.get("title", "")),
                "url": o.get("url", ""),
                "type": "html",
            }
        }
        docs.append(tmp)

    # Prepare output JSONL in-memory
    jsonl_buffer = io.StringIO()
    for doc in docs:
        json.dump(doc, jsonl_buffer)
        jsonl_buffer.write("\n")
    jsonl_buffer.seek(0)

    # Prepare CSV of records in-memory
    records_out = [{"id": d["source"], "url": d["metadata"]["url"], "title": d["metadata"]["title"]} for d in docs]
    csv_buffer = io.StringIO()
    pd.DataFrame(records_out).drop_duplicates().to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Save outputs back to S3
    now = datetime.now()
    time_str = now.strftime("%H_%M_%S") + f".{now.microsecond}"
    base_name = f"{csvInputKey}/{now.date()}/{time_str}_html"
    s3.put_object(Bucket=output_bucket, Key=f"{base_name}.jsonl", Body=jsonl_buffer.getvalue(), ContentType="application/json")
    # s3.put_object(Bucket=output_bucket, Key=f"{base_name}_records.csv", Body=csv_buffer.getvalue(), ContentType="text/csv")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Scraping completed",
            "jsonl_file": f"{base_name}.jsonl",
            "csv_file": f"{base_name}_records.csv",
            "processed": len(docs)
        })
    }
