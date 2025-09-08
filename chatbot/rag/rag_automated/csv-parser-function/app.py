import os
import csv
import io
import json
import uuid
import boto3
import logging
from datetime import datetime

s3 = boto3.client('s3')
# s3 = boto3.client('s3', endpoint_url='http://host.docker.internal:4566')

sqs = boto3.client('sqs')
# sqs = boto3.client('sqs', endpoint_url='http://host.docker.internal:4566')

HTML_QUEUE_URL = os.environ['HTML_QUEUE_URL']
PDF_QUEUE_URL = os.environ['PDF_QUEUE_URL']
MULTIMEDIA_QUEUE_URL = os.environ['MULTIMEDIA_QUEUE_URL']

OUTPUT_BUCKET = os.environ['OUTPUT_BUCKET']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VALID_TYPES = {"html", "pdf", "audio", "video"}

def lambda_handler(event, context):
    # Extract bucket and key from the event
    record = event['Records'][0]['s3']
    bucket = record['bucket']['name']
    key = record['object']['key']

    logger.info(f"File to download: {bucket}, {key}")

    # Read CSV from S3
    obj = s3.get_object(Bucket=bucket, Key=key)
    csv_content = obj['Body'].read().decode('utf-8')

    reader = csv.DictReader(io.StringIO(csv_content))
    rows = list(reader)

    # Validate rows
    for i, row in enumerate(rows, start=1):
        if not row.get("title") or not row.get("url") or not row.get("type"):
            raise ValueError(f"CSV validation failed: Missing required field in row {i}: {row}")
        if row["type"].lower() not in VALID_TYPES:
            raise ValueError(f"CSV validation failed: Invalid type '{row['type']}' in row {i}")

    count = 0
    jsonl_created = 0

    for row_idx, row in enumerate(rows, start=1):
        messageType = row['type'].lower()

        # Case 1: transcription already exists -> create JSONL, skip SQS
        if "transcription" in row and row["transcription"]:
            jsonl_obj = {
                "uuid": str(uuid.uuid4()),
                "text": row["transcription"],
                "source": 0,
                "metadata": {
                    "title": row["title"],
                    "url": row["url"],
                    "type": messageType
                }
            }

            now = datetime.now()
            time_str = now.strftime("%H_%M_%S") + f".{now.microsecond}"
            base_name = f"{key}/{now.date()}/{time_str}_{messageType}_with_transcription"
            jsonl_key = f"{base_name}.jsonl"

            s3.put_object(
                Bucket=OUTPUT_BUCKET,
                Key=jsonl_key,
                Body=json.dumps(jsonl_obj).encode("utf-8") + b"\n",
                ContentType="application/json"
            )
            logger.info(f"JSONL file created for row {row_idx} at s3://{OUTPUT_BUCKET}/{jsonl_key}")
            jsonl_created += 1
            continue  # ⬅️ skip SQS for this row

        # Case 2: no transcription -> send SQS
        msg = {
            "csv_input_key": key,
            "job_id": key.split('/')[-2] if '/' in key else str(uuid.uuid4()),
            "item_id": str(uuid.uuid4()),
            "title": row['title'],
            "type": messageType,
            "url": row['url']
        }
        queue_url = ""
        match messageType:
            case "html":
                queue_url = HTML_QUEUE_URL
            case "pdf":
                queue_url = PDF_QUEUE_URL
            case "video" | "audio":
                queue_url = MULTIMEDIA_QUEUE_URL
            case _:
                logger.error("The message type cannot be processed")
                continue

        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(msg))
        logger.info(f"Successfully sent SQS message: {queue_url}, {json.dumps(msg)}")
        count += 1

    logger.info(f"Successfully enqueued: {count}, JSONL files created: {jsonl_created}")
    return {"status": "ok", "enqueued": count, "jsonl_created": jsonl_created}
