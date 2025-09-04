from datetime import datetime
import os
import json
import uuid
import boto3
import requests
import PyPDF2
from unidecode import unidecode
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
# s3 = boto3.client('s3', endpoint_url='http://host.docker.internal:4566')
output_bucket = os.getenv("OUTPUT_BUCKET") 
print("============================================", output_bucket)

# Optional: set default offset via env var
ID_OFFSET = int(os.getenv("ID_OFFSET", 0))

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return ' '.join(text.split())  # normalize whitespace

def download_pdf(url, tmp_dir):
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        filename = os.path.join(tmp_dir, url.split('/')[-1])
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        raise

def lambda_handler(event, context):
    logger.info(f"Start PDF parse function")
    sqs_records = event['Records']
    logger.info(f"Received records: {len(sqs_records)}")
    logger.info(f"Records: {sqs_records}")

    docs = []
    tmp_pdf_dir = "/tmp/pdfs"
    os.makedirs(tmp_pdf_dir, exist_ok=True)

    for record in sqs_records:
        try:
            message_body = json.loads(record['body'])
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in SQS message: {record['body']}")
            raise

        url = message_body.get("url")
        title = message_body.get("title", "Untitled")
        job_id = message_body.get("job_id")
        item_id = message_body.get("item_id")
        doc_type = message_body.get("type", "pdf")
        csv_input_key = message_body.get("csv_input_key", "")

        if not url:
            logger.error(f"No URL found in message: {message_body}")
            raise Exception(f"No URL found in message: {message_body}")

        if doc_type.lower() != "pdf":
            logger.error(f"Skipping non-PDF type: {doc_type}")
            raise Exception(f"Skipping non-PDF type: {doc_type}")

        pdf_path = download_pdf(url, tmp_pdf_dir)
        if not pdf_path:
            logger.error(f"Failed to download: {url}")
            raise Exception(f"Failed to download: {url}")


        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise

            

        doc = {
            "uuid": str(uuid.uuid4()),
            "text": unidecode(text),
            "source": len(docs) + ID_OFFSET,
            "metadata": {
                "title": unidecode(title),
                "url": url,
                "type": "pdf"
            }
        }
        docs.append(doc)

    if not docs:
        return {"statusCode": 200, "body": json.dumps({"message": "No documents processed"})}

    # Save to /tmp as JSONL
    jsonl_tmp_path = "/tmp/output.jsonl"
    with open(jsonl_tmp_path, "w", encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            

    # Optionally upload to S3 (change bucket/key as needed)
    now = datetime.now()
    time_str = now.strftime("%H_%M_%S") + f".{now.microsecond}"
    output_key = f"{csv_input_key}/{now.date()}/{time_str}_pdf.jsonl"
    s3.upload_file(jsonl_tmp_path, output_bucket, output_key)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "PDF processing completed",
            "output_s3_path": f"s3://{output_bucket}/{output_key}"
        })
    }
