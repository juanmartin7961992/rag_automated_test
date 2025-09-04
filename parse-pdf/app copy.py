import os
import json
import boto3
import pandas as pd
import requests
import PyPDF2
from unidecode import unidecode

# s3 = boto3.client("s3")
s3 = boto3.client('s3', endpoint_url='http://host.docker.internal:4566')

# Optional: set default offset via env var
ID_OFFSET = int(os.getenv("ID_OFFSET", 0))

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
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
        print(f"Error downloading {url}: {e}")
    return None

def lambda_handler(event, context):
    # Get S3 bucket and object key from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Download CSV file from S3 to /tmp
    csv_tmp_path = f"/tmp/{os.path.basename(key)}"
    s3.download_file(bucket, key, csv_tmp_path)

    # Read CSV
    df = pd.read_csv(csv_tmp_path)
    df["URL"] = df["URL"].str.strip()
    df = df[df["URL"] != ""]

    # Prepare outputs
    docs = []
    tmp_pdf_dir = "/tmp/pdfs"
    os.makedirs(tmp_pdf_dir, exist_ok=True)

    for _, row in df.iterrows():
        pdf_path = download_pdf(row['URL'], tmp_pdf_dir)
        if not pdf_path:
            print(f"Failed to download: {row['URL']}")
            continue

        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue

        doc = {
            "id": len(docs) + ID_OFFSET,
            "text": unidecode(text),
            "source": len(docs) + ID_OFFSET,
            "metadata": {
                "title": unidecode(row['Name']),
                "url": row['URL']
            }
        }
        docs.append(doc)

    if not docs:
        return {"statusCode": 200, "body": json.dumps({"message": "No documents processed"})}

    # Write JSONL to /tmp
    base_name = os.path.splitext(os.path.basename(key))[0]
    jsonl_tmp_path = f"/tmp/{base_name}.jsonl"
    with open(jsonl_tmp_path, "w", encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Write CSV records to /tmp
    records = [{"id": doc["source"], "url": doc["metadata"]["url"], "title": doc["metadata"]["title"]} for doc in docs]
    csv_tmp_out_path = f"/tmp/{base_name}_records.csv"
    pd.DataFrame(records).drop_duplicates().to_csv(csv_tmp_out_path, index=False)

    # Upload outputs back to S3
    output_prefix = f"processed/{base_name}/"
    s3.upload_file(jsonl_tmp_path, bucket, output_prefix + f"{base_name}.jsonl")
    s3.upload_file(csv_tmp_out_path, bucket, output_prefix + f"{base_name}_records.csv")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Processing completed",
            "jsonl_file": output_prefix + f"{base_name}.jsonl",
            "csv_file": output_prefix + f"{base_name}_records.csv"
        })
    }
