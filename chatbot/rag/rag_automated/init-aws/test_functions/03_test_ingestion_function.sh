#!/bin/bash
set -e

# Upload CSV/JSON file to S3
if [ -f "csv/content-ingestion.csv" ]; then
  echo "📤 Uploading content-ingestion.csv to S3..."
  awslocal s3 cp csv/content-ingestion.csv s3://sam-app-input/content-ingestion.csv
else
  echo "⚠️ File csv/content-ingestion.csv not found, skipping upload."
fi

echo "✅ Resources created."

# Invoke Lambda with test event
echo "🚀 Invoking CsvParserFunction with test event..."
sam local invoke CsvParserFunction --event ./events/s3_put_csv_content_ingestion.json