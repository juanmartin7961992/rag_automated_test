#!/bin/bash
set -e

echo "🚀 Testing ingestion pipeline..."

# 1. Create a sample CSV file
# cat > test-content-ingestion.csv <<EOF
# title,url,content_type
# Doc1,https://example.com,html
# Doc2,https://example.org,pdf
# EOF


#1. Create a sample CSV file
cat > content-ingestion.csv <<EOF
title,type,url
the-science-of-reading-understanding-how-children-learn-to-read,pdf,https://neuhaus.org/the-science-of-reading-understanding-how-children-learn-to-read/
EOF

# 2. Upload it to S3 (this should trigger CsvParserFunction)
echo "📤 Uploading CSV to S3..."
awslocal s3 cp content-ingestion.csv s3://sam-app-input/content-ingestion.csv

# 3. Wait briefly for Lambda to process
echo "⏳ Waiting for Lambda to process..."
sleep 5

# 4. Check SQS messages (should contain parsed rows)
echo "📥 Checking messages in SQS..."
awslocal sqs receive-message \
  --queue-url http://localhost:4566/000000000000/sam-app-ingest \
  --max-number-of-messages 10 \
  --wait-time-seconds 2 || true

echo "✅ Test complete!"
