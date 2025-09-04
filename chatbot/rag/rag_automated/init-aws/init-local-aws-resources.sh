#!/bin/bash
set -e

echo "🚀 Creating test resources in LocalStack..."

# Create S3 bucket (same as InputBucket)
awslocal s3 mb s3://sam-app-input || true


# Create SQS queue (same as IngestQueue)
awslocal sqs create-queue --queue-name sam-app-ingest || true

echo "✅ Resources created."
