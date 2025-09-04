import json
import boto3
import os
import logging
from datetime import datetime
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")


def lambda_handler(event, context):
    source_bucket = os.environ.get("SOURCE_BUCKET")
    output_bucket = os.environ.get("OUTPUT_BUCKET")


    if not source_bucket or not output_bucket:
        logger.error("Missing required environment variables.")
        return _http_response(500, {"error": "Missing environment variables"})
    
    try:
        body = json.loads(event["body"]) if event["body"] else {}
        folder_prefix = body.get("folder_prefix", "")
    except Exception as e:
        logger.error(f"Failed to parse body: {e}")
        return _http_response(500, {"error": "Failed to parse body"})
    # folder_prefix = event.get("folder_prefix", "")  # Default to empty string (all files)

    all_data = []

    try:
        # List objects in source bucket with optional prefix
        response = s3.list_objects_v2(Bucket=source_bucket, Prefix=folder_prefix)
        objects = response.get("Contents", [])

        if not objects:
            msg = f"No files found in {source_bucket}/{folder_prefix}"
            logger.warning(msg)
            return _http_response(200, {"status": "ok", "message": msg})

        # Combine files
        for obj in objects:
            key = obj["Key"]
            logger.info(f"Processing file: {key}")
            file_obj = s3.get_object(Bucket=source_bucket, Key=key)
            body = file_obj["Body"].read()
            try:
                data = json.loads(body)
                all_data.append(data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in file {key}, skipping.")
                continue

        # Generate timestamped filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_key = f"combined_{timestamp}.json"

        # Save combined output
        s3.put_object(
            Bucket=output_bucket,
            Key=output_key,
            Body=json.dumps(all_data, indent=2),
            ContentType="application/json"
        )

        # Generate pre-signed URL (valid for 1 hour)
        presigned_url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": output_bucket,
                "Key": output_key
            },
            ExpiresIn=3600  # 1 hour
        )

        logger.info(f"Combined file saved to {output_bucket}/{output_key}")
        return _http_response(200, {
            "status": "ok",
            "count": len(all_data),
            "output_key": output_key,
            "download_url": presigned_url
        })

    except ClientError as e:
        logger.error(f"AWS ClientError: {e}")
        return _http_response(500, {"error": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return _http_response(500, {"error": str(e)})

def _http_response(status_code: int, body: dict):
    """Helper to format API Gateway HTTP responses"""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body)
    }
