from datetime import datetime, timezone
import io
import uuid
import boto3, json, os

from langchain_openai import OpenAIEmbeddings
from chatbot.rag.pgvector_dual import DualPGVector
from chatbot.utils.rag_utils import load_vectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

s3 = boto3.client("s3")
# s3 = boto3.client('s3', endpoint_url='http://host.docker.internal:4566')

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4o",
    chunk_size=500,
    chunk_overlap=100,
)

OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET")


def lambda_handler(event, context):
    print("S3 endpoint URL:", s3.meta.endpoint_url)
    print("Result will be saved in: ", OUTPUT_BUCKET)

    successfull_added = []

    if "Records" in event:  # S3 event
        for record in event["Records"]:
            bucket = record["s3"]["bucket"]["name"]
            key = record["s3"]["object"]["key"]

            print("Bucket name:", bucket)
            print("Key object:", key)

            obj = s3.get_object(Bucket=bucket, Key=key)
            content = obj["Body"].read().decode("utf-8")
            json_data = json.loads(content)

            # Ensure json_data is always a list
            if isinstance(json_data, dict):
                json_data = [json_data]

            print(f"Loaded {len(json_data)} records from {bucket}/{key}")

            """
            AWS Lambda handler to process and upload JSON data to the vector store.
            """

            batch_size = int(event.get("batch_size", 300))

            # Load vectorizer (path inside container)
            vectorizer_path = "/var/task/chatbot/rag/assets/tfidf_vectorizer.pkl"
            sparse_encoder = load_vectorizer(vectorizer_path)

            # Load embeddings
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=os.getenv("OPENAI_API_KEY")
            )

            # Build DB connection from environment variables
            host = os.getenv("DB_HOST")
            port = os.getenv("DB_PORT")
            dbname = os.getenv("DB_NAME")
            user = os.getenv("DB_USER")
            password = os.getenv("DB_PASSWORD")
            connection_string = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"

            # Create vector store
            vector_store = DualPGVector.from_existing_index(
                embeddings=embeddings,
                sparse_encoder=sparse_encoder,
                collection_name="dual_lanchain_db",
                connection=connection_string,
            )

            # Split each record and create new items
            for record_obj in json_data:
                text_chunks = splitter.split_text(record_obj.get("text", ""))
                resource_id = record_obj.get("uuid")

                for idx, chunk in enumerate(text_chunks):
                    new_item = {
                        "uuid": str(uuid.uuid4()),
                        "text": chunk,
                        "metadata": {
                            **record_obj.get("metadata", {}),  # copy existing metadata
                            "id": f"{resource_id}_{idx}",       # update id to resource_id + chunk index
                            "created": datetime.now(timezone.utc)
                        },
                    }
                    successfull_added.append(new_item)

            # Add in batches to vector store
            total_items = len(successfull_added)
            for i in range(0, total_items, batch_size):
                batch = successfull_added[i:i + batch_size]
                texts = [item.get("text", "") for item in batch]
                metadatas = [item.get("metadata", {}) for item in batch]
                ids = [item.get("uuid") for item in batch]
                idsSaved = vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                print(f"Batch {i // batch_size + 1}: {len(idsSaved)} records added.")

            # Convert JSON to string
            json_string = json.dumps(successfull_added, ensure_ascii=False, indent=2)

            # Generate new S3 key (append _processed with timestamp)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_key = f"ingested/{key.replace('.json', '')}_{timestamp}.json"

            # Upload processed JSON to S3
            s3.put_object(
                Bucket=OUTPUT_BUCKET,
                Key=processed_key,
                Body=io.BytesIO(json_string.encode("utf-8")),
                ContentType="application/json"
            )

            return {
                "status": "success",
                "batches": (total_items + batch_size - 1) // batch_size,
                "ingested_json_file": processed_key
            }

    else:
        print("Triggered via API Gateway event:", event)

    return {"status": "success"}
