# Docker image for RAG automate AWS lambda function

## To build the docker image and push it:

Docker login:

```
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 988360969626.dkr.ecr.us-east-2.amazonaws.com
```

To build and push, in the root of this project run this command:

```
docker buildx build   -t 988360969626.dkr.ecr.us-east-2.amazonaws.com/automated-rag-dev:latest   --platform linux/amd64   -f chatbot/rag/aws_lambda_function/dockerfile-pgvector_dual --provenance=false  --push   .
```

## SAM project

In the SAM template, should be are something like this:

```
IngestionFunction:
    Type: AWS::Serverless::Function
    Properties:
      Environment:
        Variables:
          DB_HOST: host.docker.internal
          DB_PORT: 5432
          DB_NAME: askellisvector
          DB_USER: postgres
          DB_PASSWORD: password
          OPENAI_API_KEY: secret
      PackageType: Image
      ImageUri: dlymi-pgvector:latest
      Architectures:
        - x86_64
      Timeout: 180
      # Handler: chatbot.rag.app.lambda_handler
      MemorySize: 512
      Events:
        NewJsonUpload:
          Type: S3
          Properties:
            Bucket: !Ref InputBucket
            Events: s3:ObjectCreated:*
            Filter:
              S3Key:
                Rules:
                  - Name: suffix
                    Value: final/output.json
        IngestApi:
          Type: Api
          Properties:
            RestApiId: !Ref MyApi
            Path: /ingest
            Method: post
```

- Build project

```
sam build --use-container  --no-cached
```

- Run Localstack:

```
localstack start
```

- Run test IngestionFunction locally:

```
sam local invoke IngestionFunction --event .\events\your-bucket-name\event_ingestion.json
```
