### Compile SAM project

```bash
sam build --use-container
```

### Create s3 bucket and SQS

```bash
awslocal s3 mb s3://input-bucket-000000000000-us-east-2
awslocal s3 mb s3://parsed-bucket-000000000000-us-east-2
awslocal s3 mb s3://processed-bucket-000000000000-us-east-2

awslocal sqs create-queue --queue-name sam-app-ingest
```

### Initial CSV with name, type and url

```bash
awslocal s3 cp init-aws/test_functions/csv/initial.csv s3://sam-app-input/content-ingestion.csv
sam local invoke CsvParserFunction --event init-aws/test_functions/events/s3_put_csv_initial.json
```

### HTML process

```bash
sam local invoke ParseHtmlFunction --event init-aws/test_functions/events/sqs_new_html.json
```

### Multimedia process

```bash
sam local invoke ParseMultimediaFunction --event init-aws/test_functions/events/sqs_new_multimedia_mp3.json

sam local invoke ParseMultimediaFunction --event init-aws/test_functions/events/sqs_new_multimedia_youtube.json

```

### PDF process

```bash
sam local invoke ParsePdfFunction --event init-aws/test_functions/events/sqs_new_pdf.json
```

### Agregator: Join all /success/_/_.jsonl and create /final/output.json

```bash
sam local invoke AggregatorFunction
```

### Ingestion:

```bash
sam local invoke IngestionFunction --event init-aws/test_functions/events/s3_put_json_ingestion.json
```

### Put parameters in SSM

```
awslocal ssm put-parameter --name "/sam-app/dev/db/host" --value "host.docker.internal" --type String

awslocal ssm put-parameter --name "/sam-app/dev/db/port" --value "5432" --type String

awslocal ssm put-parameter --name "/sam-app/dev/db/name" --value "askellisvector" --type String

awslocal ssm put-parameter --name "/sam-app/dev/db/user" --value "postgres" --type String

awslocal ssm put-parameter --name "/sam-app/dev/db/password" --value "password" --type String

awslocal ssm put-parameter --name "/sam-app/dev/openai/api-key" --value "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" --type String
```
