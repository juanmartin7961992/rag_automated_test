import json
import boto3
import os
import tempfile
import requests
from urllib.parse import urlparse
import yt_dlp
from openai import OpenAI
from openai import OpenAI
from openai import AsyncOpenAI
import soundfile as sf  # ✅ replaces pydub
import asyncio
import aiofiles
from urllib.parse import urlparse, parse_qs

# Initialize clients
# s3 = boto3.client('s3')
s3 = boto3.client('s3', endpoint_url='http://host.docker.internal:4566')
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
print("OPEN API KEY:", os.environ['OPENAI_API_KEY'])

def clean_youtube_url(url: str) -> str:
    """
    Extracts a clean YouTube video URL from messy links
    (removes playlist & other query parameters).
    """
    parsed = urlparse(url)

    # youtube.com/watch?v=...
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return f"https://www.youtube.com/watch?v={qs['v'][0]}"

    # youtu.be short links
    if "youtu.be" in parsed.netloc:
        return f"https://www.youtube.com/watch?v={parsed.path.strip('/')}"

    return url  # return unchanged if not YouTube

def download_media(url):
    url = clean_youtube_url(url)  # ✅ sanitize first

    # Direct audio file case
    if url.endswith(('.mp3', '.m4a', '.wav')):
        temp_file = tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(url)[1], delete=False
        )
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            raise Exception(f"Failed to download direct audio: {str(e)}")

    # YouTube/audio-only case
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True,
        'noplaylist': True,
        'postprocessors': [],  # no ffmpeg
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # safer than info["_filename"]
            filename = ydl.prepare_filename(info)

            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f"yt-dlp reported filename {filename} but it was not created"
                )

            return filename
    except Exception as e:
        raise Exception(f"Failed to download audio: {str(e)}")


# Async client
async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

async def transcribe_audio_chunk(file_path: str) -> str:
    # Pass the file path directly
    response = await async_client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=open(file_path, "rb"),  # file object
        response_format="text"
    )
    return response  # ✅ already a string

async def transcribe_audio(audio_file_path: str, chunk_length_sec: int = 60, max_concurrency: int = 5) -> str:
    """
    Transcribe audio file by splitting into chunks and sending them concurrently.
    Uses AsyncOpenAI for true async concurrency.
    """
    file_size = os.path.getsize(audio_file_path)
    if file_size <= 25 * 1024 * 1024:
        with open(audio_file_path, "rb") as f:  # ✅ use file object, not raw bytes
            resp = await async_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return resp

    print(f"Audio is {file_size/1024/1024:.2f} MB, splitting into chunks...")

    chunk_files = []
    with sf.SoundFile(audio_file_path) as f:
        samplerate = f.samplerate
        samples_per_chunk = samplerate * chunk_length_sec

        while True:
            chunk = f.read(samples_per_chunk)
            if len(chunk) == 0:
                break

            tmp_chunk = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            # Explicitly specify WAV format with proper parameters
            sf.write(tmp_chunk.name, chunk, samplerate, format='WAV', subtype='PCM_16')
            chunk_files.append(tmp_chunk.name)

    # Limit concurrency with semaphore
    sem = asyncio.Semaphore(max_concurrency)

    async def sem_transcribe(path):
        async with sem:
            return await transcribe_audio_chunk(path)

    # Launch tasks concurrently
    tasks = [asyncio.create_task(sem_transcribe(path)) for path in chunk_files]
    results = await asyncio.gather(*tasks)

    # cleanup temp files
    for path in chunk_files:
        os.remove(path)

    # Preserve order (tasks created in order → results aligned)
    return "\n".join(results)

def upload_to_s3(content, bucket_name, key):
    """Upload content to S3 bucket"""
    try:
        print(f"Pinchi Uploading to {bucket_name}/{key}")
        s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content,
            ContentType='application/json'
        )
        return f"s3://{bucket_name}/{key}"
    except Exception as e:
        raise Exception(f"S3 upload failed: {str(e)}")


def lambda_handler(event, context):
    try:
        for record in event.get('Records', []):
            message_body = json.loads(record['body'])
            
            title = message_body.get('title', 'untitled')
            media_type = message_body.get('type', '').lower()
            url = message_body.get('url', '')
            
            if not url:
                raise ValueError("URL is required in the message")
            
            print(f"Processing {media_type}: {title} from {url}")
            
            audio_file_path = download_media(url)
            # transcription = transcribe_audio(audio_file_path)
            transcription = asyncio.run(transcribe_audio(audio_file_path))
            
            result = {
                "title": title,
                "type": media_type,
                "source_url": url,
                "transcription": transcription,
                "timestamp": context.get_remaining_time_in_millis()
            }
            
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            s3_key = f"transcriptions/{safe_title.replace(' ', '_')}_{context.aws_request_id}.json"
            
            bucket_name = os.environ['OUTPUT_BUCKET']
            s3_url = upload_to_s3(json.dumps(result), bucket_name, s3_key)
            
            print(f"Successfully processed and uploaded to: {s3_url}")
            
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Transcription completed successfully',
                    's3_location': s3_url
                })
            }
    
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Failed to process transcription'
            })
        }
