from datetime import datetime
import json
import uuid
import boto3
import os
import tempfile
from moviepy import VideoFileClip
import requests
from urllib.parse import urlparse, parse_qs
import yt_dlp
from openai import AsyncOpenAI
import soundfile as sf
import asyncio
import aiofiles
from contextlib import asynccontextmanager, contextmanager
import logging
from typing import Optional, List, Dict, Any
import requests
from pydub import AudioSegment
from io import BytesIO
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
s3 = boto3.client('s3')
sqs = boto3.client('sqs')
# s3 = boto3.client('s3', endpoint_url='http://host.docker.internal:4566')

openai_api_key = os.environ['OPENAI_API_KEY']
async_client = AsyncOpenAI(api_key=openai_api_key)
OUTPUT_BUCKET = os.environ['OUTPUT_BUCKET']

# Constants
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
CHUNK_LENGTH_SEC = 60
MAX_CONCURRENCY = 5
SUPPORTED_AUDIO_EXTENSIONS = ('.mp3', '.m4a', '.wav', '.flac', '.ogg')
SUPPORTED_VIDEO_EXTENSIONS = ( ".mp4", ".mov", "wm.v", ".webm", ".avi")

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


@contextmanager
def download_direct_audio(url: str):
    """Context manager for downloading direct audio files."""
    temp_file = tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(url)[1], delete=False
    )
    # try:
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
    temp_file.close()
    yield temp_file.name
    # finally:
    #     if os.path.exists(temp_file.name):
    #         os.remove(temp_file.name)


def download_media(url: str) -> str:
    """Download media from URL (YouTube or direct audio)."""
    url = clean_youtube_url(url)

    # Direct audio file case
    if url.endswith(SUPPORTED_AUDIO_EXTENSIONS):
        with download_direct_audio(url) as file_path:
            return file_path
        
    if url.endswith(SUPPORTED_VIDEO_EXTENSIONS):
        return convert_mp4_url_to_audio_pydub(url)

    # YouTube/audio-only case
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True,
        'noplaylist': True,
        'postprocessors': [],  # no ffmpeg
        'socket_timeout': 30,
        'retries': 3,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f"yt-dlp reported filename {filename} but it was not created"
                )

            return filename
    except Exception as e:
        raise Exception(f"Failed to download audio: {str(e)}")

def convert_mp4_url_to_audio_pydub(url):
    # Download MP4 to temp file
    response = requests.get(url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(response.content)
        temp_video_path = temp_video.name

    # Load video and extract audio
    clip = VideoFileClip(temp_video_path)
    output_filename = f"{uuid.uuid4()}.mp3"
    clip.audio.write_audiofile(output_filename)  # moviepy handles ffmpeg internally
    clip.close()

    return output_filename

async def transcribe_audio_chunk(file_path: str) -> str:
    """Transcribe a single audio chunk."""
    try:
        async with aiofiles.open(file_path, 'rb') as audio_file:
            audio_data = await audio_file.read()
            
        response = await async_client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=("chunk.wav", audio_data),
            response_format="text"
        )
        return response
    except Exception as e:
        logger.error(f"Failed to transcribe chunk {file_path}: {str(e)}")
        return ""  # Return empty string for failed chunks


@asynccontextmanager
async def create_audio_chunks(audio_file_path: str, chunk_length_sec: int = CHUNK_LENGTH_SEC):
    """Context manager to create and clean up audio chunks."""
    chunk_files = []
    try:
        with sf.SoundFile(audio_file_path) as f:
            samplerate = f.samplerate
            samples_per_chunk = samplerate * chunk_length_sec

            while True:
                chunk = f.read(samples_per_chunk)
                if len(chunk) == 0:
                    break

                # Create temporary file for chunk
                tmp_chunk = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_chunk.name, chunk, samplerate, format='WAV', subtype='PCM_16')
                chunk_files.append(tmp_chunk.name)
        
        yield chunk_files
    finally:
        # Cleanup chunk files
        for path in chunk_files:
            if os.path.exists(path):
                os.remove(path)


async def transcribe_audio(audio_file_path: str, chunk_length_sec: int = CHUNK_LENGTH_SEC, 
                          max_concurrency: int = MAX_CONCURRENCY) -> str:
    """
    Transcribe audio file by splitting into chunks and sending them concurrently.
    Uses AsyncOpenAI for true async concurrency.
    """
    try:
        # Check file size
        file_size = os.path.getsize(audio_file_path)
        if file_size <= MAX_FILE_SIZE:
            async with aiofiles.open(audio_file_path, 'rb') as f:
                audio_data = await f.read()
                
            resp = await async_client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", audio_data),
                response_format="text"
            )
            return resp

        logger.info(f"Audio is {file_size/1024/1024:.2f} MB, splitting into chunks...")

        # Create and process chunks
        async with create_audio_chunks(audio_file_path, chunk_length_sec) as chunk_files:
            # Limit concurrency with semaphore
            sem = asyncio.Semaphore(max_concurrency)

            async def sem_transcribe(path):
                async with sem:
                    return await transcribe_audio_chunk(path)

            # Launch tasks concurrently
            tasks = [asyncio.create_task(sem_transcribe(path)) for path in chunk_files]
            results = await asyncio.gather(*tasks)

        # Preserve order and combine results
        return "\n".join(results)
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise


def upload_to_s3(content: str, bucket_name: str, key: str) -> str:
    """Upload content to S3 bucket."""
    try:
        logger.info(f"Uploading to {bucket_name}/{key}")
        s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content,
            ContentType='application/json'
        )
        return f"s3://{bucket_name}/{key}"
    except Exception as e:
        raise Exception(f"S3 upload failed: {str(e)}")


def create_safe_filename(title: str, request_id: str, media_type:str, csv_input_key: str) -> str:
    """Create a safe filename for S3 storage."""
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    now = datetime.now()
    time_str = now.strftime("%H_%M_%S") + f".{now.microsecond}"
    base_name = f"{csv_input_key}/{now.date()}/{time_str}_{media_type}"
    return f"{base_name}.jsonl"


def lambda_handler(event, context):
    """AWS Lambda handler function."""
    results = []
    for record in event.get('Records', []):
        try:
            message_body = json.loads(record['body'])

            title = message_body.get('title', 'untitled')
            media_type = message_body.get('type', '').lower()
            csv_input_key = message_body.get('csv_input_key', '').lower()
            url = message_body.get('url', '')

            if not url:
                raise ValueError("URL is required in the message")

            logger.info(f"Processing {media_type}: {title} from {url}")

            # Download + transcribe
            audio_file_path = download_media(url)
            transcription = asyncio.run(transcribe_audio(audio_file_path))

            result = {
                "uuid": str(uuid.uuid4()),
                "text": transcription,
                "source": 0,
                "metadata": {
                    "title": title,
                    "url": url,
                    "type": media_type,
                },
            }

            s3_key = create_safe_filename(title, context.aws_request_id, media_type, csv_input_key)
            s3_url = upload_to_s3(json.dumps(result), OUTPUT_BUCKET, s3_key)

            logger.info(f"Successfully uploaded to {s3_url}")

            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

            results.append({"status": "ok", "s3": s3_url})

        except Exception as e:
            logger.error(f"Error processing record: {str(e)}", exc_info=True)
            # re-raise so Lambda reports a failure → message stays in queue
            raise

    return {"statusCode": 200, "body": json.dumps(results)}



# ------------------ Local Debug ------------------ #
if __name__ == "__main__":
    url = "https://neuhaus.azureedge.net/library/videos/2020/08/Questioning-Strategies-to-Deepen-Comprehension.mp4"
    # audio_file_path = convert_mp4_url_to_audio_pydub(url)
    audio_file_path = download_media(url)
    print("Pinchi", audio_file_path)
    transcription = asyncio.run(transcribe_audio(audio_file_path))
    print(f"Pinchi transcription: {transcription}")
