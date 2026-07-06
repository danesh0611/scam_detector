import os
import time
import requests
import json
import boto3
from typing import Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load local environment variables on startup
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env.local"))

def transcribe_audio_azure(
    audio_path: str,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_version: str = "2024-02-01",
    deployment_name: Optional[str] = None
) -> str:
    """
    Transcribes an audio file using Azure OpenAI's Whisper model deployment.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Fall back to environment variables if parameters not supplied
    api_key = api_key or os.getenv("AZURE_OPENAI_KEY")
    endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT", "whisper")

    if not api_key or not endpoint:
        raise ValueError("Azure OpenAI API key and Endpoint must be configured.")

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )

    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=deployment_name,
            file=audio_file
        )
    return result.text


def transcribe_audio_aws(
    audio_path: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region: Optional[str] = None,
    s3_bucket_name: Optional[str] = None
) -> str:
    """
    Transcribes an audio file using Amazon Transcribe.
    Requires uploading the file to S3 first.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Fall back to environment variables
    aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = aws_region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    if not s3_bucket_name:
        raise ValueError("An S3 Bucket Name is required for AWS Transcribe.")

    filename = os.path.basename(audio_path)
    s3_key = f"scamshield-temp-audio/{filename}"

    # 1. Initialize AWS Clients
    s3_client = boto3.client(
        "s3",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    transcribe_client = boto3.client(
        "transcribe",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # 2. Upload file to S3
    print(f"Uploading {audio_path} to S3 bucket {s3_bucket_name}...")
    s3_client.upload_file(audio_path, s3_bucket_name, s3_key)

    s3_uri = f"s3://{s3_bucket_name}/{s3_key}"
    job_name = f"scamshield_stt_{int(time.time())}"
    media_format = filename.split(".")[-1].lower()
    if media_format == "m4a":
        media_format = "mp4" # AWS Transcribe uses 'mp4' format option for m4a files

    # 3. Start Transcription Job
    print(f"Starting AWS Transcribe job: {job_name}")
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": s3_uri},
        MediaFormat=media_format,
        LanguageCode="en-US" # Defaulting to English, can parse multi-lingual audio
    )

    # 4. Poll Job Completion
    try:
        while True:
            status = transcribe_client.get_search_job_status = transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]
            if job_status in ["COMPLETED", "FAILED"]:
                break
            print("Waiting for AWS Transcribe job to complete...")
            time.sleep(3)

        if job_status == "FAILED":
            reason = status["TranscriptionJob"].get("FailureReason", "Unknown error")
            raise RuntimeError(f"AWS Transcribe job failed: {reason}")

        # 5. Fetch Transcript from Result URL
        transcript_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        response = requests.get(transcript_url)
        response.raise_for_status()
        transcript_data = response.json()
        
        transcript_text = transcript_data["results"]["transcripts"][0]["transcript"]
        return transcript_text

    finally:
        # Clean up: Delete S3 object
        try:
            print("Cleaning up S3 temporary audio object...")
            s3_client.delete_object(Bucket=s3_bucket_name, Key=s3_key)
        except Exception as e:
            print(f"Failed to clean up S3 object: {e}")


def transcribe_audio_local_free(audio_path: str) -> str:
    """
    Transcribes an audio file using SpeechRecognition Google Web Speech API.
    (Completely free, online, requires no keys).
    Supports WAV format natively.
    """
    import speech_recognition as sr

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    filename = os.path.basename(audio_path)
    file_format = filename.split(".")[-1].lower()

    if file_format != "wav":
        raise ValueError(
            "The free local transcriber only supports .wav files natively. "
            "Please upload a WAV file or use Azure/AWS transcribers."
        )

    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = r.record(source)
    
    # Recognize speech using Google Speech Recognition
    text = r.recognize_google(audio_data)
    return text


def transcribe_audio_local_whisper(audio_path: str) -> str:
    """
    Transcribes an audio file locally using open-source openai-whisper library.
    Requires: pip install openai-whisper
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "The 'openai-whisper' package is not installed. "
            "Please run: pip install openai-whisper"
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load medium model
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path)
    return result["text"]


# --- Legacy compatibility interface ---
def transcribe_placeholder(audio_path: str) -> str:
    """Fallback using the free local transcriber (requires WAV format)."""
    return transcribe_audio_local_free(audio_path)
