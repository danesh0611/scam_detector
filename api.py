import os
import tempfile
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from pydantic import BaseModel
from dotenv import load_dotenv

from model import ScamDetector
from bedrock_detector import BedrockScamDetector
from transcribe_stub import (
    transcribe_audio_azure,
    transcribe_audio_aws,
    transcribe_audio_local_whisper,
    transcribe_audio_local_free
)

# Load environment variables on startup
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env.local"))

app = FastAPI(
    title="ScamShield Pro REST API",
    description="REST API endpoints for Speech-To-Text translation and Scam Detection analysis.",
    version="1.0"
)

# --- Default Credentials ---
DEFAULT_AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DEFAULT_BEDROCK_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
DEFAULT_AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
DEFAULT_AZURE_KEY = os.getenv("AZURE_OPENAI_KEY", "")

# --- Models ---
class TextInput(BaseModel):
    text: str
    engine: str = "local"  # "local" or "bedrock"
    model_id: Optional[str] = "meta.llama3-8b-instruct-v1:0"
    aws_region: Optional[str] = DEFAULT_AWS_REGION
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_bearer_token_bedrock: Optional[str] = DEFAULT_BEDROCK_TOKEN

@app.get("/")
def read_root():
    return {
        "message": "Welcome to ScamShield Pro REST API!",
        "endpoints": {
            "POST /analyze-text": "Analyze raw message text for scam signals.",
            "POST /analyze-audio": "Upload an audio recording, transcribe it, and analyze it.",
            "GET /health": "Verify API service health."
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ScamShield Pro API"}

@app.post("/analyze-text")
def analyze_text(payload: TextInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

    try:
        if payload.engine == "bedrock":
            detector = BedrockScamDetector(
                model_id=payload.model_id,
                aws_region=payload.aws_region,
                aws_access_key_id=payload.aws_access_key_id,
                aws_secret_access_key=payload.aws_secret_access_key,
                aws_bearer_token_bedrock=payload.aws_bearer_token_bedrock
            )
        else:
            detector = ScamDetector()
        
        result = detector.analyze(payload.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    stt_provider: str = Form("local_free"),  # "azure", "aws", "local_whisper", "local_free"
    engine: str = Form("local"),  # "local", "bedrock"
    # Azure Whisper Credentials
    azure_endpoint: Optional[str] = Form(DEFAULT_AZURE_ENDPOINT),
    azure_key: Optional[str] = Form(DEFAULT_AZURE_KEY),
    azure_deployment: Optional[str] = Form("whisper"),
    # AWS Transcribe Credentials
    aws_s3_bucket: Optional[str] = Form(None),
    aws_region_stt: Optional[str] = Form(DEFAULT_AWS_REGION),
    aws_access_key_id_stt: Optional[str] = Form(None),
    aws_secret_access_key_stt: Optional[str] = Form(None),
    # Bedrock Credentials (if engine="bedrock")
    aws_bearer_token_bedrock: Optional[str] = Form(DEFAULT_BEDROCK_TOKEN),
    aws_region_detector: Optional[str] = Form(DEFAULT_AWS_REGION),
    model_id_detector: Optional[str] = Form("meta.llama3-8b-instruct-v1:0")
):
    # Validate file format
    file_format = file.filename.split(".")[-1].lower()
    if stt_provider == "local_free" and file_format != "wav":
        raise HTTPException(
            status_code=400,
            detail="The local free STT engine only supports WAV files natively. Please upload a WAV file or use Azure OpenAI Whisper."
        )

    # 1. Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    # 2. Transcribe Audio
    try:
        transcript = ""
        if stt_provider == "azure":
            if not azure_key or not azure_endpoint:
                raise HTTPException(status_code=400, detail="Azure Endpoint and API Key are required.")
            transcript = transcribe_audio_azure(
                audio_path=temp_file_path,
                api_key=azure_key,
                endpoint=azure_endpoint,
                deployment_name=azure_deployment
            )
        elif stt_provider == "aws":
            if not aws_s3_bucket:
                raise HTTPException(status_code=400, detail="An S3 Bucket Name is required for AWS Transcribe.")
            transcript = transcribe_audio_aws(
                audio_path=temp_file_path,
                aws_access_key_id=aws_access_key_id_stt,
                aws_secret_access_key=aws_secret_access_key_stt,
                aws_region=aws_region_stt,
                s3_bucket_name=aws_s3_bucket
            )
        elif stt_provider == "local_whisper":
            transcript = transcribe_audio_local_whisper(audio_path=temp_file_path)
        elif stt_provider == "local_free":
            transcript = transcribe_audio_local_free(audio_path=temp_file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported STT provider: {stt_provider}")

        # 3. Analyze Transcript
        if engine == "bedrock":
            detector = BedrockScamDetector(
                model_id=model_id_detector,
                aws_region=aws_region_detector,
                aws_bearer_token_bedrock=aws_bearer_token_bedrock
            )
        else:
            detector = ScamDetector()

        result = detector.analyze(transcript)
        
        # Append transcript to output
        result["transcript"] = transcript
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
