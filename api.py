import os
import tempfile
import shutil
import time
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

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ScamShield Pro REST API",
    description="REST API endpoints for Speech-To-Text translation and Scam Detection analysis.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import uuid

# --- In-Memory Hackathon Database ---
db_officers = {
    "POL-12345": {"password": "password123", "name": "Insp. Rajesh Kumar", "rank": "Inspector", "station": "Cyber Crime Cell, HQ"},
    "POL-99999": {"password": "admin", "name": "Chief Anita Sharma", "rank": "Chief", "station": "Central Command"}
}

db_heatmap = [
    {"lat": 13.0827, "lng": 80.2707, "intensity": 0.8},
    {"lat": 13.0604, "lng": 80.2496, "intensity": 0.5}
]

db_nodes = [
    {"id": "N1", "label": "SBI Acc. ***4521", "type": "bank_account", "riskScore": 0.95},
    {"id": "N2", "label": "+91 98765 XXXXX", "type": "phone_number", "riskScore": 0.9},
    {"id": "N3", "label": "Device #A7F3", "type": "ip_address", "riskScore": 0.85}
]

db_edges = [
    {"source": "N1", "target": "N2", "relationship": "registered_phone", "strength": 0.95},
    {"source": "N2", "target": "N3", "relationship": "used_device", "strength": 0.9}
]

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

class SessionInput(BaseModel):
    text: str
    metadata: Optional[dict] = None
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
            "POST /analyze-session": "Analyze active calls/sessions with transcript and caller/video metadata.",
            "POST /analyze-audio": "Upload an audio recording, transcribe it, and analyze it.",
            "GET /health": "Verify API service health."
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ScamShield Pro API"}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(req: LoginRequest):
    officer = db_officers.get(req.username)
    if not officer or officer["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid badge number or password")
        
    return {
        "token": f"token-{uuid.uuid4().hex}",
        "officer": {
            "id": f"OFF-{req.username}",
            "name": officer["name"],
            "badge": req.username,
            "rank": officer["rank"],
            "station": officer["station"]
        }
    }

@app.post("/analyze-session")
def analyze_session(payload: SessionInput):
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
        
        result = detector.analyze(payload.text, payload.metadata)
        
        # Determine telecom actions and victim alert based on scam probability
        scam_prob = result.get("scam_probability", 0)
        telecom_actions = []
        victim_alert = False
        
        if scam_prob >= 75:
            telecom_actions = ["quarantine_call", "sms_warning_sent", "log_suspect_number", "block_caller"]
            victim_alert = True
        elif scam_prob >= 40:
            telecom_actions = ["sms_warning_sent", "log_suspect_number"]
            victim_alert = True
            
        result["telecom_intervention"] = {
            "actions_triggered": telecom_actions,
            "victim_warning_active": victim_alert,
            "system_timestamp": int(time.time())
        }
        
        # MHA Alert generation
        if scam_prob >= 40:
            meta = payload.metadata or {}
            caller_id = meta.get("caller_id", "Unknown")
            voip_flag = meta.get("voip", False)
            stir_flag = meta.get("stir_shaken", "UNKNOWN")
            video_platform = meta.get("platform", "Unknown")
            background_flag = meta.get("synthetic_background", False)
            screen_share_flag = meta.get("screen_sharing", False)
            
            stage_info = result.get("digital_arrest_flow", {}).get("current_stage", 0)
            
            result["mha_alert"] = {
                "incident_id": f"MHA-I4C-{int(time.time())}",
                "reporting_agency": "ScamShield Telecom Intercept Grid",
                "suspect_details": {
                    "caller_id": caller_id,
                    "voip_indicator": voip_flag,
                    "stir_shaken_status": stir_flag,
                    "video_platform": video_platform,
                    "synthetic_background_detected": background_flag,
                    "screen_sharing_active": screen_share_flag
                },
                "evidence": {
                    "active_stage": stage_info,
                    "transcript_snippet": payload.text[-500:],
                    "full_transcript_length": len(payload.text),
                    "triggered_flags": [r["flag"] for r in result.get("reasons", [])]
                },
                "status": "DRAFT_ALERT_GENERATED",
                "dispatch_endpoint": "https://cybercrime.gov.in/api/v1/alerts"
            }
        else:
            result["mha_alert"] = None
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    stt_provider: str = Form("local_whisper"),  # "azure", "aws", "local_whisper", "local_free"
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
    # 1. Save uploaded file temporarily
    file_format = file.filename.split(".")[-1].lower()
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

@app.post("/report")
def add_report(
    phone: str = Form(...),
    description: str = Form(...),
    lat: float = Form(13.0827),
    lng: float = Form(80.2707),
    screenshot: Optional[UploadFile] = File(None),
    currencyImage: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    # 1. Add to Heatmap
    db_heatmap.append({
        "lat": lat,
        "lng": lng,
        "intensity": 1.0
    })
    
    # 2. Add to Graph Network
    report_id = f"R-{uuid.uuid4().hex[:6].upper()}"
    phone_id = f"P-{uuid.uuid4().hex[:6].upper()}"
    
    # Create a Suspect Node for the report
    db_nodes.append({
        "id": report_id,
        "label": f"Report: {report_id}",
        "type": "suspect",
        "riskScore": 0.99
    })
    
    # Create a Phone Node for the scammer
    db_nodes.append({
        "id": phone_id,
        "label": phone,
        "type": "phone_number",
        "riskScore": 0.90
    })
    
    # Link them
    db_edges.append({
        "source": report_id,
        "target": phone_id,
        "relationship": "used_by_suspect",
        "strength": 1.0
    })
    
    return {"success": True, "complaintId": report_id, "message": "Report submitted and graph updated."}

@app.get("/heatmap")
def get_heatmap():
    return db_heatmap

@app.get("/fraud-network")
def get_fraud_network():
    return {
        "nodes": db_nodes,
        "edges": db_edges
    }
