import os
import tempfile
import shutil
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import boto3

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

class CurrencyInput(BaseModel):
    image_base64: Optional[str] = None
    mime_type: Optional[str] = "image/jpeg"
    text_prompt: Optional[str] = None
    system_instruction: str
    aws_region: Optional[str] = DEFAULT_AWS_REGION
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

@app.post("/analyze-currency")
def analyze_currency(payload: CurrencyInput):
    """
    Fallback endpoint for currency analysis via AWS Bedrock (Nova Pro).
    Receives base64 image and text, invokes multimodal LLM, returns strict JSON.
    """
    if payload.aws_bearer_token_bedrock:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = payload.aws_bearer_token_bedrock
        
    client = boto3.client("bedrock-runtime", region_name=payload.aws_region)
    
    content_blocks = []
    if payload.image_base64:
        content_blocks.append({
            "image": {
                "format": payload.mime_type.split("/")[-1] if "/" in payload.mime_type else "jpeg",
                "source": {
                    "bytes": bytes.fromhex(payload.image_base64) if payload.image_base64.startswith("hex:") else __import__("base64").b64decode(payload.image_base64)
                }
            }
        })
    if payload.text_prompt:
        content_blocks.append({"text": payload.text_prompt})
        
    if not content_blocks:
        raise HTTPException(status_code=400, detail="Must provide image or text")
        
    try:
        response = client.converse(
            modelId="us.amazon.nova-pro-v1:0",
            system=[{"text": payload.system_instruction + "\n\nYou MUST return raw valid JSON. Do not use markdown blocks."}],
            messages=[{"role": "user", "content": content_blocks}]
        )
        
        raw_text = response["output"]["message"]["content"][0]["text"]
        # Strip potential markdown blocks
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if len(lines) > 1:
                raw_text = "\n".join(lines[1:])
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
        import json
        return json.loads(raw_text.strip())
        
    except Exception as e:
        print(f"AWS Bedrock Fallback Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    source: Optional[str] = Form(None),
    suspect_phone: Optional[str] = Form(None),
    incident_date: Optional[str] = Form(None),
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
    import re
    report_id = f"R-{uuid.uuid4().hex[:6].upper()}"
    victim_id = f"V-{uuid.uuid4().hex[:6].upper()}"
    
    # Create the Report Node
    db_nodes.append({
        "id": report_id,
        "label": f"Report: {report_id}" + (f" ({source})" if source else ""),
        "type": "scam_report",
        "riskScore": 0.50
    })
    
    # Create the Victim Node (the phone number from the form)
    db_nodes.append({
        "id": victim_id,
        "label": f"Victim ({phone})",
        "type": "victim",
        "riskScore": 0.10
    })
    
    # Link Victim to Report
    db_edges.append({
        "source": victim_id,
        "target": report_id,
        "relationship": "reported_by",
        "strength": 1.0
    })
    
    # 3. Dynamic Extraction & Explicit Suspects
    phone_matches = re.findall(r'\b\d{10}\b', description)
    upi_matches = re.findall(r'\b[a-zA-Z0-9.\-_]+@[a-zA-Z]+\b', description)
    
    has_suspects = False

    # Use explicitly provided suspect phone if available
    if suspect_phone and suspect_phone != phone:
        suspect_id = f"S-{uuid.uuid4().hex[:6].upper()}"
        db_nodes.append({"id": suspect_id, "label": f"Scammer Ph: {suspect_phone}", "type": "phone_number", "riskScore": 0.98})
        db_edges.append({"source": report_id, "target": suspect_id, "relationship": "mentions_suspect", "strength": 0.95})
        has_suspects = True
    
    # Use regex matches
    if phone_matches:
        for p in set(phone_matches):
            if p != phone and p != suspect_phone: # Don't duplicate explicit suspect
                suspect_id = f"S-{uuid.uuid4().hex[:6].upper()}"
                db_nodes.append({"id": suspect_id, "label": f"Scammer Ph: {p}", "type": "phone_number", "riskScore": 0.95})
                db_edges.append({"source": report_id, "target": suspect_id, "relationship": "mentions_suspect", "strength": 0.9})
                has_suspects = True
                
    if upi_matches:
        for u in set(upi_matches):
            upi_id = f"U-{uuid.uuid4().hex[:6].upper()}"
            db_nodes.append({"id": upi_id, "label": f"UPI: {u}", "type": "bank_account", "riskScore": 0.99})
            db_edges.append({"source": report_id, "target": upi_id, "relationship": "mentions_account", "strength": 0.95})
            has_suspects = True
            
    # If no suspect info found, create a generic one
    if not has_suspects:
        suspect_id = f"S-{uuid.uuid4().hex[:6].upper()}"
        db_nodes.append({"id": suspect_id, "label": "Unknown Scammer", "type": "suspect", "riskScore": 0.80})
        db_edges.append({"source": report_id, "target": suspect_id, "relationship": "investigating", "strength": 0.5})
    
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

@app.get("/dashboard")
def get_dashboard():
    total_cases = len(db_heatmap)
    active_threats = len([n for n in db_nodes if n.get("type") == "suspect"])
    officers_online = len(db_officers)
    resolution_rate = 85 if total_cases > 0 else 0
    
    return {
        "todayComplaints": {"value": total_cases, "trend": "up", "trendValue": 1},
        "counterfeitCases": {"value": 156, "trend": "down", "trendValue": 3},
        "activeFraudRings": {"value": active_threats, "trend": "up", "trendValue": 1},
        "highRiskAreas": {"value": 12, "trend": "neutral"},
        "aiAlerts": {"value": 23, "trend": "up", "trendValue": 7},
        "officersOnline": officers_online,
        "resolutionRate": resolution_rate
    }

@app.get("/cases")
def get_cases(page: int = 1, pageSize: int = 10):
    return {
        "data": [],
        "total": 0,
        "page": page,
        "pageSize": pageSize,
        "totalPages": 0
    }

@app.get("/map")
def get_map_markers():
    return []

@app.get("/statistics")
def get_statistics():
    return {
        "dailyComplaints": [{"label": "Mon", "value": 23}, {"label": "Tue", "value": 35}],
        "weeklyComplaints": [],
        "scamTypes": [{"name": "Digital Arrest", "value": 35, "color": "#EF4444"}],
        "stateWiseCases": [],
        "counterfeitAccuracy": [],
        "totalReportsAnalyzed": 15847,
        "totalFakeCurrencyDetected": 3291,
        "totalActiveOfficers": len(db_officers)
    }

@app.get("/alerts")
def get_alerts():
    return []
