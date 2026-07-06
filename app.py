import os
import streamlit as st
import tempfile
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

# --- Page Setup & Styling ---
st.set_page_config(
    page_title="ScamShield Pro - AI Call & Message Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Glassmorphism & Sleek Dark Mode)
st.markdown("""
<style>
    /* Global Styles */
    body {
        background-color: #0E1117;
        color: #E2E8F0;
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }

    /* Cards */
    .premium-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Glowing Verdict Cards */
    .verdict-scam {
        background: rgba(220, 38, 38, 0.15);
        border: 1px solid rgba(220, 38, 38, 0.4);
        box-shadow: 0 0 20px rgba(220, 38, 38, 0.15);
    }
    
    .verdict-suspicious {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.15);
    }
    
    .verdict-safe {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    
    /* Metrics styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Red flag list */
    .flag-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #F43F5E;
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 12px;
    }

    .flag-title {
        font-weight: 600;
        color: #FDA4AF;
        margin-bottom: 4px;
    }

    .flag-matched {
        font-style: italic;
        color: #E2E8F0;
        background: rgba(0,0,0,0.2);
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=70)
    st.title("ScamShield Config")
    
    # 1. Speech-To-Text API Config
    st.markdown("### 🎙️ Speech-To-Text (STT)")
    stt_provider = st.selectbox(
        "STT Backend",
        options=[
            "Azure OpenAI Whisper",
            "AWS Transcribe",
            "Local Free (Google)",
            "Local Whisper (Offline)"
        ]
    )
    
    if stt_provider == "Azure OpenAI Whisper":
        azure_endpoint = st.text_input("Azure Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
        azure_key = st.text_input("Azure API Key", value=os.getenv("AZURE_OPENAI_KEY", ""), type="password")
        azure_deployment = st.text_input("Whisper Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", "whisper"))
        
    elif stt_provider == "AWS Transcribe":
        aws_s3_bucket = st.text_input("S3 Bucket Name (Required)", value=os.getenv("AWS_S3_BUCKET", ""))
        aws_region_stt = st.text_input("AWS Region (STT)", value=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        aws_access_key_id_stt = st.text_input("AWS Access Key (STT)", value=os.getenv("AWS_ACCESS_KEY_ID", ""))
        aws_secret_access_key_stt = st.text_input("AWS Secret Key (STT)", value=os.getenv("AWS_SECRET_ACCESS_KEY", ""), type="password")
        
    elif stt_provider == "Local Free (Google)":
        st.info("Uses a lightweight Google Web Speech API locally. Completely free, no keys required. Supports any audio format.")
        
    elif stt_provider == "Local Whisper (Offline)":
        st.info("Runs standard Whisper model completely offline on your computer. Requires `openai-whisper` package installed.")
    
    st.markdown("---")
    
    # 2. Scam Detection Engine Config
    st.markdown("### ⚙️ Detection Engine")
    engine = st.selectbox("Select Model Backend", options=["Local ML + Rules Engine", "AWS Bedrock LLM"])
    
    if engine == "AWS Bedrock LLM":
        st.markdown("#### AWS Bedrock Options")
        aws_token = st.text_input("Bedrock Bearer Token", value=os.getenv("AWS_BEARER_TOKEN_BEDROCK", ""), type="password")
        aws_region = st.text_input("AWS Region", value=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        model_id = st.text_input("Bedrock Model ID", value=os.getenv("AWS_BEDROCK_MODEL_ID", "meta.llama3-8b-instruct-v1:0"))
    else:
        st.info("Using local TF-IDF & Logistic Regression with rule-based heuristics.")

# --- Main App Interface ---
st.title("🛡️ ScamShield Pro")
st.markdown("##### AI-Powered Scam Caller & Message Detection Suite")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("📤 Input Media or Text")
    
    file_types = ["mp3", "wav", "m4a"]
    help_text = "Supports .mp3, .wav, or .m4a files."
        
    input_type = st.radio("Choose Input Format", ["Audio File Upload (Call Recording)", "Raw Message Text"])
    
    text_to_analyze = ""
    transcript_retrieved = False
    
    if input_type == "Audio File Upload (Call Recording)":
        uploaded_file = st.file_uploader(f"Upload call recording ({', '.join(file_types)})", type=file_types, help=help_text)
        
        if uploaded_file is not None:
            # Play the audio back
            st.audio(uploaded_file, format=uploaded_file.type)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            if st.button("Transcribe Call & Analyze 🎙️", use_container_width=True):
                with st.spinner(f"Transcribing audio with {stt_provider}..."):
                    try:
                        if stt_provider == "Azure OpenAI Whisper":
                            transcript = transcribe_audio_azure(
                                audio_path=temp_file_path,
                                api_key=azure_key,
                                endpoint=azure_endpoint,
                                deployment_name=azure_deployment
                            )
                        elif stt_provider == "AWS Transcribe":
                            transcript = transcribe_audio_aws(
                                audio_path=temp_file_path,
                                aws_access_key_id=aws_access_key_id_stt,
                                aws_secret_access_key=aws_secret_access_key_stt,
                                aws_region=aws_region_stt,
                                s3_bucket_name=aws_s3_bucket
                            )
                        elif stt_provider == "Local Free (Google)":
                            transcript = transcribe_audio_local_free(audio_path=temp_file_path)
                        elif stt_provider == "Local Whisper (Offline)":
                            transcript = transcribe_audio_local_whisper(audio_path=temp_file_path)
                            
                        text_to_analyze = transcript
                        transcript_retrieved = True
                        st.success("Audio transcribed successfully!")
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        if "DeploymentNotFound" in str(e):
                            st.warning("⚠️ If you are using Azure OpenAI, make sure your deployment name matches exactly. You can switch to 'Local Free (Google)' to transcribe files without any setup.")
                        elif "SubscriptionRequiredException" in str(e):
                            st.warning("⚠️ Your AWS credentials require a subscription for Amazon Transcribe. Switch to 'Local Free (Google)' or 'Azure OpenAI' to bypass this.")
                    finally:
                        # Clean up temp file
                        try:
                            os.remove(temp_file_path)
                        except Exception:
                            pass
    else:
        text_to_analyze = st.text_area("Enter Message Text", placeholder="Paste SMS, WhatsApp message, or transcript snippet here...")
        if st.button("Analyze Text 🔎", use_container_width=True):
            transcript_retrieved = True
            
    st.markdown('</div>', unsafe_allow_html=True)

    if text_to_analyze:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("📝 Message Text / Transcript")
        st.text_area("Analysis Input", value=text_to_analyze, height=180, disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Display Results ---
with col2:
    if transcript_retrieved and text_to_analyze.strip():
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("📊 Security Analysis Report")
        
        with st.spinner("Analyzing text for scam indicators..."):
            if engine == "AWS Bedrock LLM":
                detector = BedrockScamDetector(
                    model_id=model_id,
                    aws_region=aws_region,
                    aws_bearer_token_bedrock=aws_token
                )
            else:
                detector = ScamDetector()
                
            result = detector.analyze(text_to_analyze)
            
        verdict = result.get("verdict", "UNKNOWN")
        trust_score = result.get("trust_score", 0)
        scam_prob = result.get("scam_probability", 0)
        reasons = result.get("reasons", [])
        
        # Format verdict style
        verdict_class = "verdict-safe"
        verdict_icon = "✅"
        
        if "SCAM" in verdict.upper():
            verdict_class = "verdict-scam"
            verdict_icon = "🚨"
        elif "SUSPICIOUS" in verdict.upper():
            verdict_class = "verdict-suspicious"
            verdict_icon = "⚠️"
            
        # Display Verdict Card
        st.markdown(f"""
        <div class="premium-card {verdict_class}">
            <h2 style='margin:0; text-align:center;'>{verdict_icon} {verdict}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Metrics
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center;">
                <div class="metric-title">🛡️ Trust Score</div>
                <div class="metric-value" style="color: {'#10B981' if trust_score >= 65 else '#F59E0B' if trust_score >= 35 else '#EF4444'}">{trust_score}%</div>
                <div style="font-size:0.8rem; color:#64748B;">Higher = More Safe</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center;">
                <div class="metric-title">🔥 Scam Risk</div>
                <div class="metric-value" style="color: {'#EF4444' if scam_prob >= 65 else '#F59E0B' if scam_prob >= 35 else '#10B981'}">{scam_prob}%</div>
                <div style="font-size:0.8rem; color:#64748B;">Higher = More Likely Scam</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Detailed Engine Metrics
        if engine == "Local ML + Rules Engine":
            st.markdown("##### Under the hood (Blended Score):")
            st.progress(int(scam_prob))
            st.write(f"- ML Model Scam Score: `{result.get('ml_probability', 0)}%`")
            st.write(f"- Rule Engine Flag Score: `{result.get('rule_score', 0)}%`")
            
        # Red Flag Explanations
        st.markdown("### 🚩 Detected Red Flags")
        if reasons:
            for r in reasons:
                if r.get("flag") == "error":
                    st.error(f"Error during Bedrock analysis: {r['why']}")
                    continue
                
                st.markdown(f"""
                <div class="flag-card">
                    <div class="flag-title">🚩 {r['flag'].replace('_', ' ').title()}</div>
                    <div style="margin-bottom:6px;">Matched phrasing: <span class="flag-matched">"{r['matched']}"</span></div>
                    <div style="font-size:0.9rem; color:#94A3B8;">{r['why']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No critical scam indicators or red flags triggered.")
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👈 Upload an audio file or write some message text to start the security analysis.")
