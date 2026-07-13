import os
import streamlit as st
import tempfile
import time
import json
import re
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

# Custom premium styling (Glassmorphism & Sleek Dark Mode with pulsing elements)
st.markdown("""
<style>
    /* Global Styles */
    body {
        background-color: #0E1117;
        color: #E2E8F0;
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0d0e15 0%, #151030 100%);
    }

    /* Cards */
    .premium-card {
        background: rgba(22, 24, 37, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }
    
    /* Glowing Verdict Cards */
    .verdict-scam {
        background: rgba(220, 38, 38, 0.18);
        border: 1px solid rgba(220, 38, 38, 0.5);
        box-shadow: 0 0 25px rgba(220, 38, 38, 0.25);
    }
    
    .verdict-suspicious {
        background: rgba(245, 158, 11, 0.18);
        border: 1px solid rgba(245, 158, 11, 0.5);
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.25);
    }
    
    .verdict-safe {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.5);
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);
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
        background: rgba(0,0,0,0.25);
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* Chat bubble */
    .chat-bubble-caller {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #E2E8F0;
    }
    
    /* Stages tracker */
    .stage-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-top: 15px;
    }

    .stage-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px;
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stage-row.active {
        border-color: #EF4444;
        background: rgba(239, 68, 68, 0.1);
    }
    
    .stage-row.passed {
        border-color: #10B981;
        background: rgba(16, 185, 129, 0.05);
    }

    .stage-title {
        font-weight: 600;
        font-size: 0.95rem;
    }

    .stage-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
    }

    .badge-inactive {
        background: #334155;
        color: #94A3B8;
    }

    .badge-active {
        background: #EF4444;
        color: #FFFFFF;
        animation: pulse 1.5s infinite;
    }

    /* Telecom intercept terminal */
    .terminal-console {
        background: #090a10;
        border: 1px solid #10b981;
        border-radius: 8px;
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        color: #10b981;
        font-size: 0.85rem;
        max-height: 200px;
        overflow-y: auto;
        margin-top: 15px;
        box-shadow: inset 0 0 10px rgba(16, 185, 129, 0.2);
    }

    /* Danger Warning Banner */
    .danger-banner {
        background: linear-gradient(90deg, #b91c1c 0%, #ef4444 50%, #b91c1c 100%);
        border: 2px solid #f87171;
        border-radius: 12px;
        color: #FFFFFF;
        font-weight: bold;
        text-align: center;
        padding: 16px;
        margin-bottom: 20px;
        font-size: 1.1rem;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% {
            opacity: 0.8;
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5);
        }
        70% {
            opacity: 1;
            box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);
        }
        100% {
            opacity: 0.8;
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=70)
    st.title("ScamShield Grid")
    
    # 1. Detection Engine Config
    st.markdown("### ⚙️ Detection Engine")
    engine = st.selectbox("Select Model Backend", options=["Local ML + Rules Engine", "AWS Bedrock LLM"])
    
    if engine == "AWS Bedrock LLM":
        st.markdown("#### AWS Bedrock Options")
        aws_token = st.text_input("Bedrock Bearer Token", value=os.getenv("AWS_BEARER_TOKEN_BEDROCK", ""), type="password")
        aws_region = st.text_input("AWS Region", value=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        model_id = st.text_input("Bedrock Model ID", value=os.getenv("AWS_BEDROCK_MODEL_ID", "meta.llama3-8b-instruct-v1:0"))
    else:
        st.info("Using local TF-IDF & Logistic Regression with rule-based heuristics.")

    st.markdown("---")

    # 2. Speech-To-Text API Config (Used in static analysis)
    st.markdown("### 🎙️ Speech-To-Text (STT)")
    stt_provider = st.selectbox(
        "STT Backend",
        options=[
            "Local Free (Google)",
            "Local Whisper (Offline)",
            "Azure OpenAI Whisper",
            "AWS Transcribe"
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

# --- Main App Interface ---
st.title("🛡️ ScamShield Pro")
st.markdown("##### Enterprise AI Classifier for Scam Ingestion & Real-Time Call Safeguarding")

# Tabbed Layout
tab_static, tab_live = st.tabs(["🔎 Static Analysis (Files/Texts)", "🎙️ Real-time Digital Arrest Simulator"])

# ==================== TAB 1: STATIC ANALYSIS ====================
with tab_static:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("📤 Input Media or Text")
        
        file_types = ["mp3", "wav", "m4a"]
        help_text = "Supports .mp3, .wav, or .m4a files."
            
        input_type = st.radio("Choose Input Format", ["Audio File Upload (Call Recording)", "Raw Message Text"], key="input_format_static")
        
        text_to_analyze = ""
        transcript_retrieved = False
        
        if input_type == "Audio File Upload (Call Recording)":
            uploaded_file = st.file_uploader(f"Upload call recording ({', '.join(file_types)})", type=file_types, help=help_text, key="audio_uploader")
            
            if uploaded_file is not None:
                st.audio(uploaded_file, format=uploaded_file.type)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                
                if st.button("Transcribe Call & Analyze 🎙️", use_container_width=True, key="btn_transcribe_static"):
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
                        finally:
                            try:
                                os.remove(temp_file_path)
                            except Exception:
                                pass
        else:
            text_to_analyze = st.text_area("Enter Message Text", placeholder="Paste SMS, WhatsApp message, or transcript snippet here...", key="text_area_static")
            if st.button("Analyze Text 🔎", use_container_width=True, key="btn_analyze_static"):
                transcript_retrieved = True
                
        st.markdown('</div>', unsafe_allow_html=True)

        if text_to_analyze:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("📝 Message Text / Transcript")
            st.text_area("Analysis Input", value=text_to_analyze, height=180, disabled=True, key="input_viewer_static")
            st.markdown('</div>', unsafe_allow_html=True)

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
            
            verdict_class = "verdict-safe"
            verdict_icon = "✅"
            
            if "SCAM" in verdict.upper():
                verdict_class = "verdict-scam"
                verdict_icon = "🚨"
            elif "SUSPICIOUS" in verdict.upper():
                verdict_class = "verdict-suspicious"
                verdict_icon = "⚠️"
                
            st.markdown(f"""
            <div class="premium-card {verdict_class}">
                <h2 style='margin:0; text-align:center;'>{verdict_icon} {verdict}</h2>
            </div>
            """, unsafe_allow_html=True)
            
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
                
            if engine == "Local ML + Rules Engine":
                st.markdown("##### Under the hood (Blended Score):")
                st.progress(int(scam_prob))
                st.write(f"- ML Model Scam Score: `{result.get('ml_probability', 0)}%`")
                st.write(f"- Rule Engine Flag Score: `{result.get('rule_score', 0)}%`")
                
            st.markdown("### 🚩 Detected Red Flags")
            if reasons:
                for r in reasons:
                    if r.get("flag") == "error":
                        st.error(f"Error during analysis: {r['why']}")
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
                
            # Digital Arrest Stage details if applicable
            da_flow = result.get("digital_arrest_flow", {})
            if da_flow and da_flow.get("current_stage", 0) > 0:
                st.markdown("### 🚦 Digital Arrest Scam Progress")
                st.info(f"Detected Active Scam Stage: **Stage {da_flow['current_stage']}**")
                
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Upload an audio file or write some message text to start the security analysis.")


# ==================== TAB 2: LIVE SIMULATOR ====================
SCENARIOS = {
    "CBI / Customs Drug Parcel Scam (Digital Arrest)": [
        "FedEx Support: Hello, this is FedEx customer support. We are calling to inform you that a package sent under your Aadhar details from Mumbai to Taiwan has been intercepted by customs officers in Mumbai.",
        "FedEx Support: The parcel contains 5 illegal fake passports, 3 bank cards, and 150 grams of MDMA drugs. We have been instructed to transfer your line to the CBI Cyber Cell for further investigation. Do not hang up.",
        "CBI Inspector Kumar: This is Inspector Ajay Kumar from Mumbai CBI Cyber Cell. We have registered a money laundering and drug trafficking case under your name. You are under digital arrest.",
        "CBI Inspector Kumar: You must immediately lock yourself in a quiet room, go to Skype video call, and keep your camera active. Do not talk to family members or cut this call. If you hang up, a police team will arrest you in 2 hours.",
        "CBI Inspector Kumar: To clear your name from the suspect list, you must transfer your savings to the RBI Safety Verification account. This money will be refunded within 30 minutes after verification is complete."
    ],
    "TRAI Mobile SIM Blocking Scam (Digital Arrest)": [
        "TRAI Warning: This is the Telecom Regulatory Authority of India (TRAI). We have received complaints that your mobile number is linked to illegal advertisements and financial scams. All your SIM cards will be blocked within 2 hours.",
        "TRAI Warning: To prevent deactivation, your call is being transferred to the Delhi Police Cyber Cell. Do not disconnect the line.",
        "Delhi Police Cyber Cell: I am Deputy Commissioner Sharma from the Cyber Crime Department. We have an arrest warrant issued against your name for identity fraud and money laundering. You are placed under digital arrest.",
        "Delhi Police Cyber Cell: You must join our Skype video call immediately. Keep your camera on. Do not disconnect. You are not allowed to contact family or friends during this investigation.",
        "Delhi Police Cyber Cell: We need to verify your account balances. Transfer 1,00,000 Rupees verification deposit via UPI to the government portal. Failure to comply will lead to immediate physical arrest."
    ],
    "Legitimate Bank Inquiry (Safe Call)": [
        "SBI Support: Hello, this is SBI customer service calling. We have detected a transaction of 45,000 Rupees on your card at an online electronics store.",
        "SBI Support: We wanted to confirm if you initiated this transaction. If you did not, we will block your card immediately to protect your funds.",
        "SBI Support: We have sent an OTP to your registered mobile number for card blocking. Please check your screen. Do not share it with me, just input it in your secure SBI mobile banking application."
    ]
}

with tab_live:
    # Initialize simulator session state
    if "step_index" not in st.session_state:
        st.session_state.step_index = 0
    if "transcript_lines" not in st.session_state:
        st.session_state.transcript_lines = []
    if "telecom_logs" not in st.session_state:
        st.session_state.telecom_logs = [
            "System: Intercept pipeline initialized.",
            "System: Telecom network monitor standing by..."
        ]
    if "mha_dispatched" not in st.session_state:
        st.session_state.mha_dispatched = False

    st.markdown("### 🎙️ Live Call Ingestion & Intervention Simulator")
    st.write("Simulate an active call session to evaluate the AI classifier's ability to trace call flows, identify number spoofing, process video call metadata, flag scams, and dispatch alerts before financial transactions are finalized.")
    
    col_sim_ctrl, col_sim_dash = st.columns([5, 7], gap="medium")
    
    # Left Column: Call Ingestion Controls
    with col_sim_ctrl:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("📞 Ingestion Parameters")
        
        # Scenario Selector
        selected_scenario = st.selectbox(
            "Select Scenario Template",
            options=["Select a Template...", "CBI / Customs Drug Parcel Scam (Digital Arrest)", "TRAI Mobile SIM Blocking Scam (Digital Arrest)", "Legitimate Bank Inquiry (Safe Call)", "Custom Playground"],
            key="selected_scenario_template"
        )
        
        # Reset state if template changed
        if "prev_scenario" not in st.session_state or st.session_state.prev_scenario != selected_scenario:
            st.session_state.step_index = 0
            st.session_state.transcript_lines = []
            st.session_state.telecom_logs = [
                "System: Pipeline reset.",
                f"System: Switched to template '{selected_scenario}'."
            ]
            st.session_state.mha_dispatched = False
            st.session_state.prev_scenario = selected_scenario
            
        # Metadata simulator
        st.markdown("#### 🔢 Spoofing & Video Metadata Signatures")
        m_s1, m_s2 = st.columns(2)
        with m_s1:
            caller_num = st.text_input("Suspect Caller ID", value="+92 301 4987213" if "Scam" in selected_scenario else "+91 98765 43210")
            carrier_voip = st.checkbox("VoIP Virtual Carrier Detected", value=True if "Scam" in selected_scenario else False)
            stir_shaken = st.selectbox("STIR/SHAKEN Verification Status", ["FAILED (Spoof Signature)", "PASSED (Verified)", "UNVERIFIED"])
        with m_s2:
            video_platform = st.selectbox("Video Call Platform Used", ["Skype", "WhatsApp", "Zoom", "None"], index=0 if "Scam" in selected_scenario else 3)
            synthetic_bg = st.checkbox("Synthetic Background Detected (Mock Office)", value=True if "Scam" in selected_scenario else False)
            screen_sharing = st.checkbox("Screen Sharing Active", value=False)
            
        # Call simulation actions
        st.markdown("#### 🕹️ Call Stepping controls")
        
        if selected_scenario == "Select a Template...":
            st.info("Choose a scenario template above to start simulating.")
        elif selected_scenario == "Custom Playground":
            custom_input = st.text_area("Enter Next Call Line Transcript", placeholder="e.g., Inspector: Lock the door and verify your funds...")
            if st.button("Ingest Line 🎙️", use_container_width=True):
                if custom_input.strip():
                    st.session_state.transcript_lines.append(custom_input.strip())
                    st.session_state.telecom_logs.append(f"Ingested Custom Audio Line: \"{custom_input.strip()[:60]}...\"")
        else:
            lines = SCENARIOS[selected_scenario]
            c_idx = st.session_state.step_index
            
            if c_idx < len(lines):
                st.write(f"**Next Line to Ingest ({c_idx + 1}/{len(lines)}):**")
                st.code(lines[c_idx])
                if st.button("Advance Call Step 📞", use_container_width=True):
                    # Ingest dialogue
                    dialogue_line = lines[c_idx]
                    st.session_state.transcript_lines.append(dialogue_line)
                    st.session_state.step_index += 1
                    
                    # Update telecom log
                    st.session_state.telecom_logs.append(f"Audio Frame Ingested: \"{dialogue_line[:60]}...\"")
                    
                    # Add custom alerts into telecom console based on steps
                    if c_idx == 0:
                        st.session_state.telecom_logs.append(f"Network: Alert! Mapped Incoming Call ID {caller_num} showing carrier=VoIP.")
                        if "FAILED" in stir_shaken:
                            st.session_state.telecom_logs.append("Network: Warning! STIR/SHAKEN verification failed. High spoof signature.")
                    elif c_idx == 2:
                        st.session_state.telecom_logs.append("Classifier: [IMPERSONATION FLAG] Scammer impersonating CBI/Delhi Police.")
                        st.session_state.telecom_logs.append("Network: Injecting in-call security toast to subscriber.")
                    elif c_idx == 3:
                        st.session_state.telecom_logs.append("Classifier: [ISOLATION FLAG] Digital Arrest pattern matched in stream.")
                        st.session_state.telecom_logs.append("Telecom Operator: Subscriber quarantine sequence initialized.")
            else:
                st.success("Call template completed! Review final metrics on the right.")
                
        if st.button("Reset Simulator 🔄", use_container_width=True):
            st.session_state.step_index = 0
            st.session_state.transcript_lines = []
            st.session_state.telecom_logs = [
                "System: Pipeline reset.",
                "System: Standing by for call connection..."
            ]
            st.session_state.mha_dispatched = False
            st.rerun()

        # Transcript stream display
        st.markdown("#### 💬 Active Call Transcript Stream")
        if st.session_state.transcript_lines:
            for line in st.session_state.transcript_lines:
                st.markdown(f'<div class="chat-bubble-caller">{line}</div>', unsafe_allow_html=True)
        else:
            st.caption("No audio has been ingested yet.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Right Column: Real-Time Security Dashboard
    with col_sim_dash:
        # Run classification on current transcript state
        full_transcript = " ".join(st.session_state.transcript_lines)
        metadata_payload = {
            "caller_id": caller_num,
            "voip": carrier_voip,
            "stir_shaken": "FAILED" if "FAILED" in stir_shaken else "PASSED" if "PASSED" in stir_shaken else "UNVERIFIED",
            "platform": video_platform if video_platform != "None" else "",
            "synthetic_background": synthetic_bg,
            "screen_sharing": screen_sharing
        }
        
        # Analyze
        if full_transcript.strip():
            if engine == "AWS Bedrock LLM":
                detector = BedrockScamDetector(
                    model_id=model_id,
                    aws_region=aws_region,
                    aws_bearer_token_bedrock=aws_token
                )
            else:
                detector = ScamDetector()
                
            result = detector.analyze(full_transcript, metadata_payload)
            verdict = result.get("verdict", "UNKNOWN")
            trust_score = result.get("trust_score", 100.0)
            scam_prob = result.get("scam_probability", 0.0)
            reasons = result.get("reasons", [])
            da_flow = result.get("digital_arrest_flow", {})
        else:
            # Empty state
            verdict = "LIKELY SAFE"
            trust_score = 100.0
            scam_prob = 0.0
            reasons = []
            da_flow = {
                "current_stage": 0,
                "stages": {
                    "1": {"name": "Contact & Identity Impersonation", "active": False, "description": "Scammer impersonates TRAI, FedEx, DHL, or Customs officers."},
                    "2": {"name": "Accusation & Secrecy", "active": False, "description": "Scammer accuses victim of illegal parcels or money laundering."},
                    "3": {"name": "Digital Arrest & Isolation", "active": False, "description": "Scammer forces victim onto a video call under 'digital arrest'."},
                    "4": {"name": "Financial Transfer Demand", "active": False, "description": "Scammer demands victim transfer funds for verification."}
                }
            }

        # Danger warning banner overlay for victim
        if scam_prob >= 75:
            st.markdown("""
            <div class="danger-banner">
                🚨 CRITICAL WARNING: ACTIVE DIGITAL ARREST SCAM DETECTED!<br>
                <span style="font-size:0.9rem; font-weight:normal;">
                CBI, Police, or TRAI will NEVER place you under digital arrest via video call. 
                Do NOT share your screen, keep your door locked, or transfer money. HANG UP IMMEDIATELY.
                </span>
            </div>
            """, unsafe_allow_html=True)
        elif scam_prob >= 40:
            st.markdown("""
            <div class="danger-banner" style="background: linear-gradient(90deg, #d97706 0%, #f59e0b 50%, #d97706 100%); border-color: #fbbf24; box-shadow: 0 0 20px rgba(245, 158, 11, 0.4);">
                ⚠️ SUSPICIOUS ACTIVITY MATCHED: DIGITAL ARREST CALL SEQUENCE<br>
                <span style="font-size:0.85rem; font-weight:normal;">
                The conversation contains pattern phrases commonly associated with government agency impersonation. Proceed with caution.
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("📊 Live Security Metrics")
        
        # Display Verdict
        v_class = "verdict-safe"
        v_icon = "✅"
        if scam_prob >= 75:
            v_class = "verdict-scam"
            v_icon = "🚨"
        elif scam_prob >= 40:
            v_class = "verdict-suspicious"
            v_icon = "⚠️"
            
        st.markdown(f"""
        <div class="premium-card {v_class}" style="padding:15px; margin-bottom:15px;">
            <h3 style='margin:0; text-align:center;'>{v_icon} {verdict}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Trust Gauge & Scam Probability
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center; padding:15px;">
                <div class="metric-title">🛡️ Trust Score</div>
                <div class="metric-value" style="color: {'#10B981' if trust_score >= 65 else '#F59E0B' if trust_score >= 35 else '#EF4444'}">{trust_score}%</div>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center; padding:15px;">
                <div class="metric-title">🔥 Scam Probability</div>
                <div class="metric-value" style="color: {'#EF4444' if scam_prob >= 65 else '#F59E0B' if scam_prob >= 35 else '#10B981'}">{scam_prob}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.progress(int(scam_prob))
        st.markdown('</div>', unsafe_allow_html=True)

        # 4-Stage Call Flow Tracker
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🚦 Digital Arrest Sequence Stages")
        
        current_stage = da_flow.get("current_stage", 0)
        stages_data = da_flow.get("stages", {})
        
        stages_list = [
            ("1", "Contact & Identity Impersonation", "Scammer impersonates TRAI, FedEx, DHL, or Customs officers."),
            ("2", "Accusation & Secrecy", "Scammer accuses victim of illegal parcels or money laundering."),
            ("3", "Digital Arrest & Isolation", "Scammer forces victim onto a video call under 'digital arrest' in a locked room."),
            ("4", "Financial Transfer Demand", "Scammer demands victim transfer funds to 'verification' account.")
        ]
        
        st.markdown('<div class="stage-container">', unsafe_allow_html=True)
        for idx_str, name, desc in stages_list:
            idx = int(idx_str)
            active_info = stages_data.get(idx_str, {})
            is_active = active_info.get("active", False) or idx <= current_stage
            
            row_class = "stage-row active" if is_active else "stage-row"
            badge_class = "stage-badge badge-active" if is_active else "stage-badge badge-inactive"
            badge_text = "Detected" if is_active else "Pending"
            
            st.markdown(f"""
            <div class="{row_class}">
                <div>
                    <div class="stage-title">{idx}. {name}</div>
                    <div style="font-size:0.8rem; color:#94A3B8;">{desc}</div>
                </div>
                <span class="{badge_class}">{badge_text}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Telecom grid log
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🛡️ Telecom Provider Intervention Console")
        st.caption("Logs real-time network interactions, automated firewall flags, and victim intercepts:")
        
        last_log = st.session_state.telecom_logs[-1]
        if scam_prob >= 40 and "Injected in-call warning message" not in last_log and "Subscriber quarantine" not in last_log and "dispatched" not in last_log:
            if scam_prob >= 75:
                st.session_state.telecom_logs.append("Network: [INTERCEPT TRIGGERED] Activating voice channel quarantine. Blocking outbound money transfer messages.")
                st.session_state.telecom_logs.append("Database: Flagging suspect number to national registry (Chakshu / Cyber Safe).")
            else:
                st.session_state.telecom_logs.append("Network: [WARNING TRIGGERED] Dispatching warning flash SMS to victim's handset.")
        
        log_content = "\n".join(st.session_state.telecom_logs)
        st.markdown(f'<div class="terminal-console">{log_content}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # MHA alert generation
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🇮🇳 MHA National Cyber Crime Reporting Portal (I4C)")
        st.write("When an active digital arrest sequence is flagged, ScamShield's grid automatically compiles an official cybercrime case dispatch payload for the Indian Ministry of Home Affairs (MHA) Indian Cyber Crime Coordination Centre (I4C).")
        
        if scam_prob >= 40:
            mha_report = {
                "alert_id": f"MHA-I4C-{int(time.time())}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "incident_category": "Impersonation of Government Authority (Digital Arrest Scam)",
                "source_node": "Telecom_Interceptor_South_Zone",
                "suspect": {
                    "calling_number": caller_num,
                    "voip_flag": carrier_voip,
                    "stir_shaken_status": "FAILED" if "FAILED" in stir_shaken else "PASSED" if "PASSED" in stir_shaken else "UNVERIFIED",
                    "visual_metadata": {
                        "platform": video_platform,
                        "forced_video_feed": True,
                        "synthetic_background_detected": synthetic_bg,
                        "screen_sharing_active": screen_sharing
                    }
                },
                "incident_flow": {
                    "maximum_stage_reached": current_stage,
                    "stages_triggered": [idx for idx, name, desc in stages_list if int(idx) <= current_stage]
                },
                "evidence_packet": {
                    "intercepted_text": full_transcript[-400:],
                    "associated_red_flags": [r["flag"] for r in reasons]
                },
                "recipient_endpoint": "https://cybercrime.gov.in/api/v1/alerts/ingest"
            }
            
            st.json(mha_report)
            
            if st.session_state.mha_dispatched:
                st.success("🇮🇳 Incident Alert successfully dispatched to the MHA National Cybercrime Grid!")
            else:
                if st.button("Dispatch Alert to MHA Grid 🚀", use_container_width=True):
                    st.session_state.mha_dispatched = True
                    st.session_state.telecom_logs.append("MHA Grid: Case report transmitted successfully. ID: " + mha_report["alert_id"])
                    st.rerun()
        else:
            st.info("MHA Alert Generation pending. The system will compile incident reports once call sequence risk rises above suspicious levels (>= 40%).")
        st.markdown('</div>', unsafe_allow_html=True)
