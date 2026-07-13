"""
flags.py — Rule-based "red flag" detector for scam / fraud text.

Works on raw text: an SMS, an email body, a chat message, or a
transcript of a phone call (see transcribe_stub.py for turning
audio into text first).

Each rule has a weight. The sum of triggered weights becomes the
"rule score" that gets blended with the ML model's score in model.py.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Flag:
    name: str
    weight: float
    pattern: str
    explanation: str


# ---- Red flag catalogue -----------------------------------------------
# Weight is roughly "how damning is this signal on its own", 0-1 scale.
FLAGS = [
    Flag(
        "urgency_pressure", 0.55,
        r"\b(act now|immediately|urgent(ly)?|right away|within (24|1|one) hours?|"
        r"expire[sd]? (today|soon|very soon)|going to expire|final (notice|warning)|last chance|"
        r"time[- ]sensitive|don'?t delay)\b",
        "Uses urgency/pressure language to stop you from thinking it through."
    ),
    Flag(
        "credential_request", 0.75,
        r"\b(ot[ptb]|o\.?t\.?p\.?|one[- ]time password|one[- ]time pin|cvv|pin number|verification code|"
        r"login (id|details)|update your (password|kyc|account)|"
        r"confirm your (ssn|social security|account number|card number)|"
        r"credit card (details|number)|card (number|details))\b",
        "Asks for a password, OTP, PIN, or account credentials — legitimate "
        "banks/services never ask for this over chat/call."
    ),
    Flag(
        "payment_request", 0.7,
        r"\b(gift card|itunes card|google play card|wire transfer|western union|"
        r"moneygram|crypto(currency)?|bitcoin|usdt|pay(ment)? via (upi|paypal)|"
        r"send money|processing fee|advance fee|clearance fee|customs fee)\b",
        "Asks for payment via gift cards, wire transfer, or crypto — a classic "
        "scam payment method because it's untraceable and irreversible."
    ),
    Flag(
        "too_good_to_be_true", 0.6,
        r"\b(you('ve| have) won|congratulations.*(won|selected)|lottery|"
        r"lucky draw|free (gift|prize|iphone)|claim your (prize|reward)|"
        r"inheritance|million dollars?)\b",
        "Promises money, prizes, or windfalls you didn't sign up for."
    ),
    Flag(
        "impersonation", 0.5,
        r"\b(irs|income tax dept|hmrc|social security administration|"
        r"microsoft support|apple support|amazon (support|security)|"
        r"bank(?:'s)? security (team|department)|government (agency|office)|courier)\b",
        "Claims to be a bank, government body, or well-known company — a "
        "common impersonation tactic."
    ),
    Flag(
        "threat_or_legal", 0.65,
        r"\b(arrest(ed)? warrant|legal action|lawsuit|suspend(ed)? your (account|card|number)|"
        r"(account|card|number) (will be |)(closed|locked|frozen|terminated|blocked)|police|"
        r"failure to comply|criminal charges)\b",
        "Threatens arrest, legal trouble, or account suspension to scare you "
        "into acting fast."
    ),
    Flag(
        "suspicious_link", 0.45,
        r"(https?://)?(bit\.ly|tinyurl|t\.co|goo\.gl|[0-9]{1,3}(\.[0-9]{1,3}){3})"
        r"|click (here|this link)|verify (here|now)",
        "Contains a shortened/obscured link or an urge to 'click here' — used "
        "to hide the real destination of a phishing site."
    ),
    Flag(
        "callback_request", 0.35,
        r"\bcall (this|us at|back on)\b|\breturn call\b|contact.*within \d+",
        "Pushes you to call an unfamiliar number back, often a premium-rate "
        "or spoofed line."
    ),
    Flag(
        "secrecy_request", 0.6,
        r"\bdo not (tell|discuss|inform)\b|keep this (confidential|private|secret)|"
        r"don'?t (share|mention) this to (anyone|family|bank)",
        "Asks you to keep the conversation secret — a strong sign of social "
        "engineering, since legitimate organizations don't ask for secrecy."
    ),
    Flag(
        "remote_access_request", 0.7,
        r"\b(anydesk|teamviewer|remote access|screen share|install this app|"
        r"download this tool)\b",
        "Asks you to install remote-access software — commonly used to take "
        "control of your device or bank app."
    ),
    Flag(
        "digital_arrest_authority", 0.65,
        r"\b(cbi|central bureau of investigation|enforcement directorate|ed|customs officer|"
        r"delhi police|mumbai police|cyber crime cell|trai|telecom regulatory authority|dot|department of telecom)\b",
        "Impersonates Indian law enforcement or regulatory authorities (CBI, ED, Police, TRAI) which is key to digital arrest setups."
    ),
    Flag(
        "digital_arrest_accusation", 0.70,
        r"\b(mdma|narcotics|drug parcel|illegal package|money laundering|identity theft|"
        r"passport misuse|arrest warrant|illegal advertisements|illegal transaction)\b",
        "Accuses you of illegal activity like drug trafficking or money laundering to induce panic."
    ),
    Flag(
        "digital_arrest_isolation", 0.85,
        r"\b(digital arrest|keep camera on|don'?t turn off (your |)camera|stay in (the |)frame|"
        r"go to a quiet room|do not hang up|skype video|connect to Skype|zoom verification)\b",
        "Demands you stay on a video call in isolation ('digital arrest') — a major red flag as law enforcement never does this."
    ),
    Flag(
        "digital_arrest_finance", 0.80,
        r"\b(verification (of )?funds|verify your (funds|balance|savings)|safety account|rbi clearance|"
        r"refundable deposit|transfer.*(savings|funds)|verify wealth|government account)\b",
        "Demands transferring money to a 'safety' or 'government' account for verification."
    ),
    Flag(
        "caller_spoofing", 0.60,
        r"\b(caller_id=\+92|stir_shaken=FAILED|voip=True)\b",
        "Incoming call shows signs of spoofing, like virtual numbers, international prefixes for local cases, or STIR/SHAKEN failure."
    ),
    Flag(
        "suspicious_video", 0.65,
        r"\b(platform=(Skype|WhatsApp|Zoom)|synthetic_background=True|screen_sharing=True)\b",
        "Video metadata shows signs of scam activity: synthetic backgrounds to mimic police offices, or requests to share screens."
    ),
]

_COMPILED = [(f, re.compile(f.pattern, re.IGNORECASE)) for f in FLAGS]


def format_metadata(metadata: dict) -> str:
    if not metadata:
        return ""
    parts = []
    if "caller_id" in metadata and metadata["caller_id"]:
        parts.append(f"caller_id={metadata['caller_id']}")
    if "voip" in metadata and metadata["voip"]:
        parts.append(f"voip={metadata['voip']}")
    if "stir_shaken" in metadata and metadata["stir_shaken"]:
        parts.append(f"stir_shaken={metadata['stir_shaken']}")
    if "platform" in metadata and metadata["platform"]:
        parts.append(f"platform={metadata['platform']}")
    if "synthetic_background" in metadata and metadata["synthetic_background"]:
        parts.append(f"synthetic_background={metadata['synthetic_background']}")
    if "screen_sharing" in metadata and metadata["screen_sharing"]:
        parts.append(f"screen_sharing={metadata['screen_sharing']}")
    return "\n[Metadata: " + ", ".join(parts) + "]"


def detect_flags(text: str, metadata: dict = None):
    """Return list of (Flag, matched_snippet) for every rule that fires."""
    if metadata:
        text = text + format_metadata(metadata)
    hits = []
    for flag, rx in _COMPILED:
        m = rx.search(text)
        if m:
            hits.append((flag, m.group(0)))
    return hits


def rule_score(text: str, metadata: dict = None) -> float:
    """
    Combine triggered flag weights into a 0-1 'rule score'.
    Uses a soft-OR (1 - product of (1-w)) so multiple weak signals still
    add up, but a single very strong signal isn't diluted.
    """
    hits = detect_flags(text, metadata)
    if not hits:
        return 0.0
    remaining = 1.0
    for flag, _ in hits:
        remaining *= (1 - flag.weight)
    return round(1 - remaining, 4)


def analyze_digital_arrest_flow(text: str) -> dict:
    """
    Analyzes the transcript text and determines which digital arrest scam stages are active.
    Stages:
      1. Contact & Identity: Impersonation of agencies or delivery services.
      2. Accusation & Threat: Allegations of drug parcels, money laundering, SIM blocking.
      3. Digital Arrest / Isolation: Demands to stay on video, lock the room, Skype transition.
      4. Financial Transfer: Demands for fund verification or account transfers.
    """
    stages = {
        1: {
            "name": "Contact & Identity Impersonation",
            "active": False,
            "description": "Scammer impersonates TRAI, FedEx, DHL, or Customs officers.",
            "triggers": [r"\b(fedex|dhl|customs|trai|dot|department of telecom|telecom regulatory)\b"]
        },
        2: {
            "name": "Accusation & Secrecy",
            "active": False,
            "description": "Scammer accuses victim of illegal parcels (MDMA, passports) or money laundering.",
            "triggers": [r"\b(mdma|drugs?|narcotics?|passports?|money laundering|illegal transaction|arrest warrant|cbi|police|ed|enforcement directorate|identity theft)\b"]
        },
        3: {
            "name": "Digital Arrest & Isolation",
            "active": False,
            "description": "Scammer forces victim onto a video call (Skype/Zoom) under 'digital arrest' in a closed room.",
            "triggers": [r"\b(digital arrest|keep camera on|quiet room|don'?t hang up|skype|zoom|stay in frame)\b"]
        },
        4: {
            "name": "Financial Transfer Demand",
            "active": False,
            "description": "Scammer demands victim transfer funds to a 'government safety account' for verification.",
            "triggers": [r"\b(verification (of )?funds|verify your bank|verify.*(funds|balance|savings)|safety account|rbi clearance|refundable deposit|transfer.*(savings|funds)|send money|payment)\b"]
        }
    }
    
    active_stage = 0
    highest_active = 0
    for stage_idx, stage_data in stages.items():
        for trigger in stage_data["triggers"]:
            if re.search(trigger, text, re.IGNORECASE):
                stage_data["active"] = True
                highest_active = max(highest_active, stage_idx)
                break
                
    return {
        "current_stage": highest_active,
        "stages": {idx: {"name": d["name"], "active": d["active"], "description": d["description"]} for idx, d in stages.items()}
    }

