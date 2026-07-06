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
]

_COMPILED = [(f, re.compile(f.pattern, re.IGNORECASE)) for f in FLAGS]


def detect_flags(text: str):
    """Return list of (Flag, matched_snippet) for every rule that fires."""
    hits = []
    for flag, rx in _COMPILED:
        m = rx.search(text)
        if m:
            hits.append((flag, m.group(0)))
    return hits


def rule_score(text: str) -> float:
    """
    Combine triggered flag weights into a 0-1 'rule score'.
    Uses a soft-OR (1 - product of (1-w)) so multiple weak signals still
    add up, but a single very strong signal isn't diluted.
    """
    hits = detect_flags(text)
    if not hits:
        return 0.0
    remaining = 1.0
    for flag, _ in hits:
        remaining *= (1 - flag.weight)
    return round(1 - remaining, 4)
