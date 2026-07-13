# Improved BedrockScamDetector (Hackathon Edition)
import os
import json
import re
import boto3
from typing import Optional, Dict, Any
from flags import FLAGS

class BedrockScamDetector:
    """
    AI-powered Digital Public Safety Intelligence detector.
    Hybrid approach:
      • Rule-assisted preprocessing
      • Bedrock LLM reasoning
      • Structured JSON output
    """

    def __init__(
        self,
        model_id: str = "us.amazon.nova-pro-v1:0",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = "us-east-1",
        aws_bearer_token_bedrock: Optional[str] = None,
    ):
        self.model_id = model_id

        if aws_bearer_token_bedrock:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = aws_bearer_token_bedrock
            self.client = boto3.client("bedrock-runtime", region_name=aws_region)
        elif aws_access_key_id and aws_secret_access_key:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        else:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=aws_region,
            )

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        return text.strip()

    def _entities(self, text: str):
        return {
            "phones": re.findall(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b", text),
            "upi_ids": re.findall(r"\b[\w.\-]{2,}@[A-Za-z]{2,}\b", text),
            "urls": re.findall(r"https?://\S+|www\.\S+", text),
        }

    def _default(self):
        return {
            "verdict": "UNKNOWN",
            "risk_level": "LOW",
            "confidence": 0,
            "trust_score": None,
            "scam_probability": None,
            "category": "UNKNOWN",
            "reasons": [],
            "psychological_tactics": [],
            "entities": {},
            "victim_status": "UNKNOWN",
            "recommended_actions": [],
            "estimated_loss": "Unknown",
            "digital_arrest_flow": {
                "current_stage": 0,
                "stages": {}
            }
        }

    def analyze(self, text: str, metadata: dict = None) -> Dict[str, Any]:
        if not text.strip():
            return self._default()

        flags_desc = "\n".join(
            f"- {f.name}: {f.explanation}. Pattern: {f.pattern}"
            for f in FLAGS
        )

        prompt = f"""
ROLE
You are India's AI Digital Public Safety Intelligence Agent.

OBJECTIVE
Analyse transcript + metadata.

TASKS
1. Decide if SAFE or SCAM.
2. Classify category.
3. Estimate confidence.
4. Estimate scam probability.
5. Extract entities.
6. Detect psychological tactics.
7. Detect digital arrest stage.
8. Recommend actions.
9. Minimise false positives.

IMPORTANT
A real request asking a citizen to physically visit a genuine police station,
without asking for money/video calls/secrecy, is usually SAFE.
Also make sure legitimate calls are never classified as scam such as visit physically to legitimate place or legitimate links reduce false positives.

Indian scam categories:
- DIGITAL_ARREST
- UPI
- KYC
- COURIER
- CUSTOMS
- POLICE_IMPERSONATION
- INVESTMENT
- TASK
- JOB
- ELECTRICITY
- BANK
- LOTTERY
- UNKNOWN

Dynamic flags:
{flags_desc}

Example SAFE
Input:
Please come to Mylapore Police Station tomorrow.

Output:
SAFE

Example SCAM
Input:
Your Aadhaar is linked to money laundering.
Stay on Skype.
Transfer ₹50000 immediately.

Output:
DIGITAL_ARREST

Return ONLY JSON:

{{
"verdict":"",
"risk_level":"LOW|MEDIUM|HIGH|CRITICAL",
"confidence":0,
"trust_score":0,
"scam_probability":0,
"category":"",
"reasons":[{{"flag":"","matched":"","why":""}}],
"psychological_tactics":[],
"entities":{{
"phones":[],
"upi_ids":[],
"urls":[],
"banks":[],
"couriers":[],
"government_agencies":[]
}},
"victim_status":"",
"recommended_actions":[],
"estimated_loss":"",
"digital_arrest_flow":{{
"current_stage":0,
"stages":{{}}
}}
}}
"""

        user = text if metadata is None else (
            "Metadata:\n" + json.dumps(metadata, indent=2) +
            "\n\nTranscript:\n" + text
        )

        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user}]
                    }
                ]
            )

            raw = response["output"]["message"]["content"][0]["text"]
            parsed = json.loads(self._extract_json(raw))

            defaults = self._default()
            for k, v in defaults.items():
                parsed.setdefault(k, v)

            ents = self._entities(text)
            parsed["entities"].setdefault("phones", ents["phones"])
            parsed["entities"].setdefault("upi_ids", ents["upi_ids"])
            parsed["entities"].setdefault("urls", ents["urls"])

            return parsed

        except Exception as e:
            out = self._default()
            out["reasons"] = [{
                "flag": "runtime_error",
                "matched": "",
                "why": str(e)
            }]
            return out
