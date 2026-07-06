import os
import json
import boto3
from typing import Optional, Dict, Any
from flags import FLAGS

class BedrockScamDetector:
    """
    BedrockScamDetector uses an LLM on Amazon Bedrock to analyze messages/transcripts
    for scam/fraud indicators, with specialized context for Indian scam tactics.
    """
    def __init__(
        self,
        model_id: str = "meta.llama3-8b-instruct-v1:0",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = "us-east-1",
        aws_bearer_token_bedrock: Optional[str] = None
    ):
        self.model_id = model_id
        
        # Configure Bedrock client depending on provided authentication method
        if aws_bearer_token_bedrock:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = aws_bearer_token_bedrock
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=aws_region
            )
        elif aws_access_key_id and aws_secret_access_key:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            # Fall back to AWS environment variables or default credential chain
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=aws_region
            )

    def _extract_json(self, text: str) -> str:
        """Helper to extract clean JSON from LLM response in case of markdown block wrapping."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def analyze(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {
                "verdict": "UNKNOWN",
                "trust_score": None,
                "scam_probability": None,
                "reasons": [],
            }

        # Build list of dynamic flags to pass to the model from flags.py
        flags_desc = "\n".join([
            f"- '{flag.name}': {flag.explanation} (Clues to match: {flag.pattern})"
            for flag in FLAGS
        ])

        system_prompt = f"""You are an expert security system designed to analyze messages and calls for potential scam/fraud signals.
This system is deployed in India, so you should pay special attention to common Indian scam tactics, such as:
1. UPI scams (requesting to scan a QR code to 'receive' money, Paytm/GPay/PhonePe issues, pending UPI transactions, or UPI PIN requests).
2. Courier scams (impersonating FedEx, DTDC, India Post, or customs officers claiming illegal items were found in a package addressed to you).
3. Authority impersonation (fake police officer, TRAI/DoT calling to block SIM card/phone number, electricity board threatening immediate power cut, customs/tax departments).
4. Bank/KYC updates (SBI, HDFC, ICICI, etc., asking to update KYC, PAN card, or bank details via a suspicious link).
5. Job offer or task scams (offering part-time work like rating movies or liking YouTube videos for money).

Analyze the provided text and classify it using the following list of flags if applicable:
{flags_desc}

Output your analysis strictly in JSON format. The JSON must have the following structure:
{{
  "verdict": "LIKELY SAFE" | "SUSPICIOUS — proceed with caution" | "LIKELY SCAM",
  "trust_score": float (between 0 and 100, where 100 is fully safe/trustworthy, and 0 is complete scam),
  "scam_probability": float (between 0 and 100),
  "reasons": [
     {{
       "flag": "one of the flag names listed above",
       "matched": "exact substring or phrase from the input text that triggered this flag",
       "why": "brief explanation of why this triggers the flag in the Indian scam context"
     }}
  ]
}}

Only return the raw JSON object. Do not wrap in markdown code blocks or add any other text.
"""

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[
                    {"role": "user", "content": [{"text": text}]}
                ],
                system=[{"text": system_prompt}]
            )
            raw_output = response['output']['message']['content'][0]['text']
            clean_output = self._extract_json(raw_output)
            
            result = json.loads(clean_output)
            
            # Ensure required keys exist
            for key in ["verdict", "trust_score", "scam_probability", "reasons"]:
                if key not in result:
                    result[key] = None if key != "reasons" else []
            return result

        except Exception as e:
            return {
                "verdict": "UNKNOWN (Error during Bedrock call)",
                "trust_score": None,
                "scam_probability": None,
                "reasons": [{"flag": "error", "matched": "", "why": str(e)}],
            }
