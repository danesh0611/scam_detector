"""
detect.py — command-line interface for ScamShield.

Usage:
    python detect.py "Hi this is your bank, share your OTP to verify"
    python detect.py --file transcript.txt
    echo "some message" | python detect.py
"""

import sys
import argparse
import json

from model import ScamDetector
from bedrock_detector import BedrockScamDetector


def main():
    parser = argparse.ArgumentParser(description="Analyze a message or call transcript for scam signals.")
    parser.add_argument("text", nargs="?", help="The message/transcript text to analyze")
    parser.add_argument("--file", help="Path to a .txt file containing the message/transcript")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of a formatted report")
    
    # Bedrock configuration options
    parser.add_argument("--provider", choices=["local", "bedrock"], default="local", help="Which detection provider to use (local ML/rules or bedrock LLM)")
    parser.add_argument("--model-id", default="meta.llama3-8b-instruct-v1:0", help="AWS Bedrock Model ID")
    parser.add_argument("--aws-access-key-id", help="AWS Access Key ID")
    parser.add_argument("--aws-secret-access-key", help="AWS Secret Access Key")
    parser.add_argument("--aws-region", default="us-east-1", help="AWS Region for Bedrock client")
    parser.add_argument("--aws-bearer-token-bedrock", help="AWS Bedrock bearer token")
    
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("Provide text as an argument, --file, or via stdin.")
        return

    if args.provider == "bedrock":
        detector = BedrockScamDetector(
            model_id=args.model_id,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_region=args.aws_region,
            aws_bearer_token_bedrock=args.aws_bearer_token_bedrock
        )
    else:
        detector = ScamDetector()
        
    result = detector.analyze(text)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 60)
    print(f"VERDICT:            {result['verdict']}")
    print(f"Trust score:        {result['trust_score']} / 100  (higher = more trustworthy)")
    print(f"Scam probability:   {result['scam_probability']}%")
    if result.get('ml_probability') is not None:
        print(f"  - ML model says:  {result['ml_probability']}% scam-like")
    if result.get('rule_score') is not None:
        print(f"  - Rule engine:    {result['rule_score']}% red-flag intensity")
    print("=" * 60)
    if result.get("reasons"):
        print("\nRed flags detected:")
        for r in result["reasons"]:
            print(f"  - [{r['flag']}] matched \"{r['matched']}\"")
            print(f"      -> {r['why']}")
    else:
        print("\nNo rule-based red flags detected.")
    print()


if __name__ == "__main__":
    main()

