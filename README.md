# ScamShield

Analyzes a text message, chat, or call transcript and outputs a **trust
score + verdict** (Likely Safe / Suspicious / Likely Scam), along with
exactly which red flags fired and why.

It's a hybrid of two things, blended together:
- **Rule engine** (`flags.py`) — regex-based detectors for the classic
  scam tactics: urgency, OTP/credential requests, payment via gift
  cards/crypto, impersonation of banks/government, threats, secrecy
  requests, remote-access requests, suspicious links.
- **ML model** (`model.py`) — a TF-IDF + Logistic Regression classifier
  trained on labeled scam/legit examples, so it can catch phrasing the
  rules don't explicitly cover.

## Setup

```bash
pip install scikit-learn joblib
python model.py        # trains and saves scam_model.joblib (one-time)
```

## Usage

```bash
python detect.py "Congratulations! You won $10,000, send your bank details to claim"
python detect.py --file some_transcript.txt
python detect.py --json "some message"          # machine-readable output
echo "some message" | python detect.py          # via stdin
```

Or from Python directly:

```python
from model import ScamDetector

detector = ScamDetector()
result = detector.analyze("Your OTP is required to reverse a charge, don't tell anyone")

print(result["verdict"])         # "LIKELY SCAM"
print(result["trust_score"])     # 0-100, higher = more trustworthy
print(result["reasons"])         # list of triggered flags + explanations
```

## Call recordings

The model works on **text**, so a phone call needs to be transcribed
first. See `transcribe_stub.py` for exact snippets to wire in:
- **Whisper** or **faster-whisper** (free, local, offline)
- A cloud STT API (Google/AWS/Deepgram) for higher accuracy or live
  streaming

Once transcribed, feed the transcript text straight into
`ScamDetector.analyze()`. For live calls, transcribe in rolling chunks
and re-run `analyze()` on the growing transcript to get a live-updating
trust score as the call happens — that's the same approach real
scam-call-blocker apps use.

## Improving accuracy

`data/sample_data.py` has ~40 hand-written examples — enough to
demonstrate the approach, not enough for production accuracy. To make
this genuinely robust:
1. Swap in a real labeled dataset (e.g. the SMS Spam Collection
   dataset, or your own logged messages/call transcripts).
2. Retrain with `python model.py`.
3. Tune `ML_WEIGHT` / `RULE_WEIGHT` in `model.py` based on which
   signal proves more reliable on your data.
4. Add new regex rules to `flags.py` as you spot new scam scripts —
   scam wording evolves, so treat the rule list as a living document.

## Project layout

```
scamshield/
├── flags.py            # rule-based red flag detector
├── model.py             # TF-IDF + LogisticRegression, blends with rules
├── detect.py             # CLI entry point
├── transcribe_stub.py     # how to turn audio calls into text first
├── data/
│   └── sample_data.py    # seed labeled examples (swap for real data)
└── README.md
```

## Limitations (read before relying on this)

- The sample dataset is tiny — treat this as a working prototype/
  reference architecture, not a production-grade classifier.
- Rule-based regexes will miss scams phrased in new/unusual ways, and
  can false-positive on legitimate urgent messages (e.g. a real bank
  fraud alert). Always use trust_score as a signal to investigate
  further, not an automatic block/allow decision — especially for
  anything involving real financial action.
- No metadata (sender reputation, phone number spoofing checks, link
  destination scanning) is used — this is purely a language/content
  analysis. Combine with those signals for a real deployment.
