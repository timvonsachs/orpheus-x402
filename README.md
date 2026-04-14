# Orpheus Voice Intelligence API

88 acoustic biomarkers from any audio. x402 native. No API key.

## Endpoint

```
POST https://orpheus-x402-production.up.railway.app/v1/sense
```

## Pricing

| Stream | Biomarkers | Price |
|--------|-----------|-------|
| Full Spectrum | 88 biomarkers | $0.10 USDC |
| Authority Stream | 12 biomarkers | $0.03 USDC |
| Emotion Stream | 15 biomarkers | $0.03 USDC |
| Health Stream | 18 biomarkers | $0.05 USDC |
| Authenticity Stream | 10 biomarkers | $0.04 USDC |

## Free Endpoints

```
GET /health
GET /pricing
```

## Quickstart (Python)

```python
#!/usr/bin/env python3
"""
Orpheus x402 — Voice Biomarker API Quickstart
===============================================
Install: pip install -r requirements.txt
Run:     python quickstart.py [audio_file.wav]

Orpheus analyzes audio and returns 88 voice biomarkers including:
  engagement, authority, stress_level, humanness, emotion
Payment: $0.10 USDC on Base — handled automatically by x402-python
"""

import os
import sys
import struct
import asyncio
import io
import wave
import math
from pathlib import Path

# ── Dependencies check ────────────────────────────────────────────────────────
try:
    import httpx
    from x402.client.httpx import wrap_httpx_client
    from eth_account import Account
except ImportError:
    print("Install: pip install x402-python httpx eth-account")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
ORPHEUS_URL = "https://orpheus-x402-production.up.railway.app"
PRICE       = "$0.10"         # Full 88-biomarker analysis
NETWORK     = "base"          # Base mainnet (USDC)
VERBOSE     = os.getenv("VERBOSE", "0") == "1"


# ── Wallet ────────────────────────────────────────────────────────────────────
def load_or_create_wallet():
    """Load wallet from env or generate a burner for testing."""
    pk = os.getenv("PRIVATE_KEY")
    if pk:
        acct = Account.from_key(pk)
        print(f"Wallet: {acct.address}")
        return acct

    # Generate burner wallet
    acct = Account.create()
    print(f"⚠️  Generated burner wallet: {acct.address}")
    print(f"    Private key: {acct.key.hex()}")
    print(f"    Fund with USDC on Base: https://app.uniswap.org")
    print(f"    Or use Base Sepolia testnet for free testing.")
    return acct


# ── Demo audio generator ─────────────────────────────────────────────────────
def generate_test_wav(duration_s: float = 2.0, freq: float = 440.0) -> bytes:
    """Generate a simple sine wave WAV for testing structure without real audio."""
    sample_rate = 16000
    n_samples   = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = bytes(struct.pack("<h", int(32767 * math.sin(2 * math.pi * freq * i / sample_rate)))
                        for i in range(n_samples))
        wf.writeframes(samples)
    return buf.getvalue()


# ── Core API call ─────────────────────────────────────────────────────────────
def analyze_voice(audio_bytes: bytes, filename: str = "audio.wav", account=None) -> dict:
    """
    POST audio to Orpheus /v1/sense.
    x402-python handles the USDC payment automatically on HTTP 402.
    """
    with httpx.Client() as client:
        # Wrap client with x402 auto-payment
        paid_client = wrap_httpx_client(
            client,
            private_key=account.key.hex() if account else None,
            network=NETWORK,
        )

        response = paid_client.post(
            f"{ORPHEUS_URL}/v1/sense",
            files={"file": (filename, audio_bytes, "audio/wav")},
            timeout=30.0,
        )

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text[:300]}")

    return response.json()


# ── Pretty print ──────────────────────────────────────────────────────────────
def print_biomarkers(result: dict):
    """Print key biomarkers in a readable format."""
    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    def bar(value, max_val=1.0, width=20):
        if value is None: return "N/A"
        filled = int((value / max_val) * width)
        return f"[{'█' * filled}{'░' * (width - filled)}] {value:.2f}"

    h  = result.get("humanness", {})
    sm = result.get("paralinguistic", {}).get("summary", {})
    vp = result.get("paralinguistic", {}).get("voice_profile", {})
    em = result.get("paralinguistic", {}).get("summary", {}).get("ml_emotion", {})
    env = result.get("environment", {})

    print("\n  ── ORPHEUS BIOMARKERS ──────────────────────────")
    print(f"  Humanness:    {bar(h.get('score', 0), 100, 20)}  ({h.get('classification', '?')})")
    print(f"  Engagement:   {bar(sm.get('engagement'), 1.0)}")
    print(f"  Authority:    {bar(vp.get('authority', 0), 100)}")
    print(f"  Stress:       {bar(sm.get('stress_level'), 1.0)}")
    print(f"  Confidence:   {bar(sm.get('confidence_level'), 1.0)}")
    print(f"  Valence:      {sm.get('valence_estimate', '?')}")
    print(f"  Emotion:      {em.get('prediction', '?')} (p={em.get('probability', 0):.2f})")
    print(f"  Environment:  {env.get('environment', '?')} | noise={env.get('noise_level', '?')}")
    print(f"  Duration:     {result.get('audio_duration_seconds', '?')}s")
    print(f"  Proc time:    {result.get('processing_time_ms', '?')}ms")
    print("  ────────────────────────────────────────────────")

    if VERBOSE:
        import json
        print("\n  Full JSON:")
        print(json.dumps(result, indent=4, ensure_ascii=False))


# ── BONUS: Bland.ai live-listen integration ───────────────────────────────────
async def bland_live_analysis(call_id: str, bland_api_key: str, account=None):
    """
    Connect to Bland.ai live-listen WebSocket and analyze each 500ms audio chunk.
    Requires: pip install websockets aiohttp

    Usage:
        asyncio.run(bland_live_analysis("your-call-id", "your-bland-key"))
    """
    try:
        import websockets
        import aiohttp
    except ImportError:
        print("Install: pip install websockets aiohttp")
        return

    SAMPLE_RATE  = 16000
    SAMPLE_WIDTH = 2
    BUFFER_SECS  = 0.5
    BUFFER_FRAMES = int(SAMPLE_RATE * BUFFER_SECS)  # 8000 samples

    def pcm_to_wav(pcm_bytes: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    # Step 1: Get WSS URL from Bland.ai
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://api.bland.ai/v1/calls/{call_id}/listen",
            headers={"authorization": bland_api_key},
        ) as resp:
            data = await resp.json()
            ws_url = data.get("data", {}).get("url")
            if not ws_url:
                print(f"Could not get WSS URL: {data}")
                return

    print(f"Connected to Bland.ai stream: {call_id}")

    # Step 2: Stream audio → Orpheus
    pcm_buffer = bytearray()
    chunk_idx  = 0

    async with websockets.connect(ws_url) as ws:
        async for message in ws:
            if not isinstance(message, bytes):
                continue

            pcm_buffer.extend(message)
            n_samples = len(pcm_buffer) // SAMPLE_WIDTH

            if n_samples >= BUFFER_FRAMES:
                chunk_bytes = bytes(pcm_buffer[:BUFFER_FRAMES * SAMPLE_WIDTH])
                pcm_buffer  = pcm_buffer[BUFFER_FRAMES * SAMPLE_WIDTH:]

                # Analyze this 500ms chunk
                try:
                    wav = pcm_to_wav(chunk_bytes)
                    result = analyze_voice(wav, f"chunk_{chunk_idx}.wav", account)

                    t = round(chunk_idx * BUFFER_SECS, 1)
                    eng = result.get("paralinguistic", {}).get("summary", {}).get("engagement", "?")
                    auth = result.get("paralinguistic", {}).get("voice_profile", {}).get("authority", "?")
                    hum = result.get("humanness", {}).get("score", "?")
                    print(f"  t={t:5.1f}s  engagement={eng:.2f}  authority={auth:.1f}  humanness={hum:.1f}")

                    # Hook: steer agent based on biomarkers
                    # if eng < 0.4: trigger_cta_now()
                    # if eng > 0.85: send_calendly_link()

                except Exception as e:
                    print(f"  t={chunk_idx * BUFFER_SECS:.1f}s  error: {e}")

                chunk_idx += 1


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Orpheus x402 — Voice Biomarker API")
    print(f"Endpoint: {ORPHEUS_URL}")
    print(f"Price:    {PRICE} USDC per call\n")

    # Check service is live
    try:
        r = httpx.get(f"{ORPHEUS_URL}/health", timeout=5)
        health = r.json()
        print(f"✓ Service alive — {health.get('biomarkers', '?')} biomarkers, v{health.get('version', '?')}")
    except Exception as e:
        print(f"✗ Service unreachable: {e}")
        return

    # Load wallet
    print()
    account = load_or_create_wallet()

    # Load or generate audio
    print()
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
        if not audio_path.exists():
            print(f"File not found: {audio_path}")
            return
        audio_bytes = audio_path.read_bytes()
        filename    = audio_path.name
        print(f"Analyzing: {filename} ({len(audio_bytes):,} bytes)")
    else:
        print("No audio file provided — generating 2s test tone (440 Hz)")
        audio_bytes = generate_test_wav()
        filename    = "test_tone.wav"

    # Analyze
    print(f"Calling /v1/sense (payment: {PRICE} USDC on Base)...")
    try:
        result = analyze_voice(audio_bytes, filename, account)
        print_biomarkers(result)
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nNote: Without real USDC on Base, the API returns HTTP 402.")
        print("Fund your wallet or use Base Sepolia testnet for testing.")


if __name__ == "__main__":
    main()
```

## Use Cases

- Sales call optimization
- Customer support quality monitoring
- Voice authentication
- Mental health monitoring
- Podcast / content analysis

## How it works

1. Send audio + USDC payment via x402
2. Receive 88 biomarkers as JSON
3. No account. No API key. Just pay and use.

## Built with

- [OpenSMILE](https://www.audeering.com/research/opensmile/) — acoustic feature extraction
- [x402 protocol](https://x402.org) — permissionless payments on Base
- [Railway](https://railway.app) — hosting

## Contact

tim@animacompliance.com
