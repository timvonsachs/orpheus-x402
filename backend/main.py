"""FastAPI-Entrypoint fuer Orpheus Acoustic Sense — x402 + Bazaar enabled."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.streaming import router as streaming_router
from config import settings

app = FastAPI(
    title="Orpheus Voice Intelligence API",
    description=(
        "88 acoustic biomarkers from any audio. "
        "Analyzes humanness, engagement, authority, stress, emotion, and environment. "
        "Built for sales call optimization, voice authentication, and mental health monitoring."
    ),
    version=settings.version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── x402 Payment Middleware ───────────────────────────────────────────────────
try:
    from x402.http.middleware.fastapi import payment_middleware, RouteConfig
    from x402.http.facilitator_client import HTTPFacilitatorClient
    from x402.http.x402_http_server import x402ResourceServer
    from x402.mechanisms.evm.exact import ExactEvmServerScheme
    from x402.extensions.bazaar import (
        bazaar_resource_server_extension,
        declare_discovery_extension,
    )

    WALLET     = os.getenv("WALLET_ADDRESS", "0x3A48748098B08d0BdD8dd794A9F2D8F34DD888A1")
    NETWORK    = "eip155:8453"  # Base mainnet
    FACILITATOR_URL = os.getenv(
        "X402_FACILITATOR_URL",
        "https://api.cdp.coinbase.com/platform/v2/x402"
    )

    facilitator = HTTPFacilitatorClient(url=FACILITATOR_URL)
    server = x402ResourceServer(facilitator)
    server.register(NETWORK, ExactEvmServerScheme())
    server.register_extension(bazaar_resource_server_extension)

    BIOMARKER_EXAMPLE = {
        "humanness": {"score": 87.3, "classification": "human"},
        "paralinguistic": {
            "summary": {
                "engagement": 0.74,
                "stress_level": 0.32,
                "confidence_level": 0.81,
                "valence_estimate": "positive",
                "ml_emotion": {"prediction": "neutral", "probability": 0.62},
            },
            "voice_profile": {"authority": 65.2, "authenticity": 78.9},
        },
        "environment": {"environment": "indoor", "noise_level": "low"},
        "audio_duration_seconds": 8.4,
        "processing_time_ms": 312,
    }

    routes = {
        "POST /v1/sense": RouteConfig(
            accepts=[{
                "scheme": "exact",
                "price": "$0.10",
                "network": NETWORK,
                "pay_to": WALLET,
            }],
            description=(
                "Orpheus Voice Intelligence API — 88 acoustic biomarkers from any audio file. "
                "Analyzes humanness, engagement, authority, stress, emotion, confidence, and environment. "
                "Built for sales call optimization, voice authentication, and mental health monitoring. "
                "$0.10 per analysis. No API key required."
            ),
            mime_type="application/json",
            extensions=declare_discovery_extension(
                body_type="multipart/form-data",
                output={
                    "example": BIOMARKER_EXAMPLE,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "humanness": {"type": "object"},
                            "paralinguistic": {"type": "object"},
                            "environment": {"type": "object"},
                            "audio_duration_seconds": {"type": "number"},
                            "processing_time_ms": {"type": "number"},
                        },
                    },
                },
            ),
        ),
    }

    app.middleware("http")(payment_middleware(routes, server))
    print(f"[x402] Payment middleware active — wallet={WALLET} facilitator={FACILITATOR_URL}")
    print(f"[x402] Bazaar discovery: enabled")

except Exception as e:
    print(f"[x402] WARNING: Could not load x402 middleware: {e}")
    print("[x402] Running without payment protection — install x402[fastapi,evm,extensions]")

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router)
app.include_router(streaming_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
