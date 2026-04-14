const express = require('express');
const { paymentMiddleware } = require('x402-express');
const multer = require('multer');
const fetch = require('node-fetch');
const FormData = require('form-data');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const ORPHEUS_BACKEND = process.env.ORPHEUS_BACKEND_URL || 'http://localhost:8001';
const WALLET          = process.env.WALLET_ADDRESS || '0x3A48748098B08d0BdD8dd794A9F2D8F34DD888A1';

// ── Facilitator ───────────────────────────────────────────────────────────────
// CDP facilitator (mainnet + free tier 1,000 tx/month)
// Fallback: openx402.ai (no auth required)
const FACILITATOR_URL = process.env.CDP_FACILITATOR_URL || 'https://api.cdp.coinbase.com/platform/v2/x402';

const facilitatorConfig = { url: FACILITATOR_URL };

// CDP API keys for mainnet (optional — falls back to openx402.ai without them)
if (process.env.CDP_API_KEY_ID && process.env.CDP_API_KEY_SECRET) {
  facilitatorConfig.cdpApiKeyId     = process.env.CDP_API_KEY_ID;
  facilitatorConfig.cdpApiKeySecret = process.env.CDP_API_KEY_SECRET;
}

// ── Biomarker output schema (for Bazaar discovery) ────────────────────────────
const BIOMARKER_OUTPUT_SCHEMA = {
  type: 'object',
  properties: {
    humanness: {
      type: 'object',
      properties: {
        score:          { type: 'number', description: 'Humanness score 0-100' },
        classification: { type: 'string', description: 'human|ai|uncertain' }
      }
    },
    paralinguistic: {
      type: 'object',
      properties: {
        summary: {
          type: 'object',
          properties: {
            engagement:        { type: 'number', description: 'Listener engagement 0-1' },
            stress_level:      { type: 'number', description: 'Stress level 0-1' },
            confidence_level:  { type: 'number', description: 'Confidence 0-1' },
            valence_estimate:  { type: 'string', description: 'positive|negative|neutral' },
            ml_emotion: {
              type: 'object',
              properties: {
                prediction:  { type: 'string' },
                probability: { type: 'number' }
              }
            }
          }
        },
        voice_profile: {
          type: 'object',
          properties: {
            authority:   { type: 'number', description: 'Authority score 0-100' },
            authenticity:{ type: 'number', description: 'Authenticity score 0-100' }
          }
        }
      }
    },
    environment: {
      type: 'object',
      properties: {
        environment: { type: 'string', description: 'indoor|outdoor|vehicle|studio' },
        noise_level: { type: 'string', description: 'low|medium|high' }
      }
    },
    audio_duration_seconds: { type: 'number' },
    processing_time_ms:     { type: 'number' }
  }
};

const BIOMARKER_OUTPUT_EXAMPLE = {
  humanness: { score: 87.3, classification: 'human' },
  paralinguistic: {
    summary: {
      engagement: 0.74,
      stress_level: 0.32,
      confidence_level: 0.81,
      valence_estimate: 'positive',
      ml_emotion: { prediction: 'neutral', probability: 0.62 }
    },
    voice_profile: { authority: 65.2, authenticity: 78.9 }
  },
  environment: { environment: 'indoor', noise_level: 'low' },
  audio_duration_seconds: 8.4,
  processing_time_ms: 312
};

// ── x402 Payment Middleware ───────────────────────────────────────────────────
app.use(paymentMiddleware(
  WALLET,
  {
    'POST /v1/sense': {
      price: '$0.10',
      network: 'base',
      config: {
        description: 'Orpheus Voice Intelligence — 88 acoustic biomarkers from any audio. Humanness, engagement, authority, stress, emotion, environment. $0.10/call. No API key.',
        mimeType: 'application/json',
        discoverable: true,
        outputSchema: BIOMARKER_OUTPUT_SCHEMA,
      }
    },
    'POST /v1/sense/authority': {
      price: '$0.03',
      network: 'base',
      config: {
        description: 'Orpheus Authority Stream — 12 biomarkers: authority score, confidence, assertiveness. $0.03/call.',
        mimeType: 'application/json',
        discoverable: true,
      }
    },
    'POST /v1/sense/emotion': {
      price: '$0.03',
      network: 'base',
      config: {
        description: 'Orpheus Emotion Stream — 15 biomarkers: valence, arousal, emotion prediction, stress, engagement. $0.03/call.',
        mimeType: 'application/json',
        discoverable: true,
      }
    },
    'POST /v1/sense/health': {
      price: '$0.05',
      network: 'base',
      config: {
        description: 'Orpheus Health Stream — 18 voice health biomarkers: vocal fatigue, tremor, breathiness. $0.05/call.',
        mimeType: 'application/json',
        discoverable: true,
      }
    },
    'POST /v1/sense/authenticity': {
      price: '$0.04',
      network: 'base',
      config: {
        description: 'Orpheus Authenticity Stream — 10 biomarkers: humanness score, deepfake detection, authenticity. $0.04/call.',
        mimeType: 'application/json',
        discoverable: true,
      }
    },
  },
  facilitatorConfig
));

// ── Route Handlers ────────────────────────────────────────────────────────────

// Full Spectrum — proxy to Python backend
app.post('/v1/sense', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No audio file provided' });

    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname || 'audio.wav',
      contentType: req.file.mimetype || 'audio/wav'
    });

    const orpheusResponse = await fetch(`${ORPHEUS_BACKEND}/v1/sense`, {
      method: 'POST',
      body: form,
      headers: form.getHeaders()
    });

    if (!orpheusResponse.ok) {
      const err = await orpheusResponse.text();
      return res.status(orpheusResponse.status).json({ error: err });
    }

    const biomarkers = await orpheusResponse.json();
    res.json(biomarkers);
  } catch (err) {
    console.error('Sense error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Stream endpoints — proxy with stream filter param
async function proxyStream(req, res, stream) {
  try {
    if (!req.file) return res.status(400).json({ error: 'No audio file provided' });

    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname || 'audio.wav',
      contentType: req.file.mimetype || 'audio/wav'
    });

    const orpheusResponse = await fetch(`${ORPHEUS_BACKEND}/v1/sense?stream=${stream}`, {
      method: 'POST',
      body: form,
      headers: form.getHeaders()
    });

    if (!orpheusResponse.ok) {
      const err = await orpheusResponse.text();
      return res.status(orpheusResponse.status).json({ error: err });
    }

    const biomarkers = await orpheusResponse.json();
    res.json(biomarkers);
  } catch (err) {
    console.error(`Stream ${stream} error:`, err.message);
    res.status(500).json({ error: err.message });
  }
}

app.post('/v1/sense/authority',    upload.single('audio'), (req, res) => proxyStream(req, res, 'authority'));
app.post('/v1/sense/emotion',      upload.single('audio'), (req, res) => proxyStream(req, res, 'emotion'));
app.post('/v1/sense/health',       upload.single('audio'), (req, res) => proxyStream(req, res, 'health'));
app.post('/v1/sense/authenticity', upload.single('audio'), (req, res) => proxyStream(req, res, 'authenticity'));

// ── Free Endpoints ────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({
    status: 'alive',
    biomarkers: 88,
    version: '1.1.0',
    bazaar: true,
    network: 'base',
    wallet: WALLET
  });
});

app.get('/pricing', (req, res) => {
  res.json({
    currency: 'USDC',
    network: 'base',
    streams: {
      full_spectrum:   { price: '0.10', biomarkers: 88, endpoint: 'POST /v1/sense' },
      authority:       { price: '0.03', biomarkers: 12, endpoint: 'POST /v1/sense/authority' },
      emotion:         { price: '0.03', biomarkers: 15, endpoint: 'POST /v1/sense/emotion' },
      health:          { price: '0.05', biomarkers: 18, endpoint: 'POST /v1/sense/health' },
      authenticity:    { price: '0.04', biomarkers: 10, endpoint: 'POST /v1/sense/authenticity' }
    },
    bazaar_discoverable: true,
    quickstart: 'https://github.com/timvonsachs/orpheus-x402'
  });
});

// Example output for Bazaar discovery browsers
app.get('/example', (req, res) => {
  res.json(BIOMARKER_OUTPUT_EXAMPLE);
});

// ── Start ─────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 8402;
app.listen(PORT, () => {
  console.log(`Orpheus x402 v1.1.0 — port ${PORT}`);
  console.log(`Wallet:      ${WALLET}`);
  console.log(`Facilitator: ${FACILITATOR_URL}`);
  console.log(`Bazaar:      enabled (5 streams discoverable)`);
  console.log(`Streams:     Full $0.10 | Authority $0.03 | Emotion $0.03 | Health $0.05 | Authenticity $0.04`);
});
