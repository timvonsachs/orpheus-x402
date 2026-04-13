const express = require('express');
const { paymentMiddleware } = require('x402-express');
const multer = require('multer');
const fetch = require('node-fetch');
const FormData = require('form-data');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// x402 Payment Config
const PRICE = '0.10';  // USDC
const WALLET = '0x3A48748098B08d0BdD8dd794A9F2D8F34DD888A1';
const FACILITATOR = 'https://facilitator.openx402.ai';

// x402 Payment Middleware (global, vor den Routes registrieren)
app.use(paymentMiddleware(
  WALLET,
  {
    'POST /v1/sense': { price: `$${PRICE}`, network: 'base' }
  },
  { url: FACILITATOR }
));

// x402 Middleware auf /v1/sense
app.post('/v1/sense', 
  upload.single('audio'),
  async (req, res) => {
    // Forward to Orpheus
    const form = new FormData();
    form.append('audio', req.file.buffer, req.file.originalname);
    
    const orpheusResponse = await fetch('http://localhost:8001/v1/sense', {
      method: 'POST',
      body: form
    });
    
    const biomarkers = await orpheusResponse.json();
    res.json(biomarkers);
  }
);

// Freier Health-Check (kein x402)
app.get('/health', (req, res) => {
  res.json({ status: 'alive', biomarkers: 88, version: '1.0' });
});

// Pricing-Info (kein x402)
app.get('/pricing', (req, res) => {
  res.json({
    full_spectrum: { price: '0.10', currency: 'USDC', biomarkers: 88 },
    authority_stream: { price: '0.03', currency: 'USDC', biomarkers: 12 },
    emotion_stream: { price: '0.03', currency: 'USDC', biomarkers: 15 },
    health_stream: { price: '0.05', currency: 'USDC', biomarkers: 18 },
    authenticity_stream: { price: '0.04', currency: 'USDC', biomarkers: 10 }
  });
});

const PORT = process.env.PORT || 8402;
app.listen(PORT, () => {
  console.log(`Orpheus x402 Service live on port ${PORT}`);
  console.log('Wallet: ' + WALLET);
  console.log('Price: $' + PRICE + ' USDC per analysis');
});
