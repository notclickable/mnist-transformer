Architecture:


Input Image (28x28)
       ↓
Split into Patches (4x4 grid of 7x7 patches)
       ↓
[ENCODER]
┌──────────────────────────────────┐
│ Patch Embedding (49 → 64 dim)    │
│       ↓                          │
│ Add CLS Token ([CLS] + 16 patches)│
│       ↓                          │
│ Add Positional Embeddings        │
│       ↓                          │
│ Transformer Encoder Layers (x2)  │
│  ┌────────────────────┐         │
│  │ Self-Attention     │         │
│  │       ↓            │         │
│  │ Feed Forward       │         │
│  └────────────────────┘         │
└──────────────────────────────────┘
       ↓
    Memory
       ↓
[DECODER]
┌──────────────────────────────────┐
│ Target Embedding                 │
│       ↓                          │
│ Add Positional Embeddings        │
│       ↓                          │
│ Transformer Decoder Layers (x2)  │
│  ┌────────────────────┐         │
│  │ Self-Attention     │         │
│  │ (with causal mask) │         │
│  │       ↓            │         │
│  │ Cross-Attention    │         │
│  │ (with memory)      │         │
│  │       ↓            │         │
│  │ Feed Forward       │         │
│  └────────────────────┘         │
└──────────────────────────────────┘
       ↓
Output Projection (64 → 10 classes)
       ↓
Final Predictions

Data shapes at each step:

Input: [batch_size, 1, 28, 28]
Patches: [batch_size, 16, 49]
After Embedding: [batch_size, 16, 64]
With CLS: [batch_size, 17, 64]
Encoder Output: [batch_size, 17, 64]
Decoder Input: [batch_size, seq_len, 64]
Final Output: [batch_size, seq_len, 10]

Key features:
The encoder processes the image patches
The decoder generates outputs autoregressively
The architecture uses both self-attention and cross-attention
Position embeddings are added in both encoder and decoder
The CLS token is used for classification
The model can handle sequence generation, though in this case it's just predicting class labels