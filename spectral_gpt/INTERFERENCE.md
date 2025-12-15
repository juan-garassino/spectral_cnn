# 🌊 Interference Attention Architecture Audit

## Overview

This document provides a complete audit of the `interference_attention` experiment architecture to understand what components are used and how they differ from both standard transformers and pure wave.

**Architecture Type**: HYBRID (Wave Embeddings + Embedding Space + Interference Attention)

---

## Architecture Comparison

| Component | Standard Transformer | Interference Attention | Pure Wave |
|-----------|---------------------|------------------------|-----------|
| Token Input | `nn.Embedding` | `WavePacketEmbedding` | `PureWaveExcitation` |
| Internal Representation | Embedding vectors | Embedding vectors | Wave parameters |
| Attention | `Q @ K.T` + softmax | Wave interference + row norm | Wave interference + row norm |
| MLP | Standard MLP | `WaveResonanceMLP` | `PureWaveMLP` |
| Output | Linear projection | `CollapseHead` | `WaveCollapse` |

**Key Insight**: `interference_attention` is a **hybrid** - it uses wave embeddings but still operates in embedding space (d_model dimension), not pure wave parameter space.

---

## Architecture Flow

```
Token IDs → WavePacketEmbedding → Embedding (B, T, d_model)
                                      ↓
                              WaveBlock × N (with InterferenceAttention)
                                      ↓
                              CollapseHead → Logits
```

**Note**: Unlike PureWaveGPT, this architecture converts waves to embeddings and operates in embedding space.

---

## Classes Used in interference_attention

### 1. `WaveGPT` (Main Model)

**Location**: `wave_gpt.py:1687`

**Configuration for interference_attention**:
```python
WaveGPTConfig(
    model_type="wave",
    use_interference_attention=True,  # Enables InterferenceAttention
    use_wave_embeddings=True          # Uses WavePacketEmbedding
)
```

**Components**:
- `self.embedding` → `WavePacketEmbedding` (wave → embedding projection)
- `self.blocks` → `nn.ModuleList[WaveBlock]` (with InterferenceAttention)
- `self.ln_f` → `nn.LayerNorm(d_model)` (standard LayerNorm!)
- `self.head` → `CollapseHead` (embedding → logits)

**⚠️ HYBRID**: Uses `nn.LayerNorm` on embedding space, not wave parameters.

---

### 2. `WavePacketEmbedding` (Token → Wave → Embedding)

**Location**: `wave_gpt.py:178`

**Purpose**: Convert tokens to wave packets, then project to embedding space

**Parameters**:
| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `base_freqs` | `(vocab_size, num_waves)` | `nn.Parameter` | Learnable frequencies |
| `phases` | `(vocab_size, num_waves)` | `nn.Parameter` | Learnable phases |
| `harmonic_amps` | `(vocab_size, num_waves, num_harmonics)` | `nn.Parameter` | Learnable amplitudes |
| `wave_to_embed` | `nn.Linear(wave_dim, d_model)` | `nn.Linear` | **Wave → Embedding projection!** |
| `simple_embed` | `nn.Embedding(vocab_size, d_model)` | `nn.Embedding` | Standard embedding for annealing |

**Forward Pass**:
```python
def forward(self, token_ids, standard_embed_ratio=0.0):
    # Get wave parameters
    base_f = self.base_freqs[token_ids]    # (B, T, num_waves)
    phases = self.phases[token_ids]         # (B, T, num_waves)
    harm_a = self.harmonic_amps[token_ids]  # (B, T, num_waves, num_harmonics)
    
    # Generate wave packet
    sin_waves = harm_a * torch.sin(wave_phase)
    cos_waves = harm_a * torch.cos(wave_phase)
    
    # Flatten wave state
    wave_state = torch.cat([sin_waves, cos_waves, phase_direct, freq_direct], dim=-1)
    
    # PROJECT TO EMBEDDING SPACE!
    wave_embed = self.wave_to_embed(wave_state)  # (B, T, d_model)
    
    return embeddings  # Returns embedding vectors, NOT wave parameters!
```

**⚠️ HYBRID**: The `wave_to_embed` projection converts wave parameters to embedding space. This is the key difference from PureWaveGPT.

---

### 3. `WaveBlock` (Transformer Block)

**Location**: `wave_gpt.py:1012`

**Components**:
- `self.ln1` → `nn.LayerNorm(d_model)` (standard LayerNorm on embeddings!)
- `self.ln2` → `nn.LayerNorm(d_model)` (standard LayerNorm on embeddings!)
- `self.attn` → `InterferenceAttention` (when `use_interference_attention=True`)
- `self.mlp` → `WaveResonanceMLP`

**Forward Pass**:
```python
def forward(self, x):
    # x is embedding tensor (B, T, d_model), NOT wave parameters!
    x = x + self.attn(self.ln1(x))  # LayerNorm on embeddings
    x = x + self.mlp(self.ln2(x))   # LayerNorm on embeddings
    return x
```

**⚠️ HYBRID**: Uses `nn.LayerNorm` on embedding vectors, not wave parameters.

---

### 4. `InterferenceAttention` (Physics-Based Attention)

**Location**: `wave_gpt.py:608`

**Purpose**: Compute attention using wave interference physics instead of dot product

**⚠️ NOTE**: This class has TWO different signatures depending on usage:
1. **In WaveBlock (hybrid)**: Takes embedding tensor `x: (B, T, d_model)`
2. **In PureWaveBlock**: Takes wave parameters `(freqs, phases, amps)`

**For interference_attention experiment, it's used in HYBRID mode with embeddings.**

**Parameters**:
```python
# Embedding → Wave projections
self.freq_transform = nn.Linear(num_waves, num_heads * num_waves)
self.phase_transform = nn.Linear(num_waves, num_heads * num_waves)
self.amp_transform = nn.Linear(num_waves * num_harmonics, num_heads * num_waves)

# Value projections (wave-to-wave)
self.value_freq_transform = nn.Linear(num_waves, num_waves)
self.value_phase_transform = nn.Linear(num_waves, num_waves)
self.value_amp_transform = nn.Linear(num_waves * num_harmonics, num_waves * num_harmonics)
```

**Attention Mechanism**:
```python
# Phase evolution: θ(t) = ω*t + φ_0
theta = freq * positions + phase

# Create phasors: A * e^(iθ)
phasor = amp * torch.exp(1j * theta)

# Wave interference: Re(Q · K*)
interference_term = torch.matmul(phasor, phasor.conj().transpose(-2, -1)).real

# Full intensity: I = A_Q² + A_K² + 2*A_Q*A_K*cos(Δφ)
intensity = amp_q_sq + amp_k_sq + 2 * interference_term

# Physics-based normalization (NOT softmax!)
transmission = intensity / max_energy

# Row normalization (NOT softmax!)
attn_weights = scores / row_sum
```

**✅ PHYSICS-BASED**: Uses wave interference formula, NOT dot product attention.
**✅ NO SOFTMAX**: Uses row normalization, not softmax competition.

---

### 5. `WaveResonanceMLP` (Hybrid MLP)

**Location**: `wave_gpt.py:980`

**Purpose**: MLP with wave-inspired activation

**Structure**:
```python
def forward(self, x):
    h = self.fc1(x)  # x is embedding (B, T, d_model)
    h = F.gelu(h) + 0.1 * torch.sin(h)  # GELU + sin for wave character
    h = self.fc2(h)
    return h
```

**⚠️ HYBRID**: Operates on embedding vectors, not wave parameters. Uses GELU (standard) + sin (wave).

---

### 6. `CollapseHead` (Embedding → Logits)

**Location**: `wave_gpt.py:1052`

**Purpose**: Project embeddings to vocabulary

**Structure**:
```python
class CollapseHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        self.ln = nn.LayerNorm(d_model)  # Standard LayerNorm!
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.ln(x)  # LayerNorm on embeddings
        logits = self.proj(x)
        return logits
```

**⚠️ HYBRID**: Uses `nn.LayerNorm` on embedding vectors.

---

## Key Differences: interference_attention vs pure_wave

| Aspect | interference_attention | pure_wave |
|--------|------------------------|-----------|
| **Internal Representation** | Embedding vectors `(B, T, d_model)` | Wave parameters `(freqs, phases, amps)` |
| **Embedding Layer** | `WavePacketEmbedding` → projects to `d_model` | `PureWaveExcitation` → returns wave params |
| **Normalization** | `nn.LayerNorm(d_model)` on embeddings | `WaveRMSNorm` on wave params |
| **MLP** | `WaveResonanceMLP` on embeddings | `PureWaveMLP` on wave params |
| **Output** | `CollapseHead` with LayerNorm | `WaveCollapse` with wave normalization |
| **Residual Connections** | In embedding space | In wave parameter space |

---

## What's WAVE in interference_attention

✅ **Wave Embeddings**: `WavePacketEmbedding` uses physics-based wave parameters
✅ **Wave Attention**: `InterferenceAttention` uses interference formula `I = A_Q² + A_K² + 2*A_Q*A_K*cos(Δφ)`
✅ **No Softmax**: Uses physics-based energy normalization
✅ **Wave MLP**: `WaveResonanceMLP` adds `sin()` to activation

---

## What's STANDARD in interference_attention

⚠️ **Embedding Space**: Operates in `d_model` dimension, not wave parameters
⚠️ **LayerNorm**: Uses `nn.LayerNorm` on embedding vectors
⚠️ **Projection**: `wave_to_embed` projects waves to embedding space
⚠️ **Residuals**: Residual connections in embedding space

---

## Conclusion

**interference_attention is a HYBRID architecture:**

```
Token → Wave Parameters → PROJECTION → Embedding Space → Interference Attention → Embedding Space → Logits
                         ↑
                    This projection
                    makes it hybrid!
```

**pure_wave is PURE wave:**

```
Token → Wave Parameters → Wave Attention → Wave Parameters → Wave MLP → Wave Parameters → Logits
                         ↑                                                              ↑
                    No projection!                                               Direct collapse!
```

### Summary Table

| Experiment | Embedding | Attention | Internal Space | Purity |
|------------|-----------|-----------|----------------|--------|
| `standard_transformer` | `nn.Embedding` | `Q @ K.T` + softmax | Embedding | 0% wave |
| `interference_attention` | `WavePacketEmbedding` | Wave interference | Embedding | ~50% wave |
| `pure_wave` | `PureWaveExcitation` | Wave interference | Wave params | 100% wave |

**The key test**: Does `pure_wave` (100% wave) outperform `interference_attention` (50% wave)?

If yes, it proves that **pure wave-to-wave computation** is superior to hybrid approaches.
