# Wave-Native GPT Experiments

## Components Under Test

| Component | Options | Description |
|-----------|---------|-------------|
| **Embeddings** | Standard / Wave / **Pure Wave** | Standard = lookup, Wave = physics-based, **Pure Wave = DC+AC modes** |
| **Attention** | Standard / Hybrid / Interference / **Multi-Head Wave** | Standard = GPT-2, **Multi-Head Wave = per-head full spectrum** |
| **Optimizer** | AdamW / WaveOpt | AdamW = standard, WaveOpt = parameter-specific masses |
| **Loss** | CE / QFE | CE = CrossEntropy, QFE = + phase/energy regularization |
| **Trinity** | Off / On | **NEW**: Saturation + FM + Interferometer gates |
| **DC Modes** | Off / On | **NEW**: Zero-frequency waves for global attention |

---

## 🔴 NEW: Diagonal Blindness Experiments

### The Problem
Standard phase evolution `θ = ω*t + φ` causes **diagonal attention patterns**:
- Phase difference `Δθ = ω*(i-j)` grows with distance
- Distant tokens → rapid phase rotation → destructive interference
- Result: Model can ONLY attend locally

### The Solution: DC Modes + Trinity

| Experiment | DC Modes | Trinity | Description |
|------------|----------|---------|-------------|
| `diagonal_baseline` | ❌ | ❌ | Baseline (shows diagonal blindness) |
| `diagonal_dc` | ✅ | ❌ | DC modes only |
| `diagonal_trinity` | ❌ | ✅ | Trinity only |
| `diagonal_full` | ✅ | ✅ | **DC + Trinity** (recommended) |

### DC Modes Physics
```
Standard: θ = ω*t + φ → Δθ = ω*(i-j) → position-dependent
DC Mode:  θ = 0*t + φ → Δθ = φ_i - φ_j → content-dependent only!
```

### Expected Results
- `diagonal_baseline`: Perfect diagonal attention, loss ~6.5 (unigram)
- `diagonal_dc`: Off-diagonal attention via DC modes
- `diagonal_trinity`: Sharper logic, better mixing
- `diagonal_full`: Global attention + sharp logic

---

## 🌊 Pure Wave Architecture (Updated)

### Architecture Evolution

```
v1: Token → Wave → WaveAttn → Wave → WaveMLP → Wave → Logits
                    ↓
v2: Token → Wave(DC+AC) → MultiHead-Interference → [Gate] → Wave → [Saturation]MLP → [Gate] → Wave → Logits
```

### Key Components

| Component | v1 | v2 (Current) |
|-----------|----|----|
| Wave Excitation | AC only | **DC + AC modes** |
| Attention | Single-head | **Multi-head (full spectrum per head)** |
| MLP | Linear | **4x expansion + Saturation** |
| Residuals | Standard | **Phase Interferometer Gates** |
| Context | None | **FM Modulation** |

### Wave State Representation
```python
WaveState = {
    freqs: (B, T, num_waves),           # DC modes = 0, AC modes > 0
    phases: (B, T, num_waves),          # DC: semantic content, AC: position
    amps: (B, T, num_waves, harmonics)  # Harmonic amplitudes
}
```

---

## 🎯 The Non-Linear Physics Trinity

### 1. WaveSaturation (Overdrive)
```python
saturated = tanh(gain * (x + bias))
# gain > 1 → square-wave logic states
```
**Purpose**: Creates sharp 0/1 decisions from smooth waves

### 2. WaveFM (Frequency Modulation)
```python
ω_new = ω_old + mod_index * context_signal
# Past tokens shift future token frequencies
```
**Purpose**: Context physically alters meaning

### 3. PhaseInterferometer (Gated Residuals)
```python
gate = sigmoid(sharpness * cos(φ_input - φ_control))
out = input + gate * delta
# φ_control ≈ 0 → PASS, φ_control ≈ π → BLOCK
```
**Purpose**: Logic NOT via destructive interference

---

## 🚀 Multi-Head Wave Interference

### Key Principle: All Heads Learn All Frequencies

```python
# Every head starts with FULL spectrum (0.005 - 2.0 Hz)
# No artificial frequency constraints!
# Heads specialize through learning

def _init_projection_freqs(num_heads, num_waves):
    base_spectrum = logspace(0.005, 2.0, num_waves)
    freqs = base_spectrum.expand(num_heads, -1)
    freqs *= (1 + randn(num_heads, num_waves) * 0.05)  # Break symmetry
    return freqs
```

### Per-Head Components
- Emitter projection waves (freqs, phases, amps)
- Receiver projection waves (freqs, phases, amps)
- Field projection waves (freqs, phases, amps)
- Output projection waves (freqs, phases, amps)
- Interference strength (per head)
- Learned combination weights (softmax)

---

## Experiment Matrix (Updated)

| Experiment | Embed | Attention | Trinity | DC Modes | What It Tests |
|------------|-------|-----------|---------|----------|---------------|
| `standard_transformer` | Std | Standard | ❌ | ❌ | **CONTROL** |
| `wave_baseline` | Wave | Hybrid | ❌ | ❌ | Wave embeddings |
| `interference_attention` | Wave | Interference | ❌ | ❌ | Physics attention |
| `pure_wave` | Pure | Multi-Head | ❌ | ❌ | Pure wave-to-wave |
| `pure_wave_trinity` | Pure | Multi-Head | ✅ | ❌ | + Non-linear physics |
| `pure_wave_dc` | Pure | Multi-Head | ❌ | ✅ | + DC modes |
| **`pure_wave_full`** | Pure | Multi-Head | ✅ | ✅ | **FULL PHYSICS** |

---

## Key Comparisons (Updated)

### 1. DC Modes Effect (Diagonal Blindness)
```
pure_wave  vs  pure_wave_dc
 (AC only)     (DC + AC)
```
**Question:** Do DC modes break diagonal attention patterns?

### 2. Trinity Effect (Sharp Logic)
```
pure_wave  vs  pure_wave_trinity
 (Linear)      (Saturation + FM + Gates)
```
**Question:** Does the Trinity enable sharper reasoning?

### 3. Multi-Head Effect
```
single_head_wave  vs  multi_head_wave
   (1 head)           (8 heads, full spectrum each)
```
**Question:** Do multiple heads with full spectrum improve?

### 4. Full Stack vs Baseline
```
standard_transformer  vs  pure_wave_full
      (GPT-2)              (DC + Trinity + Multi-Head)
```
**Question:** Can pure wave physics match transformers?

---

## Results (Updated)

| Experiment | Val Loss | Perplexity | Diagonal? | Notes |
|------------|----------|------------|-----------|-------|
| `standard_transformer` | 4.35 | 97 | N/A | Control |
| `wave_baseline` | 5.18 | 190 | N/A | Hybrid |
| `pure_wave` | 6.51 | 669 | ✅ Yes | Broke unigram! But diagonal |
| `pure_wave_trinity` | ? | ? | ? | Pending |
| `pure_wave_dc` | ? | ? | ? | Pending |
| **`pure_wave_full`** | ? | ? | ? | **Next experiment** |

### Diagonal Blindness Status
- **Loss 6.51 < 7.6 (unigram)**: Model IS learning beyond frequency statistics
- **Attention plots**: Still show diagonal patterns
- **Hypothesis**: DC modes will enable off-diagonal attention

---

## Commands (Updated)

```bash
# Run diagonal blindness experiments
python wave_experiments.py --experiment diagonal_baseline --dataset fineweb --steps 5000
python wave_experiments.py --experiment diagonal_dc --dataset fineweb --steps 5000
python wave_experiments.py --experiment diagonal_trinity --dataset fineweb --steps 5000
python wave_experiments.py --experiment diagonal_full --dataset fineweb --steps 5000

# Run Pure Wave with all features
python wave_experiments.py --experiment pure_wave_full --dataset fineweb --steps 10000

# Compare DC modes effect
python wave_experiments.py --experiment pure_wave pure_wave_dc --dataset fineweb --steps 5000

# Compare Trinity effect
python wave_experiments.py --experiment pure_wave pure_wave_trinity --dataset fineweb --steps 5000

# Full comparison
python wave_experiments.py --experiment standard_transformer pure_wave pure_wave_full --dataset fineweb --steps 10000
```

---

## Training Logs (Updated)

### PureWaveGPT Log Format
```
Step 100 | Loss 8.234 | LR 0.00030 | Gain 1.82 | FM 0.45 | Gate 0.67 | 9,707 tok/s
         │            │            │           │         │
         │            │            │           │         └─ Interferometer sharpness
         │            │            │           └─ FM modulation index
         │            │            └─ Saturation gain (>1 = sharp)
         │            └─ Learning rate
         └─ Cross-entropy loss
```

### Standard WaveGPT Log Format
```
Step 100 | Loss 8.234 | LR 0.00030 | R 0.85 | 10,234 tok/s
                                    │
                                    └─ Annealing ratio (wave ↔ standard)
```

---

## Conclusions (To Fill In)

### What Works
- [x] Pure wave architecture (loss 6.51 < unigram 7.6)
- [x] Multi-head wave interference
- [ ] DC modes for global attention (pending)
- [ ] Trinity for sharp logic (pending)
- [ ] Full stack competitive with transformers (pending)

### What Doesn't Work (Yet)
- [x] Diagonal blindness persists without DC modes
- [ ] (More to be determined)

### Best Configuration (Hypothesis)
```
pure_wave_full = DC Modes + Trinity + Multi-Head + 4x MLP
```

### Key Metrics to Watch
1. **Loss**: Should be < 6.5 (current best)
2. **Attention plots**: Should show off-diagonal patterns with DC modes
3. **Generation quality**: Should be coherent, not repetitive
4. **Trinity metrics**: Gain > 1.5, FM > 0.3, Gate variance > 0.1

---

## Theoretical Predictions

### DC Modes Should Enable:
1. **Global attention**: Token 0 can attend to Token 5000 if phases align
2. **Content-based routing**: Attention based on semantic similarity, not position
3. **Breaking diagonal**: Off-diagonal patterns in attention visualization

### Trinity Should Enable:
1. **Sharp decisions**: Binary-like outputs from saturation
2. **Context sensitivity**: FM makes meaning position-dependent
3. **Information gating**: Interferometer can block irrelevant info

### Multi-Head Should Enable:
1. **Diverse patterns**: Each head learns different attention patterns
2. **Full spectrum**: No artificial frequency constraints
3. **Learned combination**: Softmax weights for head mixing
