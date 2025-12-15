# 🌊 Pure Wave GPT Expressivity Enhancements

## Goal: Achieve ~4.0 Loss with Enhanced Wave Physics

Current loss: ~7.4 → Target: ~4.0

## Core Principle: EVERY Token Gets FULL Spectrum

**CRITICAL**: Every token (Token 0, Token 50256, and everything in between) has:
- **ALL frequencies** (sentence-level to phoneme-level)
- **ALL harmonics** (1f, 2f, 3f, 4f...)
- **ALL learnable parameters**

The ONLY difference between tokens is their **learned wave parameters**.
Position encoding comes ONLY from temporal phase evolution: `φ(t) = ω*t + φ₀`

## Wave Parameter Structure

```python
# For vocab_size=50257, num_waves=48, num_harmonics=4:

base_freqs:   (50257, 48)      # Each token has 48 frequencies spanning ALL scales
phases:       (50257, 48)      # Each token has 48 initial phases  
amplitudes:   (50257, 48, 4)   # Each token has 48×4=192 harmonic amplitudes

# Total learnable wave parameters per token: 48 + 48 + 192 = 288
# Total learnable wave parameters: 50257 × 288 = 14.5M parameters
```

## Frequency Spectrum (Log-Spaced, ALL Scales)

Every token's 48 waves span the FULL linguistic spectrum:

```
Wave 0:  0.005 Hz → Period ~200 tokens (sentence-level)
Wave 12: 0.02 Hz  → Period ~50 tokens (phrase-level)
Wave 24: 0.1 Hz   → Period ~10 tokens (word-level)
Wave 36: 0.4 Hz   → Period ~2.5 tokens (morpheme-level)
Wave 47: 2.0 Hz   → Period ~0.5 tokens (phoneme-level)
```

**Key**: Token 0 ("the") and Token 50256 (rare word) BOTH have all these frequencies!

## Enhancement Strategies Implemented

### 1. **Full Spectrum for Every Token** 🎵
**Problem**: Previous implementation gave different tokens different frequency subsets
**Solution**: Log-spaced frequencies covering ALL scales for EVERY token

```python
# Log-spaced from 0.005 Hz to 2.0 Hz
base_spectrum = torch.logspace(log10(0.005), log10(2.0), num_waves)

# EVERY token gets this FULL spectrum
init_freqs = base_spectrum.unsqueeze(0).expand(vocab_size, -1)
```

**Benefits**:
- Every token can capture patterns at ANY scale
- Model learns which frequencies matter for each token
- No artificial constraints on token expressivity

### 2. **Cross-Wave Coupling** 🔗
**Problem**: Waves operate independently
**Solution**: Learnable coupling matrix between waves

```python
self.wave_coupling = nn.Parameter(torch.eye(num_waves) + randn * 0.05)
coupled_freqs = torch.matmul(token_freqs, self.wave_coupling)
```

**Benefits**:
- Low-frequency waves can influence high-frequency waves
- Captures hierarchical linguistic interactions
- Emergent cross-scale patterns

### 3. **Harmonic Coupling** 🎼
**Problem**: Harmonics (1f, 2f, 3f, 4f) operate independently
**Solution**: Learnable coupling between harmonics

```python
self.harmonic_coupling = nn.Parameter(torch.eye(num_harmonics) + randn * 0.05)
coupled_amps = torch.matmul(token_amps, self.harmonic_coupling)
```

**Benefits**:
- Fundamental can influence overtones
- Captures natural harmonic relationships
- Richer wave shapes emerge

### 4. **Pure Temporal Position Encoding** ⏰
**Problem**: Need position information without breaking wave physics
**Solution**: Phase evolution IS the position encoding

```python
# Position in sequence = Time
positions = torch.arange(T)

# Phase evolves: φ(t) = ω*t + φ₀
evolved_phases = coupled_freqs * positions + token_phases
```

**Benefits**:
- No separate positional embeddings needed
- Position naturally encoded in wave physics
- Relative positions from phase differences

### 5. **Wave Superposition Projections** 📐
**Problem**: Matrix projections are not wave physics
**Solution**: PROJECT waves by ADDING other learnable waves!

```python
# Instead of: projected = input @ matrix
# We do: projected = input_wave + projection_wave

# Learnable projection waves (full wave parameters!)
emit_proj_freqs = nn.Parameter(...)   # (num_channels, num_waves)
emit_proj_phases = nn.Parameter(...)  # (num_channels, num_waves)
emit_proj_amps = nn.Parameter(...)    # (num_channels, num_waves, num_harmonics)

# Projection = Wave Superposition!
emitter_wave = input_wave + mix * projection_wave
```

**This is INFINITELY expressive because**:
- Projection wave has its own frequencies, phases, amplitudes
- Superposition creates beating patterns
- Phase differences create interference
- Frequency mixing creates harmonics
- Everything emerges from wave physics!

### 6. **Emitter-Receiver-Field Model** 🌊
**Problem**: Q/K/V is attention terminology, not physics
**Solution**: Pure wave field model

```python
# Each position is an EMITTER and RECEIVER
emitter_wave = input + emit_projection_wave
receiver_wave = input + recv_projection_wave
field_wave = input + field_projection_wave

# Interference determines coupling (not learned attention!)
interference = Re(emitter_phasor · receiver_phasor*)

# Coupling emerges from PHYSICS
coupling = intensity / max_intensity  # Transmission coefficient
```

**Key insight**: "Attention" is just wave interference!
- Constructive interference → strong coupling
- Destructive interference → weak coupling
- It's PHYSICS, not learned weights!

### 6. **Enhanced Wave Collapse** 🎯
**Problem**: Simple projection loses wave structure
**Solution**: Multi-stage processing with attention

**Components**:
- Separate processors for freqs, phases, amps
- Self-attention over wave parameters
- Harmonic content analysis
- Multi-layer projection to vocabulary

### 7. **Wave Physics Regularization** ⚖️
**Problem**: Unconstrained waves can become chaotic
**Solution**: Physics-based regularization

```python
# Regularization terms:
freq_stability = mean((final_freqs - initial_freqs)²) * 0.01
phase_coherence = mean(sin(phase_diff)²) * 0.005
amp_conservation = mean((final_amp_mean - initial_amp_mean)²) * 0.01
```

## What Makes This Different

### Before (Wrong):
```
Token 0: waves 0-10 (low freq only)
Token 1: waves 5-15 (mid freq only)
Token 2: waves 10-20 (high freq only)
```

### After (Correct):
```
Token 0: waves 0-47 (ALL frequencies, ALL harmonics) - LEARNABLE
Token 1: waves 0-47 (ALL frequencies, ALL harmonics) - LEARNABLE
Token 2: waves 0-47 (ALL frequencies, ALL harmonics) - LEARNABLE
...
Token 50256: waves 0-47 (ALL frequencies, ALL harmonics) - LEARNABLE
```

## Expected Loss Progression

- **Current**: ~7.4 (basic pure wave)
- **With full spectrum**: ~6.0 (every token has all scales)
- **With coupling**: ~5.0 (wave and harmonic interactions)
- **With enhanced collapse**: ~4.0 (better wave→logits projection)

## Key Physics Principles

✅ **Every token = Full oscillator system** (all frequencies, all harmonics)
✅ **Position = Time** (phase evolution φ(t) = ωt + φ₀)
✅ **Attention = Interference** (I = A₁² + A₂² + 2A₁A₂cos(Δφ))
✅ **Everything learnable** (frequencies, phases, amplitudes, couplings)
✅ **Discrete↔Continuous bridge** (tokens excite continuous waves)

## Training Configuration

```python
"pure_wave": ExperimentConfig(
    model_type="pure_wave",
    lr=5e-4,           # Higher LR for expressivity
    dropout=0.05,      # Lower dropout to preserve wave info
    warmup_steps=2000, # Longer warmup for complex dynamics
)
```

The model now has the full expressivity needed to learn the discrete-continuous mapping! 🌊