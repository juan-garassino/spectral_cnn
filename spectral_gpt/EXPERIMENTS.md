# Wave-Native GPT Experiments

## Components Under Test

| Component | Options | Description |
|-----------|---------|-------------|
| **Embeddings** | Standard / Wave | Standard = lookup table, Wave = physics-based (ω₀ = 1/√Mass) |
| **Attention** | Standard / Hybrid / Interference | Standard = vanilla GPT-2 (Q·K softmax), Hybrid = wave interference + softmax, Interference = phasor-based energy normalization (I = A²+A²+2AA·cos(Δφ), O(T²) memory) |
| **Optimizer** | AdamW / WaveOpt | AdamW = standard, WaveOpt = parameter-specific masses + resonance damping |
| **Loss** | CE / QFE | CE = CrossEntropy, QFE = + phase/energy/harmonic regularization |

---

## How the Grid Works

Each experiment is **one point** in the grid. Running all experiments together forms a complete ablation study.

```
                        ┌─────────────────────────────────────────────────┐
                        │              EXPERIMENT GRID                     │
                        │         (Each cell = 1 experiment)               │
                        └─────────────────────────────────────────────────┘

                                      Attention
                          Standard        Hybrid          Interference
                        (GPT-2 Q·K)    (wave+softmax)   (energy norm)
                       ┌────────────┬──────────────┬──────────────────┐
          Standard     │ standard_  │     N/A      │       N/A        │
Embeddings (lookup)    │ transformer│              │                  │
                       ├────────────┼──────────────┼──────────────────┤
          Wave         │    N/A     │ wave_        │ interference_    │
          (physics)    │            │ baseline     │ attention        │
                       └────────────┴──────────────┴──────────────────┘

                                      Optimizer
                              AdamW           WaveOpt
                           ┌──────────────┬──────────────────┐
              CE Loss      │ wave_        │ rgd_only         │
Loss                       │ baseline     │                  │
                           ├──────────────┼──────────────────┤
              QFE Loss     │ qfe_only     │ full_physics     │
                           │              │                  │
                           └──────────────┴──────────────────┘
```

**To run the full grid:** Run each experiment individually. Together they form the complete ablation study.

---

## Experiment Matrix

| Experiment | Embed | Attention | Optimizer | Loss | What It Tests |
|------------|-------|-----------|-----------|------|---------------|
| `standard_transformer` | Std | **Standard (GPT-2)** | AdamW | CE | **CONTROL** - Pure GPT-2 baseline |
| `wave_baseline` | Wave | Hybrid | AdamW | CE | Do wave embeddings help? |
| `interference_attention` | Wave | **Interference** | AdamW | CE | Does physics attention help? |
| `rgd_only` | Wave | Hybrid | WaveOpt | CE | Does physics optimizer help? |
| `qfe_only` | Wave | Hybrid | AdamW | QFE | Does coherence loss help? |
| `full_physics` | Wave | Hybrid | WaveOpt | QFE | Full wave stack (hybrid attention) |
| `interference_full` | Wave | **Interference** | WaveOpt | QFE | **FULL PHYSICS** - Everything wave |
| `pure_wave` | **Pure Wave** | **Pure Wave** | AdamW | CE | **🌊 PURE WAVE GPT** - Wave-to-wave throughout |
| `pure_wave_full` | **Pure Wave** | **Pure Wave** | WaveOpt | QFE | **🌊 PURE WAVE + FULL PHYSICS** |

---

## 🌊 Pure Wave Architecture

The `pure_wave` experiments test a completely new architecture:

### **Traditional Transformer:**
```
Token → Embedding → Attention → Embedding → MLP → Embedding → Logits
```

### **Hybrid Wave (interference_attention):**
```
Token → Wave → Embedding → InterferenceAttn → Embedding → Wave → Logits
```

### **🌊 Pure Wave GPT:**
```
Token → Wave → WaveAttn → Wave → WaveMLP → Wave → Logits
```

**Key Differences:**
- **No embedding space anywhere** - pure wave-to-wave computation
- **Emergent attention patterns** from learned wave interference
- **Learnable decay/re-emergence** from frequency beating
- **All patterns emerge from physics** - no hardcoded structures

### **Wave State Representation:**
```python
WaveState = {
    freqs: (B, T, num_waves),           # Learned frequencies per token
    phases: (B, T, num_waves),          # Learned phases per token  
    amps: (B, T, num_waves, harmonics) # Learned harmonic amplitudes
}
```

### **How Attention Emerges:**
1. **Each token learns frequencies** via `freq_proj(wave_state)`
2. **Phase evolution:** `θ(t) = ω*t + φ₀` (position = time)
3. **Wave interference:** `I = A_Q² + A_K² + 2*A_Q*A_K*cos(Δφ)`
4. **Beating patterns:** When `f₁ ≠ f₂`, attention oscillates with period `1/|f₁-f₂|`
5. **Decay/re-emergence:** Network learns frequency differences for linguistic patterns

---

## Key Comparisons

### 1. Wave Embeddings Effect
```
standard_transformer  vs  wave_baseline
       (Std)                 (Wave)
```
**Question:** Do physics-based embeddings improve language modeling?

### 2. Interference Attention Effect  
```
wave_baseline  vs  interference_attention
   (Hybrid)          (Interference)
```
**Question:** Does the physics attention formula beat softmax?

### 3. Physics Optimizer Effect
```
wave_baseline  vs  rgd_only
   (AdamW)         (WaveOpt)
```
**Question:** Does parameter-specific mass help optimization?

### 4. Coherence Loss Effect
```
wave_baseline  vs  qfe_only
    (CE)            (QFE)
```
**Question:** Does phase/energy regularization improve coherence?

### 5. Full Stack vs Baseline
```
standard_transformer  vs  interference_full
      (Nothing)            (Everything)
```
**Question:** Is the complete wave-native approach better?

### 6. 🌊 Pure Wave vs Hybrid Wave
```
interference_attention  vs  pure_wave
    (Hybrid Wave)         (Pure Wave)
```
**Question:** Does pure wave-to-wave beat hybrid wave-embedding-wave?

### 7. 🌊 Pure Wave vs Standard
```
standard_transformer  vs  pure_wave
    (Standard GPT)       (Pure Wave)
```
**Question:** Can pure wave physics match standard transformers?

### 8. 🌊 Emergent Physics Test
```
pure_wave  vs  pure_wave_full
 (AdamW)      (WaveOpt+QFE)
```
**Question:** Do emergent wave patterns improve with physics optimization?

---

## Results

| Experiment | Val Loss | Perplexity | Speed | Quality |
|------------|----------|------------|-------|---------|
| `standard_transformer` | 4.35 | 97 | 12,357 | ✅ Coherent |
| `wave_baseline` | 5.18 | 190 | 10,713 | ✅ Coherent |
| `interference_attention` | ? | ? | ? | ✅ Ready (memory-optimized) |
| `rgd_only` | ? | ? | ? | ? |
| `qfe_only` | ? | ? | ? | ? |
| `full_physics` | 7.86 | 2517 | 4,442 | ❌ Gibberish |
| `interference_full` | ? | ? | ? | ? |
| **`pure_wave`** | **?** | **?** | **?** | **🌊 PURE WAVE TEST** |
| **`pure_wave_full`** | **?** | **?** | **?** | **🌊 PURE WAVE + PHYSICS** |

---

## Commands

```bash
# Run single experiment
python wave_experiments.py --experiment interference_attention --dataset fineweb --steps 10000

# Run comparison pair
python wave_experiments.py --experiment standard_transformer wave_baseline --dataset fineweb --steps 10000

# Run all core experiments (the full grid)
python wave_experiments.py --experiment standard_transformer wave_baseline interference_attention rgd_only qfe_only full_physics interference_full --dataset fineweb --steps 10000

# 🌊 Run Pure Wave experiments
python wave_experiments.py --experiment pure_wave --dataset fineweb --steps 10000
python wave_experiments.py --experiment pure_wave_full --dataset fineweb --steps 10000

# Compare Pure Wave vs others
python wave_experiments.py --experiment standard_transformer interference_attention pure_wave --dataset fineweb --steps 10000
```

---

## Conclusions (To Fill In)

### What Works
- [ ] Wave embeddings (wave_baseline competitive with standard_transformer?)
- [ ] Interference attention (interference_attention beats wave_baseline?)
- [ ] Physics optimizer (rgd_only beats wave_baseline?)
- [ ] Coherence loss (qfe_only beats wave_baseline?)
- [ ] **🌊 Pure wave architecture (pure_wave competitive with standard_transformer?)**
- [ ] **🌊 Emergent wave patterns (pure_wave shows learnable decay/re-emergence?)**

### What Doesn't Work
- [ ] (To be determined)

### Best Configuration
- [ ] (To be determined from experiments)

### 🌊 Pure Wave Hypothesis
**If pure wave works, we should see:**
1. **Frequency specialization:** Function words → low freq, content words → high freq
2. **Emergent beating:** Attention patterns with periodic structure
3. **Learned dependencies:** Beat periods matching linguistic distances
4. **No hardcoded patterns:** All structure emerges from learned wave physics

**Key test:** Does `pure_wave` match `standard_transformer` performance while showing emergent wave physics?
