# Wave-Native GPT Experiments

## Components Under Test

| Component | Options | Description |
|-----------|---------|-------------|
| **Embeddings** | Standard / Wave | Standard = lookup table, Wave = physics-based (ω₀ = 1/√Mass) |
| **Attention** | Hybrid / Interference | Hybrid = softmax, Interference = energy normalization (I = A²+A²+2AA·cos(Δφ)) |
| **Optimizer** | AdamW / WaveOpt | AdamW = standard, WaveOpt = parameter-specific masses + resonance damping |
| **Loss** | CE / QFE | CE = CrossEntropy, QFE = + phase/energy/harmonic regularization |

---

## Experiment Matrix

| Experiment | Embed | Attention | Optimizer | Loss | What It Tests |
|------------|-------|-----------|-----------|------|---------------|
| `standard_transformer` | Std | Hybrid | AdamW | CE | **CONTROL** - Pure GPT-2 baseline |
| `wave_baseline` | Wave | Hybrid | AdamW | CE | Do wave embeddings help? |
| `interference_attention` | Wave | **Interference** | AdamW | CE | Does physics attention help? |
| `rgd_only` | Wave | Hybrid | WaveOpt | CE | Does physics optimizer help? |
| `qfe_only` | Wave | Hybrid | AdamW | QFE | Does coherence loss help? |
| `full_physics` | Wave | Hybrid | WaveOpt | QFE | Full wave stack (no interference attn) |
| `interference_full` | Wave | **Interference** | WaveOpt | QFE | **FULL PHYSICS** - Everything wave |

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

---

## Results

| Experiment | Val Loss | Perplexity | Speed | Quality |
|------------|----------|------------|-------|---------|
| `standard_transformer` | 4.35 | 97 | 12,357 | ✅ Coherent |
| `wave_baseline` | 5.18 | 190 | 10,713 | ✅ Coherent |
| `interference_attention` | ? | ? | ? | ? |
| `rgd_only` | ? | ? | ? | ? |
| `qfe_only` | ? | ? | ? | ? |
| `full_physics` | 7.86 | 2517 | 4,442 | ❌ Gibberish |
| `interference_full` | ? | ? | ? | ? |

---

## Commands

```bash
# Run single experiment
python wave_experiments.py --experiment interference_attention --dataset fineweb --steps 10000

# Run comparison pair
python wave_experiments.py --experiment standard_transformer wave_baseline --dataset fineweb --steps 10000

# Run all core experiments
python wave_experiments.py --experiment standard_transformer wave_baseline interference_attention rgd_only qfe_only full_physics --dataset fineweb --steps 10000
```

---

## Conclusions (To Fill In)

### What Works
- [ ] Wave embeddings (wave_baseline competitive with standard_transformer?)
- [ ] Interference attention (interference_attention beats wave_baseline?)
- [ ] Physics optimizer (rgd_only beats wave_baseline?)
- [ ] Coherence loss (qfe_only beats wave_baseline?)

### What Doesn't Work
- [ ] (To be determined)

### Best Configuration
- [ ] (To be determined from experiments)
