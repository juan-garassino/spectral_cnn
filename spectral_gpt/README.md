# 🌊 Wave-Native GPT

> "Everything in physics is a mass on a spring" — Wave-based language modeling

## Philosophy

Standard transformers fight against wave-based computation by using discrete token embeddings. **Wave-Native GPT** makes language itself continuous:

```
Token → Wave Packet → Interference → Superposition → Collapse → Next Token
```

## Architecture

### 1. Wave Packet Embedding
Tokens are embedded as wave packets, not vectors:
- **Frequency**: What "pitch" does this token resonate at?
- **Phase**: Where in the wave cycle does this token start?
- **Amplitude**: How strong is each wave component?

### 2. Wave Interference Attention
Attention via constructive/destructive interference:
- Waves in phase → amplify (high attention)
- Waves out of phase → cancel (low attention)

### 3. Wave Collapse Head
Like quantum measurement: continuous wave state "collapses" to discrete token probabilities.

## Files

| File | Description |
|------|-------------|
| `wave_gpt.py` | Wave-Native GPT model architecture |
| `wave_benchmark.py` | Benchmark + visualization suite |
| `benchmark_results/` | Training outputs and plots |
| `prototyping/` | Legacy spectral transformer experiments |

## Quick Start

```bash
# Run benchmark on Colab (GPU recommended)
python spectral_gpt/wave_benchmark.py
```

## Benchmark Results (5M params, 5000 steps)

| Model | Perplexity | Speed | Gap |
|-------|------------|-------|-----|
| Classic Transformer | ~25 | 94K tok/s | baseline |
| **Wave-Native GPT** 🌊 | ~63 | 76K tok/s | 2.5x PPL |

Wave-Native achieves **81% of Classic's speed** while learning continuous representations!

## Visualizations

The benchmark saves interpretability plots to `benchmark_results/wave_gpt_plots/`:

- 📈 Learning curves (raw + smoothed)
- 🎵 Token frequency distributions
- 🌀 Token phase heatmaps
- 🌊 Wave packet visualizations per token
- 🎯 Attention phase shifts
- ⚔️ Comparison plots

## Key Innovations

| Component | Standard GPT | Wave-Native GPT |
|-----------|--------------|-----------------|
| Embedding | Lookup table | Wave packets |
| Representation | d_model vector | (freq, phase, amp) |
| Attention | Dot product | Wave interference |
| Activation | GELU/ReLU | sin(x) + 0.1x |
| Output | Linear | Wave collapse |

## Future Directions

1. **Holographic memory**: Full wave interference as associative memory
2. **Diffusion + Waves**: Denoising in wave space
3. **Complex-valued**: Use ℂ instead of ℝ for true wave computation
4. **Resonance learning**: Let tokens "resonate" with each other

---

*Part of the Spectral Neural Networks research project*
