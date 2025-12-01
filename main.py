import os
import torch
import numpy as np
from datetime import datetime
from src.training.trainer import train_fit
from src.visualization.plotter import plot_results, plot_layer_waves

# Configuration
NUM_EPOCHS = 3
NUM_WAVES = 12
MODES = ["Standard", "UserWave", "Poly", "Wavelet", "Factor", "Siren", "GatedWave"]

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"./WeightParam_Benchmark_{timestamp}/"
    os.makedirs(base_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    print(f"STARTING 5-FIT BENCHMARK V14 (The Optimized Duel)")
    print(f"Target Baseline: ~9,700 params")

    for m in MODES:
        results[m] = train_fit(m, NUM_WAVES, NUM_EPOCHS, device)

    print("\n" + "="*125)
    print(f"FINAL LEADERBOARD ({NUM_EPOCHS} Epochs) - V14 RESULTS")
    print("="*125)
    print(f"| {'Model':<15} | {'Params':<10} | {'Comp Ratio':<10} | {'Train %':<8} | {'Test %':<8} | {'Gen Gap':<8} | {'Eff. Score':<10} |")
    print("-" * 125)

    baseline_params = results["Standard"]["params"]
    for m in MODES:
        r = results[m]
        comp_ratio = baseline_params / r['params'] if r['params'] > 0 else 0
        gen_gap = r['train_acc'] - r['test_acc']
        eff_score = (r['test_acc'] / np.log10(r['params'])) if r['params'] > 0 else 0
        print(f"| {m:<15} | {r['params']:<10,} | {comp_ratio:<9.1f}x | {r['train_acc']:<8.2f} | {r['test_acc']:<8.2f} | {gen_gap:<8.2f} | {eff_score:<10.2f} |")
    print("-" * 125)

    # Visualization
    plot_results(results, MODES, base_dir, NUM_EPOCHS)
    
    # Wave Visualization
    print("\n[Generating Wave Plots...]")
    for m in MODES:
        plot_layer_waves(results[m]['model'], m, base_dir)

    print(f"All results and plots saved to {base_dir}")

if __name__ == "__main__":
    main()
