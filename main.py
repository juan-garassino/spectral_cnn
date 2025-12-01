import os
import torch
import numpy as np
from datetime import datetime
from src.training.trainer import train_fit
from src.visualization.plotter import plot_results, plot_layer_waves, plot_wave_decomposition

# Configuration
NUM_EPOCHS = 3
NUM_WAVES = 12
MODES = ["Standard", "UserWave", "Poly", "Wavelet", "Factor", "Siren", "GatedWave"]

# Fourier Configuration (for UserWave and GatedWave)
NUM_HARMONICS = 3          # Number of Fourier components per wave (3 = cos(θ), cos(2θ), cos(4θ))
ADAPTIVE_FREQS = False     # If True, harmonic frequencies [1, 2, 4, ...] become learnable
PER_NEURON_COEFFS = False  # If True, each output neuron has its own Fourier coefficients
L1_PENALTY = 0.0           # L1 regularization strength on Fourier coefficients (0.0 = disabled)

# Experiment Configurations
EXPERIMENTS = {
    "baseline": {
        "num_harmonics": 3,
        "adaptive_freqs": False,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.0,
        "description": "Default configuration (3 harmonics, fixed frequencies)"
    },
    "rich_harmonics": {
        "num_harmonics": 5,
        "adaptive_freqs": False,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.0,
        "description": "More harmonics for richer textures"
    },
    "adaptive": {
        "num_harmonics": 3,
        "adaptive_freqs": True,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.0,
        "description": "Learnable harmonic frequencies"
    },
    "sparse": {
        "num_harmonics": 5,
        "adaptive_freqs": False,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.001,
        "description": "5 harmonics with L1 sparsity"
    },
    "per_neuron": {
        "num_harmonics": 3,
        "adaptive_freqs": False,
        "per_neuron_coeffs": True,
        "l1_penalty": 0.005,
        "description": "Per-neuron coefficients (max flexibility)"
    }
}

# Select which experiments to run (comment out to skip)
ACTIVE_EXPERIMENTS = [
    "baseline",
    # "rich_harmonics",
    # "adaptive", 
    # "sparse",
    # "per_neuron"
]

def run_experiment(exp_name, exp_config, device):
    """Run a single experiment configuration."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name.upper()}")
    print(f"Description: {exp_config['description']}")
    print(f"{'='*80}")
    
    results = {}
    spectral_modes = ["UserWave", "GatedWave"]  # Only spectral layers use Fourier config
    other_modes = ["Standard"]  # Add others if you want full comparison
    
    for m in spectral_modes:
        results[m] = train_fit(
            m, NUM_WAVES, NUM_EPOCHS, device,
            num_harmonics=exp_config['num_harmonics'],
            adaptive_freqs=exp_config['adaptive_freqs'],
            per_neuron_coeffs=exp_config['per_neuron_coeffs'],
            l1_penalty=exp_config['l1_penalty']
        )
    
    # Run baseline (Standard) for comparison
    for m in other_modes:
        results[m] = train_fit(m, NUM_WAVES, NUM_EPOCHS, device)
    
    return results

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"./results/Experiments_{timestamp}/"
    os.makedirs(base_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}
    
    print(f"\n🔬 STARTING SPECTRAL CNN EXPERIMENTS")
    print(f"Running {len(ACTIVE_EXPERIMENTS)} experiment(s)")
    print(f"Results will be saved to: {base_dir}")
    
    for exp_name in ACTIVE_EXPERIMENTS:
        if exp_name not in EXPERIMENTS:
            print(f"⚠️ Warning: Experiment '{exp_name}' not defined, skipping...")
            continue
            
        exp_config = EXPERIMENTS[exp_name]
        exp_results = run_experiment(exp_name, exp_config, device)
        all_results[exp_name] = (exp_config, exp_results)
        
        # Save experiment results
        exp_dir = f"{base_dir}/{exp_name}/"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Generate visualizations for this experiment
        modes = list(exp_results.keys())
        plot_results(exp_results, modes, exp_dir, NUM_EPOCHS)
        
        for m in modes:
            if m in ["UserWave", "GatedWave"]:
                plot_layer_waves(exp_results[m]['model'], m, exp_dir)
                # Fourier decomposition for first 2 waves
                for wave_idx in range(min(2, NUM_WAVES)):
                    plot_wave_decomposition(exp_results[m]['model'], m, exp_dir, wave_idx=wave_idx)
    
    # Final comparison report
    print("\n" + "="*120)
    print("FINAL EXPERIMENT COMPARISON")
    print("="*120)
    
    for exp_name in ACTIVE_EXPERIMENTS:
        if exp_name not in all_results:
            continue
        exp_config, results = all_results[exp_name]
        
        print(f"\n📊 {exp_name.upper()}: {exp_config['description']}")
        print(f"   Config: harmonics={exp_config['num_harmonics']}, adaptive={exp_config['adaptive_freqs']}, " +
              f"per_neuron={exp_config['per_neuron_coeffs']}, L1={exp_config['l1_penalty']}")
        print("-" * 120)
        print(f"| {'Model':<15} | {'Params':<10} | {'Train %':<8} | {'Test %':<8} | {'Time (s)':<10} | {'Speed':<12} |")
        print("-" * 120)
        
        for m in results:
            r = results[m]
            print(f"| {m:<15} | {r['params']:<10,} | {r['train_acc']:<8.2f} | {r['test_acc']:<8.2f} | " +
                  f"{r['total_time']:<10.2f} | {r['inference_speed']:<12.0f} |")
    
    print("\n" + "="*120)
    print(f"✅ All experiments completed! Results saved to: {base_dir}")


if __name__ == "__main__":
    main()
