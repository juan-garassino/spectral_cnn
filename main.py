import os
import torch
import numpy as np
from datetime import datetime
from src.training.trainer import train_fit
from src.visualization.plotter import plot_results, plot_layer_waves, plot_wave_decomposition
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Configuration
NUM_EPOCHS = 1 # 3
NUM_WAVES = 12
MODES = ["Standard", "UserWave", "Poly", "Wavelet", "Factor", "Siren", "GatedWave"]

# Fourier Configuration (for UserWave and GatedWave)
NUM_HARMONICS = 3          # Number of Fourier components per wave (3 = cos(θ), cos(2θ), cos(4θ))
ADAPTIVE_FREQS = False     # If True, harmonic frequencies [1, 2, 4, ...] become learnable
PER_NEURON_COEFFS = False  # If True, each output neuron has its own Fourier coefficients
L1_PENALTY = 0.0           # L1 regularization strength on Fourier coefficients (0.0 = disabled)
WAVE_MODE = "fourier_series"  # "outer_product" (2D patterns) or "fourier_series" (1D smooth sinusoids)

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
    "rich_harmonics",
    # "adaptive", 
    # "sparse",
    # "per_neuron"
]

def run_experiment(exp_name, exp_config, device):
    """Run a single experiment configuration."""
    console.print()
    console.print(Panel(
        f"[bold yellow]EXPERIMENT: {exp_name.upper()}[/]\n"
        f"[dim]{exp_config['description']}[/]\n\n"
        f"[cyan]Config:[/] harmonics={exp_config['num_harmonics']}, adaptive={exp_config['adaptive_freqs']}, "
        f"per_neuron={exp_config['per_neuron_coeffs']}, L1={exp_config['l1_penalty']}",
        title=f"📊 Experiment {ACTIVE_EXPERIMENTS.index(exp_name) + 1}/{len(ACTIVE_EXPERIMENTS)}",
        border_style="yellow"
    ))
    
    results = {}
    
    # Run all models for complete comparison
    for m in MODES:
        if m in ["UserWave", "GatedWave"]:
            # Spectral layers with Fourier configuration
            results[m] = train_fit(
                m, NUM_WAVES, NUM_EPOCHS, device,
                num_harmonics=exp_config['num_harmonics'],
                adaptive_freqs=exp_config['adaptive_freqs'],
                per_neuron_coeffs=exp_config['per_neuron_coeffs'],
                l1_penalty=exp_config['l1_penalty'],
                wave_mode=WAVE_MODE  # Use global WAVE_MODE setting
            )
        else:
            # Other models use default settings
            results[m] = train_fit(m, NUM_WAVES, NUM_EPOCHS, device)
    
    return results

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"./results/Experiments_{timestamp}/"
    os.makedirs(base_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}
    
    console.print(Panel(
        f"[bold green]🔬 SPECTRAL CNN EXPERIMENTS[/]\n\n"
        f"[cyan]Running:[/] {len(ACTIVE_EXPERIMENTS)} experiment(s)\n"
        f"[cyan]Wave Mode:[/] {WAVE_MODE}\n"
        f"[cyan]Results:[/] {base_dir}",
        title="Benchmark Suite",
        border_style="green"
    ))
    
    for exp_name in ACTIVE_EXPERIMENTS:
        if exp_name not in EXPERIMENTS:
            console.print(f"[yellow]⚠️  Warning: Experiment '{exp_name}' not defined, skipping...[/]")
            continue
            
        exp_config = EXPERIMENTS[exp_name]
        exp_results = run_experiment(exp_name, exp_config, device)
        all_results[exp_name] = (exp_config, exp_results)
        
        # Save experiment results
        exp_dir = f"{base_dir}/{exp_name}/"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Generate visualizations for this experiment
        console.print("[dim]Generating visualizations...[/]")
        modes = list(exp_results.keys())
        plot_results(exp_results, modes, exp_dir, NUM_EPOCHS)
        
        for m in modes:
            if m in ["UserWave", "GatedWave"]:
                plot_layer_waves(exp_results[m]['model'], m, exp_dir)
                # Fourier decomposition for first 2 waves
                for wave_idx in range(min(2, NUM_WAVES)):
                    plot_wave_decomposition(exp_results[m]['model'], m, exp_dir, wave_idx=wave_idx)
    
    # Final comparison report
    console.print()
    console.print(Panel("[bold magenta]FINAL EXPERIMENT COMPARISON[/]", border_style="magenta"))
    
    for exp_name in ACTIVE_EXPERIMENTS:
        if exp_name not in all_results:
            continue
        exp_config, results = all_results[exp_name]
        
        # Create rich table
        table = Table(title=f"{exp_name.upper()}: {exp_config['description']}", show_header=True, header_style="bold cyan")
        table.add_column("Model", style="cyan")
        table.add_column("Params", justify="right", style="yellow")
        table.add_column("Comp Ratio", justify="right", style="green")
        table.add_column("Train %", justify="right", style="blue")
        table.add_column("Test %", justify="right", style="magenta")
        table.add_column("Gen Gap", justify="right", style="red")
        table.add_column("Time (s)", justify="right", style="white")
        table.add_column("Speed (s/s)", justify="right", style="white")
        
        baseline_params = results["Standard"]["params"]
        for m in MODES:
            if m not in results:
                continue
            r = results[m]
            comp_ratio = baseline_params / r['params'] if r['params'] > 0 else 0
            gen_gap = r['train_acc'] - r['test_acc']
            
            table.add_row(
                m,
                f"{r['params']:,}",
                f"{comp_ratio:.2f}x",
                f"{r['train_acc']:.2f}",
                f"{r['test_acc']:.2f}",
                f"{gen_gap:.2f}",
                f"{r['total_time']:.2f}",
                f"{r['inference_speed']:.0f}"
            )
        
        console.print(table)
        console.print()
    
    console.print(Panel(
        f"[bold green]✅ All experiments completed![/]\n\n"
        f"[cyan]Results saved to:[/] {base_dir}",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
