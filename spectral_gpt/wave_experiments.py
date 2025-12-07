"""
Wave-Native GPT Comprehensive Experiment Suite

Ablation studies, extended training, and dataset experiments.

Usage:
    # Run all ablation studies
    python wave_experiments.py --experiment all --steps 20000
    
    # Run specific experiment
    python wave_experiments.py --experiment rgd_only --steps 15000
    
    # FineWeb-Edu experiment
    python wave_experiments.py --dataset fineweb --model large
    
    # Multi-GPU
    python wave_experiments.py --experiment all --parallel
"""

import os
import sys
import gc
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'prototyping'))
sys.path.insert(0, current_dir)

from wave_gpt import WaveGPT, WaveGPTConfig
from physics_optim import ResonantGradientDescent, QuantumFieldEntanglementLoss
from train import BasicTokenizer, get_batch

# ==========================================
# Experiment Configurations
# ==========================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    use_rgd: bool = False
    use_qfe: bool = False
    pure_wave_attention: bool = False  # New flag for pure attention
    rgd_strength: float = 0.3
    rgd_warmup: int = 500
    qfe_lambda: float = 0.05
    qfe_threshold: float = 0.01
    lr: float = 6e-4
    steps: int = 2000#0
    wave_ratio_schedule: bool = True  # Schedule wave_ratio from 0.5 to 0.9


ABLATION_EXPERIMENTS = {
    "full_physics": ExperimentConfig(
        name="Full Physics (RGD + QFE)",
        use_rgd=True, use_qfe=True
    ),
    "pure_wave": ExperimentConfig(
        name="Pure Wave (No Softmax) 🌊",
        use_rgd=True, use_qfe=True,
        pure_wave_attention=True
    ),
    "rgd_only": ExperimentConfig(
        name="RGD Only",
        use_rgd=True, use_qfe=False
    ),
    "qfe_only": ExperimentConfig(
        name="QFE Only", 
        use_rgd=False, use_qfe=True
    ),
    "baseline": ExperimentConfig(
        name="Baseline (AdamW + CE)",
        use_rgd=False, use_qfe=False
    ),
}


@dataclass
class ModelConfig:
    """Model size configurations"""
    name: str
    d_model: int
    num_layers: int
    num_heads: int
    num_waves: int
    num_harmonics: int
    vocab_size: int
    block_size: int
    batch_size: int


MODEL_CONFIGS = {
    "small": ModelConfig(
        name="Small (Shakespeare)",
        d_model=384, num_layers=8, num_heads=8,
        num_waves=48, num_harmonics=4,
        vocab_size=1024, block_size=256, batch_size=32
    ),
    "medium": ModelConfig(
        name="Medium",
        d_model=512, num_layers=10, num_heads=8,
        num_waves=64, num_harmonics=4,
        vocab_size=8192, block_size=384, batch_size=24
    ),
    "large": ModelConfig(
        name="Large (FineWeb-Edu)",
        d_model=768, num_layers=12, num_heads=12,
        num_waves=96, num_harmonics=4,
        vocab_size=32000, block_size=512, batch_size=16
    ),
}


# ==========================================
# Dataset Loaders
# ==========================================

def load_shakespeare(console):
    """Load Shakespeare dataset"""
    data_path = os.path.join(current_dir, "data", "tiny_shakespeare.txt")
    
    if not os.path.exists(data_path):
        import urllib.request
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        console.print("📥 Downloading Shakespeare...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            data_path
        )
    
    with open(data_path, 'r') as f:
        text = f.read()
    
    return text


def load_fineweb(console, subset="sample-10BT", max_tokens=10_000_000):
    """
    Load FineWeb-Edu dataset from HuggingFace.
    
    Args:
        subset: "sample-10BT" (10B tokens) or "sample-100BT" (100B tokens)
        max_tokens: Maximum tokens to load for memory efficiency
    """
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]Please install datasets: pip install datasets[/red]")
        return None
    
    console.print(f"📥 Loading FineWeb-Edu ({subset})...")
    console.print(f"   Max tokens: {max_tokens:,}")
    
    # Stream the dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=subset,
        split="train",
        streaming=True
    )
    
    # Collect text up to max_tokens
    texts = []
    total_chars = 0
    target_chars = max_tokens * 4  # ~4 chars per token estimate
    
    with Progress(console=console) as progress:
        task = progress.add_task("Loading FineWeb...", total=target_chars)
        
        for item in ds:
            text = item.get("text", "")
            texts.append(text)
            total_chars += len(text)
            progress.update(task, completed=min(total_chars, target_chars))
            
            if total_chars >= target_chars:
                break
    
    full_text = "\n\n".join(texts)
    console.print(f"   Loaded {len(full_text):,} characters")
    
    return full_text


def get_dataset(dataset_name: str, console, max_tokens: int = 10_000_000):
    """Get dataset by name"""
    if dataset_name == "shakespeare":
        return load_shakespeare(console)
    elif dataset_name == "fineweb_small":
        return load_fineweb(console, subset="sample-10BT", max_tokens=1_000_000)
    elif dataset_name == "fineweb":
        return load_fineweb(console, subset="sample-10BT", max_tokens=max_tokens)
    elif dataset_name == "fineweb_large":
        return load_fineweb(console, subset="sample-100BT", max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ==========================================
# Training Loop
# ==========================================

def train_experiment(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    exp_config: ExperimentConfig,
    model_config: ModelConfig,
    console: Console,
    device: str = "cuda"
) -> Dict:
    """Train a single experiment configuration"""
    
    console.print(Panel(f"[bold cyan]{exp_config.name}[/bold cyan]", border_style="cyan"))
    
    params = sum(p.numel() for p in model.parameters())
    console.print(f"📊 Parameters: {params:,} ({params/1e6:.2f}M)")
    console.print(f"🔄 Steps: {exp_config.steps}")
    
    # Setup optimizer
    if exp_config.use_rgd:
        optimizer = ResonantGradientDescent(
            model.parameters(),
            lr=exp_config.lr,
            resonance_strength=exp_config.rgd_strength,
            warmup_steps=exp_config.rgd_warmup,
            weight_decay=0.01
        )
        console.print(f"⚡ Optimizer: RGD (strength={exp_config.rgd_strength})")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp_config.lr, weight_decay=0.01)
        console.print(f"⚙️  Optimizer: AdamW")
    
    # Setup loss
    if exp_config.use_qfe:
        loss_fn = QuantumFieldEntanglementLoss(
            lambda_coherence=exp_config.qfe_lambda,
            amplitude_threshold=exp_config.qfe_threshold
        )
        console.print(f"🌌 Loss: QFE (λ={exp_config.qfe_lambda})")
    else:
        loss_fn = None
        console.print(f"📉 Loss: Cross-Entropy")
    
    console.print(f"📈 LR: {exp_config.lr:.0e}")
    
    # LR schedule
    warmup_steps = 300
    def get_lr(step):
        if step < warmup_steps:
            return exp_config.lr * step / warmup_steps
        progress = (step - warmup_steps) / (exp_config.steps - warmup_steps)
        return exp_config.lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    # Wave ratio schedule (push toward pure wave)
    def get_wave_ratio_target(step):
        if not exp_config.wave_ratio_schedule:
            return 0.5
        # Schedule: 0.5 -> 0.9 over training
        progress = step / exp_config.steps
        return 0.5 + 0.4 * progress
    
    model.train()
    losses = []
    coherence_losses = []
    wave_ratios = []
    
    start_time = time.perf_counter()
    total_tokens = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[yellow]Loss: {task.fields[loss]:.4f}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Training", total=exp_config.steps, loss=0.0)
        
        for step in range(exp_config.steps):
            # Update LR
            current_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Update wave_ratio if available (handle DataParallel)
            base_model = model.module if hasattr(model, 'module') else model
            if hasattr(base_model.embedding, 'wave_ratio') and exp_config.wave_ratio_schedule:
                target_ratio = get_wave_ratio_target(step)
                # Softly push toward target
                with torch.no_grad():
                    # wave_ratio uses sigmoid, so we need inverse sigmoid (logit)
                    current = torch.sigmoid(base_model.embedding.wave_ratio).item()
                    # Blend toward target
                    new_ratio = current * 0.99 + target_ratio * 0.01
                    # Convert back to logit space
                    new_ratio = max(0.01, min(0.99, new_ratio))
                    base_model.embedding.wave_ratio.data = torch.tensor(
                        math.log(new_ratio / (1 - new_ratio))
                    ).to(device)
                wave_ratios.append(torch.sigmoid(base_model.embedding.wave_ratio).item())
            
            # Get batch
            x, y = get_batch(train_data, model_config.batch_size, model_config.block_size, device)
            total_tokens += x.numel()
            
            # Forward
            optimizer.zero_grad()
            logits, ce_loss = model(x, y)
            
            # Compute loss
            if exp_config.use_qfe and loss_fn is not None:
                loss_dict = loss_fn(logits, y, return_components=True)
                loss = loss_dict['total']
                coherence_losses.append(loss_dict['coherence'].item())
            else:
                loss = ce_loss
                if loss.ndim > 0:
                    loss = loss.mean()
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                console.print(f"[red]NaN/Inf at step {step}[/red]")
                break
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            progress.update(task, advance=1, loss=loss.item())
            
            # Log every 500 steps
            if (step + 1) % 500 == 0:
                avg = sum(losses[-500:]) / len(losses[-500:])
                wave_r = wave_ratios[-1] if wave_ratios else 0.5
                
                # Fast validation check
                model.eval()
                with torch.no_grad():
                    # Check on a small batch of validation data
                    val_x, val_y = get_batch(val_data, model_config.batch_size, model_config.block_size, device) 
                    _, val_loss_check = model(val_x, val_y)
                    if val_loss_check.ndim > 0:
                        val_loss_check = val_loss_check.mean()
                model.train()
                
                console.print(f"Step {step+1:5d} | Train: {loss.item():.4f} | Val: {val_loss_check.item():.4f} | AvgTrain: {avg:.4f} | R: {wave_r:.3f}")
    
    elapsed = time.perf_counter() - start_time
    speed = total_tokens / elapsed
    
    # Final eval
    model.eval()
    with torch.no_grad():
        x, y = get_batch(val_data, model_config.batch_size, model_config.block_size, device)
        _, val_loss = model(x, y)
        
        # Handle DataParallel output (one loss per dim)
        if val_loss.ndim > 0:
            val_loss = val_loss.mean()
    
    return {
        "name": exp_config.name,
        "config": asdict(exp_config),
        "val_loss": val_loss.item(),
        "perplexity": torch.exp(val_loss).item(),
        "speed": speed,
        "time": elapsed,
        "losses": losses,
        "coherence_losses": coherence_losses,
        "wave_ratios": wave_ratios,
        "params": params,
        "model": model
    }


# ==========================================
# Experiment Runner
# ==========================================

def run_ablation_suite(
    console: Console,
    experiments: List[str] = None,
    model_size: str = "small",
    dataset: str = "shakespeare",
    steps: int = 20000,
    device: str = "cuda",
    parallel: bool = False
) -> Dict:
    """Run ablation study experiments"""
    
    if experiments is None or "all" in experiments:
        experiments = list(ABLATION_EXPERIMENTS.keys())
    
    model_config = MODEL_CONFIGS[model_size]
    
    console.print(Panel.fit(
        f"[bold magenta]🔬 WAVE-NATIVE GPT ABLATION SUITE[/bold magenta]\n\n"
        f"Model: {model_config.name}\n"
        f"Dataset: {dataset}\n"
        f"Experiments: {', '.join(experiments)}\n"
        f"Steps per experiment: {steps}",
        border_style="magenta"
    ))
    
    # Load data
    text = get_dataset(dataset, console)
    if text is None:
        return {}
    
    # Train tokenizer or load
    tokenizer_path = os.path.join(current_dir, "benchmark_results", "tokenizer.json")
    if dataset == "shakespeare" and os.path.exists(tokenizer_path):
        console.print("♻️  Loading existing tokenizer...")
        tokenizer = BasicTokenizer()
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)
            tokenizer.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
    else:
        console.print(f"🔧 Training tokenizer (vocab={model_config.vocab_size})...")
        tokenizer = BasicTokenizer()
        tokenizer.train(text, model_config.vocab_size)
    
    # Encode data
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.8 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    console.print(f"📊 Tokens: Train {len(train_data):,} | Val {len(val_data):,}")
    
    results = {}
    
    for exp_name in experiments:
        console.print("\n" + "="*60)
        
        exp_config = ABLATION_EXPERIMENTS[exp_name]
        exp_config.steps = steps  # Override steps
        
        # Create model
        wave_config = WaveGPTConfig(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            num_waves=model_config.num_waves,
            num_harmonics=model_config.num_harmonics,
            block_size=model_config.block_size,
            dropout=0.1,
            pure_wave_attention=getattr(exp_config, 'pure_wave_attention', False)
        )
        model = WaveGPT(wave_config).to(device)
        
        # DataParallel if requested
        if parallel and torch.cuda.device_count() > 1:
            console.print(f"🔀 Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        # Train
        result = train_experiment(
            model, train_data, val_data, exp_config, model_config, console, device
        )
        results[exp_name] = result
        
        # Clean up
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return results


def plot_ablation_results(results: Dict, save_dir: str):
    """Generate comparison plots for ablation study"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('dark_background')
    
    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['cyan', 'magenta', 'yellow', 'lime']
    
    for i, (name, result) in enumerate(results.items()):
        losses = result['losses']
        window = 100
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(losses)), smoothed, 
                label=f"{result['name']} (final: {result['val_loss']:.3f})",
                color=colors[i % len(colors)], linewidth=2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Ablation Study: Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/ablation_loss_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Wave ratio progression (if available)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, result) in enumerate(results.items()):
        if result.get('wave_ratios'):
            ax.plot(result['wave_ratios'], 
                    label=result['name'],
                    color=colors[i % len(colors)], linewidth=2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Wave Ratio')
    ax.set_title('Wave Ratio Progression (0.5 → 0.9 schedule)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/wave_ratio_progression.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [r['name'] for r in results.values()]
    val_losses = [r['val_loss'] for r in results.values()]
    perplexities = [r['perplexity'] for r in results.values()]
    
    axes[0].barh(names, val_losses, color=colors[:len(names)])
    axes[0].set_xlabel('Validation Loss')
    axes[0].set_title('Final Validation Loss')
    
    axes[1].barh(names, perplexities, color=colors[:len(names)])
    axes[1].set_xlabel('Perplexity')
    axes[1].set_title('Final Perplexity')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Save results JSON
    summary = {
        name: {
            "val_loss": r["val_loss"],
            "perplexity": r["perplexity"],
            "speed": r["speed"],
            "time": r["time"],
            "config": r["config"]
        }
        for name, r in results.items()
    }
    with open(f"{save_dir}/ablation_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def print_results_table(results: Dict, console: Console):
    """Print results as a nice table"""
    table = Table(title="🔬 Ablation Study Results")
    table.add_column("Experiment", style="cyan")
    table.add_column("RGD", justify="center")
    table.add_column("QFE", justify="center")
    table.add_column("Val Loss", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("Speed (tok/s)", justify="right")
    
    for name, result in results.items():
        config = result['config']
        table.add_row(
            result['name'],
            "✓" if config['use_rgd'] else "✗",
            "✓" if config['use_qfe'] else "✗",
            f"{result['val_loss']:.4f}",
            f"{result['perplexity']:.2f}",
            f"{result['speed']:,.0f}"
        )
    
    console.print(table)


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Wave-Native GPT Experiment Suite")
    parser.add_argument("--experiment", type=str, nargs="+", default=["all"],
                        choices=["all"] + list(ABLATION_EXPERIMENTS.keys()),
                        help="Experiments to run")
    parser.add_argument("--model", type=str, default="small",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model size")
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        choices=["shakespeare", "fineweb_small", "fineweb", "fineweb_large"],
                        help="Dataset to use")
    parser.add_argument("--steps", type=int, default=20000,
                        help="Training steps per experiment")
    parser.add_argument("--parallel", action="store_true",
                        help="Use DataParallel for multi-GPU")
    parser.add_argument("--output", type=str, default="experiment_results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    console = Console()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run experiments
    results = run_ablation_suite(
        console,
        experiments=args.experiment,
        model_size=args.model,
        dataset=args.dataset,
        steps=args.steps,
        device=device,
        parallel=args.parallel
    )
    
    if not results:
        console.print("[red]No results to save[/red]")
        return
    
    # Save results
    output_dir = os.path.join(current_dir, args.output)
    summary = plot_ablation_results(results, output_dir)
    
    # Print table
    print_results_table(results, console)
    
    console.print(f"\n📁 Results saved to: {output_dir}")
    console.print("[bold green]✅ Experiment suite complete![/bold green]")


if __name__ == "__main__":
    main()
