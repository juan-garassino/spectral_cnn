"""
Wave-Native GPT Benchmark 🌊

Quick comparison: Classic Transformer vs Wave-Native GPT
"""

import torch
import torch.nn as nn
import time
import os
import sys
import gc
import math
import numpy as np
import matplotlib.pyplot as plt

# Add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Add prototyping folder for legacy modules
prototyping_dir = os.path.join(current_dir, "prototyping")
if prototyping_dir not in sys.path:
    sys.path.insert(0, prototyping_dir)

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from wave_gpt import WaveGPT, WaveGPTConfig
from train import BasicTokenizer, get_batch  # From prototyping/

# Config - SCALED UP for 16GB GPU
STEPS = 5000          # More training for convergence
BATCH_SIZE = 32       # Keep same (GPU at 98%)
BLOCK_SIZE = 256      # Keep same
D_MODEL = 256         # Keep same
NUM_LAYERS = 6        # Keep same
NUM_HEADS = 8         # Keep same
NUM_WAVES = 32        # Keep same
LR = 3e-4             # Base learning rate
WARMUP_STEPS = 200    # Warmup steps
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def get_gpu_memory_peak():
    """Get peak GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0


def save_wave_visualizations(model, losses, name, save_dir):
    """Save interpretability visualizations for wave models"""
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('dark_background')
    
    # 1. Learning Curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, alpha=0.3, color='cyan', label='Raw')
    # Smoothed
    window = min(50, len(losses)//10)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(losses)), smoothed, color='magenta', linewidth=2, label='Smoothed')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'{name} - Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_learning_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Check if this is a WaveGPT model
    if not hasattr(model, 'embedding') or not hasattr(model.embedding, 'token_freqs'):
        return  # Not a wave model, skip wave-specific plots
    
    with torch.no_grad():
        # 2. Token Frequency Distribution
        freqs = model.embedding.token_freqs.cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Frequency heatmap
        im = axes[0].imshow(freqs[:100].T, aspect='auto', cmap='viridis')
        axes[0].set_xlabel('Token ID (first 100)')
        axes[0].set_ylabel('Wave Component')
        axes[0].set_title('Token Frequencies')
        plt.colorbar(im, ax=axes[0])
        
        # Frequency histogram
        axes[1].hist(freqs.flatten(), bins=50, color='cyan', alpha=0.7)
        axes[1].set_xlabel('Frequency Value')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Frequency Distribution')
        axes[1].axvline(freqs.mean(), color='magenta', linestyle='--', label=f'Mean: {freqs.mean():.2f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_frequencies.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Token Phase Distribution  
        phases = model.embedding.token_phases.cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(phases[:100].T, aspect='auto', cmap='twilight')
        ax.set_xlabel('Token ID (first 100)')
        ax.set_ylabel('Wave Component')
        ax.set_title('Token Phases (0 to 2π)')
        plt.colorbar(im, ax=ax)
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_phases.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Sample Wave Packets for specific tokens
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        sample_tokens = [0, 1, 10, 50, 100, 500]  # Sample token IDs
        t = np.linspace(0, 2*np.pi, 200)
        
        for idx, (ax, tok_id) in enumerate(zip(axes.flat, sample_tokens)):
            if tok_id < len(freqs):
                wave = np.zeros_like(t)
                for w in range(min(5, freqs.shape[1])):  # First 5 waves
                    f = freqs[tok_id, w]
                    p = phases[tok_id, w]
                    a = model.embedding.token_amps[tok_id, w].cpu().item()
                    wave += a * np.sin(f * t + p)
                ax.plot(t, wave, color='cyan')
                ax.set_title(f'Token {tok_id}')
                ax.set_xlabel('t')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Sample Wave Packets per Token', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_wave_packets.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Attention Interference Weights (if available)
        if hasattr(model.blocks[0].attn, 'q_phase'):
            q_phase = model.blocks[0].attn.q_phase.cpu().numpy().squeeze()
            k_phase = model.blocks[0].attn.k_phase.cpu().numpy().squeeze()
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].bar(range(len(q_phase.flatten())), q_phase.flatten(), color='cyan', alpha=0.7)
            axes[0].set_title('Q Phase Shifts')
            axes[0].set_xlabel('Head × Wave')
            
            axes[1].bar(range(len(k_phase.flatten())), k_phase.flatten(), color='magenta', alpha=0.7)
            axes[1].set_title('K Phase Shifts')
            axes[1].set_xlabel('Head × Wave')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_attention_phases.png", dpi=150, bbox_inches='tight')
            plt.close()

def run_wave_benchmark(config_name, model, train_data, console, tokenizer):
    """Benchmark a single model"""
    console.print(Panel(f"[bold cyan]{config_name}[/bold cyan]", title="🧪 Model Config", border_style="cyan"))
    
    params = sum(p.numel() for p in model.parameters())
    console.print(f"📊 Parameters: [bold]{params:,}[/bold] ({params/1e6:.2f}M)")
    
    # Training with cosine LR schedule
    console.print(f"⚙️  Learning Rate: [bold]{LR:.0e}[/bold] (cosine decay, {WARMUP_STEPS} warmup)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # Cosine schedule with warmup
    def get_lr(step):
        if step < WARMUP_STEPS:
            return LR * step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)
        return LR * 0.5 * (1 + math.cos(math.pi * progress))
    
    model.train()
    losses = []
    
    console.print("🔥 Warming up...")
    try:
        for _ in range(3):
            x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            with torch.no_grad():
                _, _ = model(x, y)
    except Exception as e:
        console.print(f"[red]Warmup failed: {e}[/red]")
        return None
    
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
        task = progress.add_task(f"Training {config_name}", total=STEPS, loss=0.0)
        
        for step in range(STEPS):
            # Update learning rate
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            total_tokens += x.numel()
            
            # Forward
            _, loss = model(x, y)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                console.print(f"[red]⚠️  Training collapsed at step {step+1}[/red]")
                return None
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            progress.update(task, advance=1, loss=loss.item())
            
    # Log every 100 steps with memory
            if (step + 1) % 100 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                mem = get_gpu_memory()
                console.print(f"Step {step+1:4d} | Loss: {loss.item():.4f} | Avg: {avg:.4f} | GPU: {mem:.0f}MB")
    
    elapsed = time.perf_counter() - start_time
    speed = total_tokens / elapsed
    peak_mem = get_gpu_memory_peak()
    
    # Final eval
    model.eval()
    with torch.no_grad():
        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        _, val_loss = model(x, y)
    
    # Generate sample
    console.print("\n📝 Generation Sample:")
    idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=40)
    text = tokenizer.decode(generated[0].tolist())
    console.print(Panel(text[:200], title="Generated Text", border_style="green"))
    
    console.print(f"🎯 Peak GPU Memory: [bold]{peak_mem:.0f}MB[/bold] / 16000MB")
    
    return {
        "model": config_name,
        "params": params,
        "speed": speed,
        "val_loss": val_loss.item(),
        "perplexity": torch.exp(val_loss).item(),
        "time": elapsed,
        "peak_memory": peak_mem,
        "losses": losses,
        "model_ref": model
    }


def main():
    console = Console()
    console.print(Panel.fit(
        "[bold magenta]🌊 WAVE-NATIVE GPT BENCHMARK (SCALED UP)[/bold magenta]\n\n"
        f"d_model={D_MODEL} | layers={NUM_LAYERS} | heads={NUM_HEADS} | waves={NUM_WAVES}\n"
        f"batch={BATCH_SIZE} | context={BLOCK_SIZE} | steps={STEPS}",
        border_style="magenta"
    ))
    
    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Load data
    console.print("\n📚 Loading Shakespeare...")
    data_path = os.path.join(current_dir, "data", "tiny_shakespeare.txt")
    if not os.path.exists(data_path):
        import urllib.request
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            data_path
        )
    
    text = open(data_path, 'r').read()
    
    # Tokenizer - check benchmark_results folder first
    import json
    benchmark_results_tokenizer = os.path.join(current_dir, "benchmark_results", "tokenizer.json")
    results_tokenizer = os.path.join(current_dir, "results", "tokenizer.json")
    local_tokenizer = os.path.join(current_dir, "tokenizer.json")
    
    # Check in order of priority
    if os.path.exists(benchmark_results_tokenizer):
        tokenizer_path = benchmark_results_tokenizer
    elif os.path.exists(results_tokenizer):
        tokenizer_path = results_tokenizer
    elif os.path.exists(local_tokenizer):
        tokenizer_path = local_tokenizer
    else:
        tokenizer_path = None
    
    if tokenizer_path:
        console.print(f"♻️  Loading existing tokenizer from {tokenizer_path}...")
        tokenizer = BasicTokenizer()
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)
            tokenizer.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
    else:
        console.print("🔧 Training new tokenizer...")
        tokenizer = BasicTokenizer()
        tokenizer.train(text, 1024)
        tokenizer.save(local_tokenizer)
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data = data[:int(0.9 * len(data))]
    
    results = []
    
    # ========================================
    # Model 1: Classic Transformer (SCALED UP)
    # ========================================
    console.print("\n" + "="*60)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Import from prototyping/ folder
    from model import SpectralGPT
    from train import GPTConfig
    
    classic_config = GPTConfig(
        vocab_size=1024, d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, d_ff=D_MODEL*4, block_size=BLOCK_SIZE,
        dropout=0.1, layer_type="attention", weight_type="standard",
        num_waves=12, num_harmonics=5
    )
    classic_model = SpectralGPT(classic_config).to(DEVICE)
    
    res = run_wave_benchmark("Classic Transformer", classic_model, train_data, console, tokenizer)
    if res:
        results.append(res)
    
    del classic_model
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    # ========================================
    # Model 2: Wave-Native GPT (SCALED UP)
    # ========================================
    console.print("\n" + "="*60)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    wave_config = WaveGPTConfig(
        vocab_size=1024, d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, num_waves=NUM_WAVES, block_size=BLOCK_SIZE,
        dropout=0.1
    )
    wave_model = WaveGPT(wave_config).to(DEVICE)
    
    res = run_wave_benchmark("Wave-Native GPT 🌊", wave_model, train_data, console, tokenizer)
    if res:
        results.append(res)
    
    # ========================================
    # Save Visualizations
    # ========================================
    save_dir = os.path.join(current_dir, "benchmark_results", "wave_gpt_plots")
    console.print(f"\n📊 Saving visualizations to {save_dir}...")
    
    for r in results:
        if r and 'losses' in r and 'model_ref' in r:
            save_wave_visualizations(r['model_ref'], r['losses'], r['model'], save_dir)
            console.print(f"  ✅ Saved plots for {r['model']}")
    
    # Comparison plot
    fig, ax = plt.subplots(figsize=(12, 5))
    for r in results:
        if r and 'losses' in r:
            window = 50
            smoothed = np.convolve(r['losses'], np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(r['losses'])), smoothed, label=r['model'], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('🌊 Wave-Native GPT vs Classic Transformer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/comparison_learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    console.print(f"  ✅ Saved comparison plot")
    
    # ========================================
    # Results Table
    # ========================================
    console.print("\n")
    table = Table(title="🌊 Wave-Native GPT Benchmark Results (SCALED UP)")
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Speed (tok/s)", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Peak GPU", justify="right")
    
    for r in results:
        table.add_row(
            r["model"],
            f"{r['params']/1e6:.2f}M",
            f"{r['speed']:,.0f}",
            f"{r['perplexity']:.2f}",
            f"{r['val_loss']:.4f}",
            f"{r.get('peak_memory', 0):.0f}MB"
        )
    
    console.print(table)
    
    console.print(f"\n📁 Plots saved to: {save_dir}")
    console.print("\n[bold green]✅ Benchmark Complete![/bold green]")


if __name__ == "__main__":
    main()

