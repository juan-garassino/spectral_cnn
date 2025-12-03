"""
Spectral GPT Benchmark Suite 🏎️📊

Compares 4 architectures on Tiny Shakespeare:
1. Classic Transformer (Attention + Standard Weights)
2. Spectral Transformer (Attention + Wave Weights)
3. FFT Mixer (FFT + Standard Weights) - "Transformer Killer"
4. Full Spectral (FFT + Wave Weights)

Features:
- Metrics: Loss, Perplexity, Speed, Memory, Params
- Artifacts: Saves models, tokenizer, and config
- Visualization: Loss curves, comparison plots
- Packaging: Zips everything into benchmark_results.zip
"""

import torch
import torch.nn as nn
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import shutil
import math
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

from model import SpectralGPT
from train import BasicTokenizer, get_batch, GPTConfig

# Configuration
STEPS = 200
BATCH_SIZE = 16
BLOCK_SIZE = 128
D_MODEL = 128
LAYERS = 4
HEADS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_benchmark(config_name, layer_type, weight_type, train_data, val_data, vocab_size, results_dir, 
                  init_mode="standard", use_hamiltonian=False, use_collapse=False, activation_type="gelu"):
    print(f"\n--- Benchmarking: {config_name} ---")
    
    # Reset
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Config
    config = GPTConfig(
        vocab_size=vocab_size, d_model=D_MODEL, num_layers=LAYERS,
        num_heads=HEADS, d_ff=D_MODEL*4, block_size=BLOCK_SIZE,
        dropout=0.0, layer_type=layer_type, weight_type=weight_type,
        num_waves=12, num_harmonics=5,
        init_mode=init_mode, activation_type=activation_type
    )
    
    # Model
    try:
        model = SpectralGPT(config).to(DEVICE)
    except Exception as e:
        print(f"Failed to init model: {e}")
        return None

    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training Loop
    model.train()
    losses = []
    
    # Warmup
    for _ in range(5):
        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        _, _ = model(x, y)
        
    # Timed Run
    run_start = time.time()
    
    from train import get_collapse_penalty # Import here to avoid circular dependency issues if any
    
    for step in range(STEPS):
        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        
        # Forward
        if DEVICE == 'cuda':
            with torch.amp.autocast('cuda'):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
            
        # Physics Penalties
        if use_collapse:
            loss += 1e-4 * get_collapse_penalty(model)
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Physics Constraints
        if use_hamiltonian:
            model.constrain_energy()
        
        losses.append(loss.item())
        
    total_time = time.time() - run_start
    tokens_per_sec = (STEPS * BATCH_SIZE * BLOCK_SIZE) / total_time
    
    # Evaluation (Validation Loss & Perplexity)
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):
            x, y = get_batch(val_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            _, loss = model(x, y)
            val_losses.append(loss.item())
            
    final_val_loss = sum(val_losses) / len(val_losses)
    perplexity = math.exp(final_val_loss)
    
    memory_mb = 0
    if DEVICE == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
    # Save Model
    model_name = config_name.replace(" ", "_").lower()
    torch.save(model.state_dict(), results_dir / "models" / f"{model_name}.pt")
    
    return {
        "Model": config_name,
        "Layer": layer_type,
        "Weights": weight_type,
        "Params": params,
        "Speed (tok/s)": int(tokens_per_sec),
        "Memory (MB)": int(memory_mb),
        "Train Loss": round(losses[-1], 4),
        "Val Loss": round(final_val_loss, 4),
        "Perplexity": round(perplexity, 2),
        "Time (s)": round(total_time, 2),
        "loss_history": losses
    }

def main():
    console = Console()
    console.print("[bold green]Starting Spectral GPT Benchmark Suite...[/bold green]")
    
    # Setup Results Directory
    results_dir = Path("benchmark_results")
    if results_dir.exists(): shutil.rmtree(results_dir)
    (results_dir / "models").mkdir(parents=True)
    (results_dir / "plots").mkdir(parents=True)
    (results_dir / "logs").mkdir(parents=True)
    
    # Data
    if not os.path.exists("input.txt"):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "input.txt")
    
    with open("input.txt", 'r') as f: text = f.read()
    tokenizer = BasicTokenizer()
    tokenizer.train(text, 1024)
    tokenizer.save(results_dir / "tokenizer.json")
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    
    # Configurations to Test
    # Format: (Name, Layer, Weight, InitMode, UseHamiltonian, UseCollapse, Activation)
    configs = [
        ("Classic Transformer", "attention", "standard", "standard", False, False, "gelu"),
        ("Spectral Transformer", "attention", "wave", "standard", False, False, "gelu"),
        ("FFT Mixer (GFNet)", "fft", "standard", "standard", False, False, "gelu"),
        ("Full Spectral", "fft", "wave", "standard", False, False, "gelu"),
        ("Physics Spectral ⚛️", "fft", "wave", "holographic", True, True, "modulate"),
    ]
    
    results = []
    loss_histories = {}
    
    for name, layer, weight, init, ham, coll, act in configs:
        res = run_benchmark(name, layer, weight, train_data, val_data, 1024, results_dir, 
                          init_mode=init, use_hamiltonian=ham, use_collapse=coll, activation_type=act)
        if res:
            loss_histories[name] = res.pop("loss_history") # Separate history for plotting
            results.append(res)
        
    # Display Results
    table = Table(title="Spectral GPT Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Speed (tok/s)", justify="right", style="green")
    table.add_column("Perplexity", justify="right", style="yellow")
    table.add_column("Val Loss", justify="right", style="red")
    table.add_column("Memory (MB)", justify="right", style="magenta")
    
    for r in results:
        table.add_row(
            r["Model"], 
            f"{r['Params']:,}", 
            f"{r['Speed (tok/s)']:,}", 
            f"{r['Perplexity']}",
            f"{r['Val Loss']}",
            f"{r['Memory (MB)']}"
        )
        
    console.print(table)
    
    # Save Metrics
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "metrics.csv", index=False)
    
    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    # 1. Training Loss Curves
    plt.figure(figsize=(12, 6))
    for name, history in loss_histories.items():
        plt.plot(history, label=name, alpha=0.8)
    plt.title("Training Loss Convergence")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(results_dir / "plots" / "loss_curves.png")
    plt.close()
    
    # 2. Speed Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Speed (tok/s)", hue="Model", palette="viridis")
    plt.title("Training Speed (Tokens/Sec)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "speed_comparison.png")
    plt.close()
    
    # 3. Perplexity Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Perplexity", hue="Model", palette="magma")
    plt.title("Validation Perplexity (Lower is Better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "perplexity_comparison.png")
    plt.close()
    
    # 4. Memory Usage
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Memory (MB)", hue="Model", palette="rocket")
    plt.title("Peak Memory Usage (MB)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "memory_usage.png")
    plt.close()
    
    # --- STRESS TEST (Long Context) ---
    console.print("\n[bold red]🔥 Starting Long-Context Stress Test (N=2048)...[/bold red]")
    stress_results = []
    
    # Only compare architecture scaling (Standard Weights) to isolate O(N^2) vs O(NlogN)
    stress_configs = [
        ("Classic Transformer", "attention", "standard"),
        ("FFT Mixer (GFNet)", "fft", "standard"),
    ]
    
    # Stress Config
    STRESS_BLOCK_SIZE = 2048
    STRESS_BATCH_SIZE = 2  # Low batch size to fit in memory
    
    for name, layer, weight in stress_configs:
        print(f"\n--- Stress Testing: {name} ---")
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        config = GPTConfig(
            vocab_size=1024, d_model=D_MODEL, num_layers=LAYERS,
            num_heads=HEADS, d_ff=D_MODEL*4, block_size=STRESS_BLOCK_SIZE,
            dropout=0.0, layer_type=layer, weight_type=weight,
            num_waves=12, num_harmonics=5
        )
        
        try:
            model = SpectralGPT(config).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            # Warmup
            x = torch.randint(0, 1024, (STRESS_BATCH_SIZE, STRESS_BLOCK_SIZE)).to(DEVICE)
            y = torch.randint(0, 1024, (STRESS_BATCH_SIZE, STRESS_BLOCK_SIZE)).to(DEVICE)
            model(x, y)
            
            # Timed Run (Forward + Backward)
            start = time.time()
            for _ in range(20): # 20 steps
                if DEVICE == 'cuda':
                    with torch.amp.autocast('cuda'): _, loss = model(x, y)
                else:
                    _, loss = model(x, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            dt = time.time() - start
            tok_per_sec = (20 * STRESS_BATCH_SIZE * STRESS_BLOCK_SIZE) / dt
            
            mem = 0
            if DEVICE == 'cuda':
                mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            stress_results.append({
                "Model": name,
                "Speed (tok/s)": int(tok_per_sec),
                "Memory (MB)": int(mem),
                "Time (s)": round(dt, 2)
            })
            
        except RuntimeError as e:
            print(f"OOM or Error: {e}")
            stress_results.append({
                "Model": name,
                "Speed (tok/s)": 0,
                "Memory (MB)": -1,
                "Time (s)": -1
            })

    # Display Stress Results
    stress_table = Table(title="Long-Context Stress Test (N=2048)")
    stress_table.add_column("Model", style="cyan")
    stress_table.add_column("Speed (tok/s)", justify="right", style="green")
    stress_table.add_column("Memory (MB)", justify="right", style="magenta")
    
    for r in stress_results:
        stress_table.add_row(r["Model"], f"{r['Speed (tok/s)']:,}", f"{r['Memory (MB)']}")
    console.print(stress_table)
    
    # Save Stress Metrics
    pd.DataFrame(stress_results).to_csv(results_dir / "stress_metrics.csv", index=False)

    # Zip Results (Update with stress test)
    shutil.make_archive("benchmark_results", 'zip', results_dir)
    console.print(f"\n[bold green]✅ Benchmark & Stress Test Complete![/bold green]")
    console.print(f"Results saved to [underline]benchmark_results.zip[/underline]")

if __name__ == "__main__":
    main()
