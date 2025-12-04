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

# Add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from wave_gpt import WaveGPT, WaveGPTConfig
from train import BasicTokenizer, get_batch

# Config
STEPS = 500
BATCH_SIZE = 16
BLOCK_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_wave_benchmark(config_name, model, train_data, console, tokenizer):
    """Benchmark a single model"""
    console.print(Panel(f"[bold cyan]{config_name}[/bold cyan]", title="🧪 Model Config", border_style="cyan"))
    
    params = sum(p.numel() for p in model.parameters())
    console.print(f"📊 Parameters: [bold]{params:,}[/bold] ({params/1e6:.2f}M)")
    
    # Training
    lr = 3e-4
    console.print(f"⚙️  Learning Rate: [bold]{lr:.0e}[/bold]")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
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
            
            # Log every 50 steps
            if (step + 1) % 50 == 0:
                avg = sum(losses[-50:]) / len(losses[-50:])
                console.print(f"Step {step+1:4d} | Loss: {loss.item():.4f} | Avg: {avg:.4f}")
    
    elapsed = time.perf_counter() - start_time
    speed = total_tokens / elapsed
    
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
    
    return {
        "model": config_name,
        "params": params,
        "speed": speed,
        "val_loss": val_loss.item(),
        "perplexity": torch.exp(val_loss).item(),
        "time": elapsed
    }


def main():
    console = Console()
    console.print(Panel.fit(
        "[bold magenta]🌊 WAVE-NATIVE GPT BENCHMARK[/bold magenta]\n\n"
        "Tokens as Wave Packets • Interference Attention • Wave Collapse",
        border_style="magenta"
    ))
    
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
    
    # Tokenizer
    tokenizer_path = os.path.join(current_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        console.print("♻️  Loading existing tokenizer...")
        tokenizer = BasicTokenizer()
        import json
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)
            tokenizer.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
    else:
        console.print("🔧 Training new tokenizer...")
        tokenizer = BasicTokenizer()
        tokenizer.train(text, 1024)
        tokenizer.save(tokenizer_path)
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data = data[:int(0.9 * len(data))]
    
    results = []
    
    # ========================================
    # Model 1: Classic Transformer (baseline)
    # ========================================
    console.print("\n" + "="*60)
    from model import SpectralGPT
    from train import GPTConfig
    
    classic_config = GPTConfig(
        vocab_size=1024, d_model=128, num_layers=4,
        num_heads=4, d_ff=512, block_size=BLOCK_SIZE,
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
    # Model 2: Wave-Native GPT
    # ========================================
    console.print("\n" + "="*60)
    
    wave_config = WaveGPTConfig(
        vocab_size=1024, d_model=128, num_layers=4,
        num_heads=4, num_waves=16, block_size=BLOCK_SIZE,
        dropout=0.1
    )
    wave_model = WaveGPT(wave_config).to(DEVICE)
    
    res = run_wave_benchmark("Wave-Native GPT 🌊", wave_model, train_data, console, tokenizer)
    if res:
        results.append(res)
    
    # ========================================
    # Results Table
    # ========================================
    console.print("\n")
    table = Table(title="🌊 Wave-Native GPT Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Speed (tok/s)", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("Val Loss", justify="right")
    
    for r in results:
        table.add_row(
            r["model"],
            f"{r['params']/1e6:.2f}M",
            f"{r['speed']:,.0f}",
            f"{r['perplexity']:.2f}",
            f"{r['val_loss']:.4f}"
        )
    
    console.print(table)
    
    console.print("\n[bold green]✅ Benchmark Complete![/bold green]")


if __name__ == "__main__":
    main()
