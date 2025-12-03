"""
Unified Spectral GPT Training Script 🚀

Trains the SpectralGPT model with support for:
- Architecture: FFT Mixing ("Transformer Killer") OR Wave Attention
- Weights: Standard OR Wave-based
- Advanced Training: Frequency Consistency, Curriculum Learning, Two-Phase Annealing

Usage:
    python train.py --layer-type fft --weight-type standard (The "Transformer Killer")
    python train.py --layer-type attention --weight-type wave (The "Spectral Transformer")
"""

import os
import math
import time
import json
import shutil
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataclasses import dataclass

from model import SpectralGPT

# ==========================================
# 0. CONFIGURATION & ARGS
# ==========================================

@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    block_size: int
    dropout: float
    layer_type: str
    num_waves: int
    num_harmonics: int
    init_mode: str = "standard"
    activation_type: str = "gelu"

def parse_args():
    parser = argparse.ArgumentParser(description='Train Unified Spectral GPT')
    
    # Architecture
    parser.add_argument('--layer-type', type=str, default='fft', choices=['fft', 'attention'],
                       help='fft: Global FFT Mixing (O(NlogN)), attention: Standard Multi-Head (O(N^2))')
    parser.add_argument('--weight-type', type=str, default='standard', choices=['standard', 'wave'],
                       help='standard: nn.Linear, wave: UserWaveLinear (Sum of Sines)')
    parser.add_argument('--activation-type', type=str, default='gelu', choices=['gelu', 'bilinear', 'modulate'],
                       help='gelu: Standard, bilinear: Swish/SiLU, modulate: x*cos(x) (Spectral Mixing)')
    
    # Physics-First Strategies
    parser.add_argument('--init-mode', type=str, default='standard', 
                       choices=['standard', 'dft', 'holographic', 'standing_wave'],
                       help='Physics-based initialization strategies')
    parser.add_argument('--use-hamiltonian', action='store_true', default=False,
                       help='Hamiltonian Descent: Enforce energy conservation (unitary constraints)')
    parser.add_argument('--use-collapse', action='store_true', default=False,
                       help='Wave Function Collapse: Sparsity penalty to crystallize structure')
    parser.add_argument('--collapse-weight', type=float, default=1e-4,
                       help='Strength of the collapse (L1) penalty')
    
    # Model Config
    parser.add_argument('--vocab_size', type=int, default=1024)
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-waves', type=int, default=12)
    parser.add_argument('--num-harmonics', type=int, default=5)
    
    # Training Config
    parser.add_argument('--exp-name', type=str, default='spectral_gpt_run')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.02)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--eval-interval', type=int, default=250)
    
    # Advanced Features
    parser.add_argument('--use-curriculum', action='store_true', default=True, help='Progressive frequency unfreezing')
    parser.add_argument('--use-freq-consistency', action='store_true', default=False, help='Enforce octave spacing (for wave weights)')
    parser.add_argument('--freq-consistency-weight', type=float, default=0.01)
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()

# ==========================================
# 1. TOKENIZER (Minimal BPE)
# ==========================================

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx); i += 2
        else: newids.append(ids[i]); i += 1
    return newids

class BasicTokenizer:
    def __init__(self): self.merges = {}; self.vocab = {}
    def train(self, text, vocab_size):
        print(f"[Tokenizer] Training BPE on {len(text)} chars -> {vocab_size} tokens...")
        ids = list(text.encode("utf-8"))
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats: break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if (i+1) % 100 == 0: print(f"  Merge {i+1}/{vocab_size-256}")
    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while True:
            stats = get_stats(ids)
            pairs = [p for p in stats.keys() if p in self.merges]
            if not pairs: break
            pair = min(pairs, key=lambda p: self.merges[p])
            ids = merge(ids, pair, self.merges[pair])
        return ids
    def decode(self, ids):
        return b"".join(self.vocab.get(idx, b'\x00') for idx in ids).decode("utf-8", errors="replace")
    def save(self, path):
        with open(path, 'w') as f: json.dump({'merges': {str(k): v for k, v in self.merges.items()}, 'vocab': {k: list(v) for k, v in self.vocab.items()}}, f)

# ==========================================
# 2. TRAINING UTILS
# ==========================================

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+1+block_size] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=50):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            if device == 'cuda':
                with torch.amp.autocast('cuda'): _, loss = model(x, y, progress=1.0)
            else: _, loss = model(x, y, progress=1.0)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(step, max_steps, max_lr, warmup_steps):
    if step < warmup_steps: return max_lr * (step / warmup_steps)
    if step > max_steps: return max_lr * 0.1
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * 0.1 + coeff * (max_lr * 0.9)

def get_freq_consistency_loss(model):
    loss = 0.0
    count = 0
    for m in model.modules():
        if hasattr(m, 'freqs') and isinstance(m.freqs, nn.Parameter):
            freqs = m.freqs.flatten().abs()
            if len(freqs) > 1:
                sorted_freqs, _ = torch.sort(freqs)
                ratios = sorted_freqs[1:] / (sorted_freqs[:-1] + 1e-6)
                loss += torch.mean((ratios - 2.0) ** 2)
                count += 1
    return loss / max(count, 1)

def get_collapse_penalty(model):
    """Sadar Effect: L1 penalty on wave amplitudes to force collapse."""
    loss = 0.0
    count = 0
    for m in model.modules():
        if hasattr(m, 'amplitudes') and isinstance(m.amplitudes, nn.Parameter):
            loss += torch.abs(m.amplitudes).mean()
            count += 1
        elif hasattr(m, 'complex_weight') and isinstance(m.complex_weight, nn.Parameter):
            # For FFT layers, penalize magnitude
            loss += torch.abs(m.complex_weight).mean()
            count += 1
    return loss / max(count, 1)

# ==========================================
# 3. MAIN
# ==========================================

def main():
    args = parse_args()
    if os.path.exists(args.exp_name): shutil.rmtree(args.exp_name)
    os.makedirs(f"{args.exp_name}/logs", exist_ok=True)
    os.makedirs(f"{args.exp_name}/ckpts", exist_ok=True)
    
    print("="*60)
    print(f"🌊 SPECTRAL GPT: {args.layer_type.upper()} + {args.weight_type.upper()}")
    print(f"⚛️  PHYSICS MODE: Init={args.init_mode.upper()} | Hamiltonian={args.use_hamiltonian} | Collapse={args.use_collapse}")
    print("="*60)
    
    # Data
    data_path = os.path.join(args.exp_name, "input.txt")
    if not os.path.exists(data_path):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", data_path)
    with open(data_path, 'r', encoding='utf-8') as f: text = f.read()
    
    tokenizer = BasicTokenizer()
    tokenizer.train(text, args.vocab_size)
    tokenizer.save(os.path.join(args.exp_name, "tokenizer.json"))
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"Data: {len(train_data)} train, {len(val_data)} val tokens")
    
    # Model
    config = GPTConfig(
        vocab_size=args.vocab_size, d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, d_ff=args.d_model*4, block_size=args.block_size,
        dropout=args.dropout, layer_type=args.layer_type, weight_type=args.weight_type,
        num_waves=args.num_waves, num_harmonics=args.num_harmonics,
        init_mode=args.init_mode, activation_type=args.activation_type
    )
    model = SpectralGPT(config).to(args.device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.device == 'cuda'))
    
    # Training
    model.train()
    start_time = time.time()
    
    # Initialize curriculum
    if args.use_curriculum and args.weight_type == 'wave':
        model.freeze_high_frequencies(0.2)
        print("🎓 Curriculum: High frequencies frozen")
        
    for step in range(args.max_steps):
        # Curriculum Progress
        progress = min(1.0, step / (0.7 * args.max_steps))
        
        # Wave Weight Curriculum
        if args.use_curriculum and args.weight_type == 'wave':
            model.progressive_unfreeze(step, args.max_steps, 'linear')
            
        # LR Schedule
        lr = get_lr(step, args.max_steps, args.lr, args.warmup_steps)
        for g in optimizer.param_groups: g['lr'] = lr
        
        # Forward
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for _ in range(args.grad_accum_steps):
            x, y = get_batch(train_data, args.batch_size, args.block_size, args.device)
            if args.device == 'cuda':
                with torch.amp.autocast('cuda'): 
                    _, loss = model(x, y, progress=progress)
            else: 
                _, loss = model(x, y, progress=progress)
            
            # Add Frequency Consistency Loss
            if args.use_freq_consistency and args.weight_type == 'wave':
                loss += args.freq_consistency_weight * get_freq_consistency_loss(model)
            
            # Add Wave Function Collapse Penalty (Sadar Effect)
            if args.use_collapse:
                # Anneal penalty: Strong at start (superposition), weaker later? 
                # Or constant to force sparsity? Let's keep it constant for crystallization.
                loss += args.collapse_weight * get_collapse_penalty(model)
                
            loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Hamiltonian Descent: Enforce Energy Conservation
        if args.use_hamiltonian:
            model.constrain_energy()
        
        if step % 50 == 0:
            dt = time.time() - start_time
            print(f"Step {step:4d} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | Prog: {progress:.2f}")
            start_time = time.time()
            
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size, args.device)
            print(f"EVAL: Train {losses['train']:.4f} | Val {losses['val']:.4f}")
            
    # Generation
    print("\nGENERATION TEST:")
    model.eval()
    ctx = torch.tensor([tokenizer.encode("\n")], dtype=torch.long, device=args.device)
    out = model.generate(ctx, 400, temperature=0.8)
    print(tokenizer.decode(out[0].tolist()))

if __name__ == "__main__":
    main()
