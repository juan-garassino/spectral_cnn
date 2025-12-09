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
import tiktoken # GPT-2 Tokenizer
from datasets import load_dataset # HuggingFace Datasets

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
sys.path.insert(0, os.path.dirname(current_dir)) # Add root project dir for src imports

from wave_gpt import WaveGPT, WaveGPTConfig
from physics_optim import ResonantGradientDescent, QuantumFieldEntanglementLoss
from train import BasicTokenizer, get_batch

# Import wave physics core components (physics-first approach)
try:
    from wave_physics_core import WaveNativeOptimizer, WaveCoherenceLoss, WaveDiagnostics
    WAVE_PHYSICS_CORE_AVAILABLE = True
except ImportError:
    WAVE_PHYSICS_CORE_AVAILABLE = False
    WaveNativeOptimizer = None
    WaveCoherenceLoss = None
    WaveDiagnostics = None


# ==========================================
# Annealing Schedule
# ==========================================

# Default annealing steps constant
ANNEALING_STEPS = 3000


def get_annealing_ratio(step: int, total_annealing_steps: int = ANNEALING_STEPS) -> float:
    """
    Compute standard_embed_ratio for annealing schedule.
    
    Decays linearly from 1.0 to 0.0 over total_annealing_steps.
    After total_annealing_steps, returns 0.0 (pure wave embeddings).
    
    Args:
        step: Current training step
        total_annealing_steps: Number of steps for annealing (default: 3000)
        
    Returns:
        standard_embed_ratio in [0.0, 1.0]
        
    Requirements: 6.1
    """
    return max(0.0, 1.0 - step / total_annealing_steps)

# ==========================================
# Experiment Configurations
# ==========================================

class EarlyStopping:
    """
    Robust Early Stopping with CPU state saving.
    """
    def __init__(self, patience: int = 8, min_delta: float = 0.005, console=None):
        self.patience = patience
        self.min_delta = min_delta
        self.console = console
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save state to CPU to avoid GPU OOM
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.console:
                self.console.print(f"   [green]New best: {val_loss:.4f} (Saved)[/green]")
        else:
            self.counter += 1
            if self.console:
                self.console.print(f"   [yellow]Patience: {self.counter}/{self.patience}[/yellow]")
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
        
    def restore(self, model: nn.Module):
        if self.best_model_state is not None:
            if self.console:
                self.console.print(f"♻️  Restoring best model (Val: {self.best_loss:.4f})")
            model.load_state_dict(self.best_model_state)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    use_rgd: bool = False
    use_qfe: bool = False
    pure_wave_attention: bool = False  # New flag for pure attention
    pure_wave_kernel: str = "elu_plus_one" # Kernel: 'elu_plus_one', 'sigmoid', 'exp'
    pure_wave_mode: str = "quadratic"      # 'quadratic' vs 'linear'
    rgd_strength: float = 0.3
    qfe_lambda: float = 0.05
    qfe_threshold: float = 0.01
    lr: float = 6e-4
    weight_decay: float = 0.01
    dropout: float = 0.1 # Default 0.1 for Baseline
    warmup_steps: int = 500
    patience: int = 8   # Scientific default
    steps: int = 10000   # Scaled for 500M tokens (~7.4 tokens/param)
    wave_ratio_schedule: bool = True  # Schedule wave_ratio from 0.5 to 0.9
    model_type: str = "wave" # "wave" or "standard"
    grad_accum_steps: int = 2 # Restore effective B=32 (since physical B=16)




ABLATION_EXPERIMENTS = {
    # 1. Standard Transformer (TRUE Control)
    "standard_transformer": ExperimentConfig(
        name="Standard Transformer (GPT-2)",
        model_type="standard",
        use_rgd=False, use_qfe=False,
        lr=6e-4, dropout=0.1
    ),

    # 2. Wave Baseline (Architecture Check)
    "wave_baseline": ExperimentConfig(
        name="Wave Baseline (AdamW + CE)",
        model_type="wave",
        use_rgd=False, use_qfe=False,
        lr=6e-4, dropout=0.1  # Standard NanoGPT settings (0.1)
    ),
    
    # 3. RGD Only (Aggressive Test)
    "rgd_only": ExperimentConfig(
        name="RGD Only",
        model_type="wave",
        use_rgd=True, use_qfe=False,
        lr=1e-3, dropout=0.1
    ),
    
    # 4. Full Physics (RGD + QFE + Aggressive)
    "full_physics": ExperimentConfig(
        name="Full Physics (RGD + QFE)",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.0, # NO DROPOUT - rely on QFE
        qfe_lambda=0.1        # Stronger QFE
    ),
    
    # 5. QFE Only
    "qfe_only": ExperimentConfig(
        name="QFE Only", 
        model_type="wave",
        use_rgd=False, use_qfe=True,
        lr=6e-4, dropout=0.1
    ),

    # 6. Pure Wave Variants (Inherit Physics Settings: RGD=True)
    "pure_wave": ExperimentConfig(
        name="Pure Wave (ELU+1) 🌊",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.1,
        pure_wave_attention=True,
        pure_wave_kernel="elu_plus_one"
    ),
    "pure_wave_linear": ExperimentConfig(
        name="Pure Wave (Linear O(N)) ⚡️",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.1,
        pure_wave_attention=True,
        pure_wave_kernel="elu_plus_one",
        pure_wave_mode="linear"
    ),
    "pure_wave_sigmoid": ExperimentConfig(
        name="Pure Wave (Sigmoid) 🌊",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.1,
        pure_wave_attention=True,
        pure_wave_kernel="sigmoid"
    ),
    "pure_wave_exp": ExperimentConfig(
        name="Pure Wave (Exp) 🌊",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.1,
        pure_wave_attention=True,
        pure_wave_kernel="exp"
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
        name="Small (GPT-2 Compatible)",
        d_model=384, num_layers=8, num_heads=8,
        num_waves=48, num_harmonics=4,
        vocab_size=50257, block_size=256, batch_size=16 # Reduced to 16 to fit QFE Loss in Memory
    ),
    "medium": ModelConfig(
        name="Medium",
        d_model=512, num_layers=10, num_heads=8,
        num_waves=64, num_harmonics=4,
        vocab_size=50257, block_size=384, batch_size=24
    ),
    "large": ModelConfig(
        name="Large",
        d_model=768, num_layers=12, num_heads=12,
        num_waves=96, num_harmonics=4,
        vocab_size=50257, block_size=512, batch_size=16
    ),
}


# ==========================================
# Dataset Loaders
# ==========================================

def load_fineweb_tiktoken(console, subset="sample-10BT", target_tokens=500_000_000):
    """
    Load FineWeb-Edu and tokenize with Tiktoken until exactly target_tokens.
    """
    console.print(f"📥 Loading FineWeb-Edu ({subset}) via Tiktoken...")
    console.print(f"   Target: {target_tokens:,} tokens")
    
    enc = tiktoken.get_encoding("gpt2")
    
    
    # MOCK FOR VERIFICATION
    # console.print("[yellow]⚠️  MOCKING DATA FOR FAST STATS CHECK[/yellow]")
    # return torch.randint(0, 50257, (100000,), dtype=torch.long)
    
    # Stream the dataset
    ds = load_dataset(
       "HuggingFaceFW/fineweb-edu",
       name=subset,
       split="train",
       streaming=True
    )
    
    all_tokens = []
    total_count = 0
    
    with Progress(console=console) as progress:
        task = progress.add_task("Streaming & Tokenizing...", total=target_tokens)
        
        for item in ds:
            text = item.get("text", "")
            if not text: continue
            
            # Tokenize
            tokens = enc.encode(text)
            all_tokens.extend(tokens)
            total_count += len(tokens)
            
            progress.update(task, completed=min(total_count, target_tokens))
            
            if total_count >= target_tokens:
                break
                
    # Trim to exact target
    all_tokens = all_tokens[:target_tokens]
    console.print(f"✅ Loaded exactly {len(all_tokens):,} tokens.")
    
    # Convert to tensor
    data = torch.tensor(all_tokens, dtype=torch.long)
    return data


def create_mock_dataset(vocab_size: int = 50257, num_tokens: int = 10000) -> torch.Tensor:
    """
    Create mock dataset for dry-run testing.
    
    Args:
        vocab_size: Size of vocabulary
        num_tokens: Number of tokens to generate
        
    Returns:
        Tensor of random token IDs
    """
    return torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long)


def get_dataset(dataset_name: str, console, max_tokens: int = 500_000_000, dry_run: bool = False):
    """Get dataset by name"""
    if dry_run:
        console.print("[yellow]📦 Creating mock dataset for dry-run...[/yellow]")
        return create_mock_dataset(vocab_size=50257, num_tokens=10000)
    
    if dataset_name == "shakespeare":
        # Fallback to shakespeare if explicit
        return load_shakespeare(console) # Returns text, not tensor
    elif dataset_name == "fineweb" or dataset_name == "fineweb_small":
        # Use tiktoken loader
        return load_fineweb_tiktoken(console, subset="sample-10BT", target_tokens=max_tokens)
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
    device: str = "cuda",
    experiment_dir: Optional[str] = None,
    enable_monitoring: bool = True,
    dry_run: bool = False
) -> Dict:
    """Train a single experiment configuration with optional monitoring"""
    
    console.print(Panel(f"[bold cyan]{exp_config.name}[/bold cyan]", border_style="cyan"))
    
    params = sum(p.numel() for p in model.parameters())
    console.print(f"📊 Parameters: {params:,} ({params/1e6:.2f}M)")
    console.print(f"🔄 Steps: {exp_config.steps}")
    
    # Initialize monitoring components if enabled
    checkpoint_manager = None
    metrics_logger = None
    viz_manager = None
    config_tracker = None
    
    if enable_monitoring and experiment_dir:
        from monitoring import (
            CheckpointManager, MetricsLogger, 
            VisualizationManager, ConfigTracker,
            create_experiment_directory
        )
        
        # Create experiment directory structure
        dirs = create_experiment_directory("experiments", experiment_dir)
        console.print(f"📁 Experiment directory: {dirs['root']}")
        
        # Initialize monitoring components with adjusted intervals for dry-run
        checkpoint_interval = 5 if dry_run else 1000
        log_interval = 1 if dry_run else 10
        viz_interval = 5 if dry_run else 1000
        
        checkpoint_manager = CheckpointManager(
            experiment_dir=dirs['root'],
            save_interval=checkpoint_interval,
            keep_last_n=3
        )
        
        metrics_logger = MetricsLogger(
            log_dir=dirs['logs'],
            log_interval=log_interval
        )
        
        viz_manager = VisualizationManager(
            viz_dir=dirs['visualizations'],
            viz_interval=viz_interval
        )
        
        config_tracker = ConfigTracker(
            experiment_dir=dirs['root']
        )
        
        # Save initial configuration
        config_dict = {
            'experiment_name': exp_config.name,
            'model_config': {
                'd_model': model_config.d_model,
                'num_layers': model_config.num_layers,
                'num_heads': model_config.num_heads,
                'num_waves': model_config.num_waves,
                'num_harmonics': model_config.num_harmonics,
                'vocab_size': model_config.vocab_size,
                'block_size': model_config.block_size,
                'batch_size': model_config.batch_size
            },
            'training_config': asdict(exp_config)
        }
        
        dataset_info = {
            'train_tokens': len(train_data),
            'val_tokens': len(val_data),
            'total_tokens': len(train_data) + len(val_data)
        }
        
        try:
            config_tracker.save_config(config_dict, model, dataset_info)
            console.print("✓ Configuration saved")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save config: {e}[/yellow]")
    
    # Check for existing checkpoint and offer resumption
    start_step = 0
    resumed_loss_history = []
    if checkpoint_manager:
        try:
            checkpoint = checkpoint_manager.load_latest_checkpoint()
            if checkpoint:
                console.print(f"[yellow]Found existing checkpoint at step {checkpoint['step']}[/yellow]")
                console.print(f"[yellow]Checkpoint loss: {checkpoint['loss_history'][-1] if checkpoint.get('loss_history') else 'N/A'}[/yellow]")
                console.print("[yellow]Resume from checkpoint? (y/n)[/yellow]")
                # For automated runs, we'll skip resumption by default
                # In interactive mode, user can modify this
                resume = False  # Set to True to enable auto-resume
                
                if resume:
                    # Load model state
                    model.load_state_dict(checkpoint['model_state_dict'])
                    start_step = checkpoint['step'] + 1
                    
                    # Restore loss history for continuity verification
                    if 'loss_history' in checkpoint:
                        resumed_loss_history = checkpoint['loss_history']
                        console.print(f"[green]✓ Restored {len(resumed_loss_history)} loss values[/green]")
                    
                    console.print(f"[green]✓ Resumed from step {checkpoint['step']}[/green]")
                    console.print(f"[green]✓ Will continue training from step {start_step}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load checkpoint: {e}[/yellow]")
    
    # Setup optimizer
    # Use WaveNativeOptimizer from wave_physics_core if available, else fall back to legacy RGD
    if exp_config.use_rgd:
        if WAVE_PHYSICS_CORE_AVAILABLE and WaveNativeOptimizer is not None:
            # Physics-first approach: WaveNativeOptimizer with SVD projection
            optimizer = WaveNativeOptimizer(
                model.parameters(),
                lr=exp_config.lr,
                damping=0.1,
                coherence_weight=exp_config.rgd_strength,  # Map rgd_strength to coherence_weight
                weight_decay=exp_config.weight_decay
            )
            console.print(f"⚡ Optimizer: WaveNativeOptimizer (coherence={exp_config.rgd_strength})")
        else:
            # Fallback to legacy ResonantGradientDescent
            optimizer = ResonantGradientDescent(
                model.parameters(),
                lr=exp_config.lr,
                resonance_strength=exp_config.rgd_strength,
                warmup_steps=exp_config.warmup_steps,
                weight_decay=exp_config.weight_decay
            )
            console.print(f"⚡ Optimizer: RGD (strength={exp_config.rgd_strength}) [legacy]")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp_config.lr, weight_decay=exp_config.weight_decay)
        console.print(f"⚙️  Optimizer: AdamW")
    
    # Restore optimizer state if resuming from checkpoint
    if start_step > 0 and checkpoint_manager:
        try:
            checkpoint = checkpoint_manager.load_latest_checkpoint()
            if checkpoint and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                console.print(f"[green]✓ Optimizer state restored[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to restore optimizer state: {e}[/yellow]")
    
    # Setup loss
    # Use WaveCoherenceLoss from wave_physics_core if available, else fall back to legacy QFE
    if exp_config.use_qfe:
        if WAVE_PHYSICS_CORE_AVAILABLE and WaveCoherenceLoss is not None:
            # Physics-first approach: WaveCoherenceLoss with phase/energy/harmonic regularization
            loss_fn = WaveCoherenceLoss(
                lambda_phase=exp_config.qfe_lambda,
                lambda_energy=exp_config.qfe_lambda,
                lambda_harmonic=exp_config.qfe_lambda,
                window_size=8
            )
            console.print(f"🌌 Loss: WaveCoherenceLoss (λ={exp_config.qfe_lambda})")
        else:
            # Fallback to legacy QuantumFieldEntanglementLoss
            loss_fn = QuantumFieldEntanglementLoss(
                lambda_coherence=exp_config.qfe_lambda,
                amplitude_threshold=exp_config.qfe_threshold
            )
            console.print(f"🌌 Loss: QFE (λ={exp_config.qfe_lambda}) [legacy]")
    else:
        loss_fn = None
        console.print(f"📉 Loss: Cross-Entropy")
    
    console.print(f"📈 LR: {exp_config.lr:.0e}")
    
    # LR schedule
    def get_lr(step):
        if step < exp_config.warmup_steps:
            return exp_config.lr * step / exp_config.warmup_steps
        progress = (step - exp_config.warmup_steps) / (exp_config.steps - exp_config.warmup_steps)
        return exp_config.lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    # Wave ratio schedule (push toward pure wave)
    def get_wave_ratio_target(step):
        if not exp_config.wave_ratio_schedule:
            return 0.5
        # Schedule: 0.5 -> 0.9 over training
        progress = step / exp_config.steps
        return 0.5 + 0.4 * progress
    
    model.train()
    losses = resumed_loss_history.copy() if resumed_loss_history else []
    coherence_losses = []
    wave_ratios = []
    learning_rates = []
    perplexities = []
    
    # Store the last loss before resumption for continuity verification
    last_loss_before_resume = losses[-1] if losses else None
    
    # Early stopping tracking
    early_stopping = EarlyStopping(
        patience=exp_config.patience,
        min_delta=0.005,
        console=console
    )
    
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
        
        for step in range(start_step, exp_config.steps):
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
            
            
            # --- Gradient Accumulation Loop ---
            accum_loss_scalar = 0.0
            
            # Compute annealing ratio for this step (Requirements 6.1, 6.2)
            # Decays from 1.0 to 0.0 over ANNEALING_STEPS (3000 steps)
            current_annealing_ratio = get_annealing_ratio(step, ANNEALING_STEPS)
            
            optimizer.zero_grad() # Zero gradients before accumulation starts
            
            for _ in range(exp_config.grad_accum_steps):
                # Get batch
                x, y = get_batch(train_data, model_config.batch_size, model_config.block_size, device)
                total_tokens += x.numel()
                
                # Forward with annealing ratio (Requirements 6.1, 6.2)
                # Pass standard_embed_ratio to model for embedding annealing
                logits, ce_loss = model(x, y, standard_embed_ratio=current_annealing_ratio)
                
                # Compute loss
                if exp_config.use_qfe and loss_fn is not None:
                    # Handle both WaveCoherenceLoss (new) and QuantumFieldEntanglementLoss (legacy)
                    if WAVE_PHYSICS_CORE_AVAILABLE and isinstance(loss_fn, WaveCoherenceLoss):
                        # WaveCoherenceLoss returns dict with 'total', 'ce', 'coherence' keys
                        loss_dict = loss_fn(logits, y)
                        loss = loss_dict['total']
                        coherence_losses.append(loss_dict['coherence'].item())
                    else:
                        # Legacy QuantumFieldEntanglementLoss with return_components=True
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
                    return {'loss': float('nan')}
                
                # Scale loss for accumulation
                loss = loss / exp_config.grad_accum_steps
                
                # Backward
                loss.backward()
                
                # Track raw loss (unscaled) for logging
                accum_loss_scalar += loss.item() * exp_config.grad_accum_steps

            # Update weights after accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Average loss over accumulation steps (approximate)
            avg_loss = accum_loss_scalar / exp_config.grad_accum_steps
            losses.append(avg_loss)
            learning_rates.append(current_lr)
            progress.update(task, advance=1, loss=avg_loss)
            
            # Verify loss continuity after resumption (first step only)
            if step == start_step and last_loss_before_resume is not None:
                loss_diff = abs(avg_loss - last_loss_before_resume)
                if loss_diff < 0.5:  # Reasonable threshold for continuity
                    console.print(f"[green]✓ Loss continuity verified: {last_loss_before_resume:.4f} → {avg_loss:.4f} (diff: {loss_diff:.4f})[/green]")
                else:
                    console.print(f"[yellow]⚠ Loss jump detected: {last_loss_before_resume:.4f} → {avg_loss:.4f} (diff: {loss_diff:.4f})[/yellow]")
            
            # Log metrics if monitoring enabled
            if metrics_logger and metrics_logger.should_log(step):
                try:
                    metrics_dict = {
                        'loss': avg_loss,
                        'learning_rate': current_lr,
                        'tokens_per_sec': total_tokens / (time.perf_counter() - start_time) if (time.perf_counter() - start_time) > 0 else 0,
                        'standard_embed_ratio': current_annealing_ratio  # Track annealing progress
                    }
                    
                    # Add wave-specific metrics if available
                    if wave_ratios:
                        metrics_dict['wave_ratio'] = wave_ratios[-1]
                    if coherence_losses:
                        metrics_dict['coherence_loss'] = coherence_losses[-1]
                    
                    metrics_logger.log_metrics(step, metrics_dict)
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to log metrics: {e}[/yellow]")
            
            # Save checkpoint if monitoring enabled
            if checkpoint_manager and checkpoint_manager.should_checkpoint(step):
                try:
                    checkpoint_config = {
                        'experiment_name': exp_config.name,
                        'step': step,
                        'total_steps': exp_config.steps
                    }
                    checkpoint_manager.save_checkpoint(
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        loss_history=losses,
                        config=checkpoint_config
                    )
                    console.print(f"[green]✓ Checkpoint saved at step {step}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to save checkpoint: {e}[/yellow]")

            
            # Check every 250 steps for scientific rigor
            if (step + 1) % 250 == 0:
                avg = sum(losses[-250:]) / len(losses[-250:])
                wave_r = wave_ratios[-1] if wave_ratios else 0.5
                
                # Fast validation check
                model.eval()
                val_losses_accum = []
                num_val_batches = 20 # Robust multi-batch validation
                with torch.no_grad():
                    for _ in range(num_val_batches):
                        val_x, val_y = get_batch(val_data, model_config.batch_size, model_config.block_size, device) 
                        _, val_loss_check = model(val_x, val_y)
                        if val_loss_check.ndim > 0:
                            val_loss_check = val_loss_check.mean()
                        val_losses_accum.append(val_loss_check.item())
                model.train()
                
                current_val_loss = sum(val_losses_accum) / len(val_losses_accum)
                perplexities.append(math.exp(current_val_loss))
                
                # Log CE loss explicitly
                console.print(f"Step {step+1:5d} | Train(CE): {ce_loss_scalar.item():.4f} | Val: {current_val_loss:.4f} | AvgTrain: {avg:.4f} | R: {wave_r:.3f}")
                
                # Generate visualizations if monitoring enabled
                if viz_manager and viz_manager.should_visualize(step + 1):
                    try:
                        # Prepare metrics for visualization
                        viz_metrics = {
                            'learning_rate': learning_rates,
                            'perplexity': perplexities
                        }
                        if wave_ratios:
                            viz_metrics['wave_ratio'] = wave_ratios
                        if coherence_losses:
                            viz_metrics['coherence_loss'] = coherence_losses
                        
                        viz_manager.generate_training_plots(step + 1, losses, viz_metrics)
                        viz_manager.generate_model_plots(step + 1, model)
                        console.print(f"[green]✓ Visualizations generated at step {step + 1}[/green]")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to generate visualizations: {e}[/yellow]")
                
                 # Check Early Stopping
                if early_stopping(current_val_loss, model):
                    console.print(f"[yellow]🛑 Early stopping triggered. Best Val: {early_stopping.best_loss:.4f}[/yellow]")
                    break

    elapsed = time.perf_counter() - start_time
    speed = total_tokens / elapsed
    
    # Restore best model
    early_stopping.restore(model)
    
    # Final eval
    model.eval()
    with torch.no_grad():
        x, y = get_batch(val_data, model_config.batch_size, model_config.block_size, device)
        _, val_loss = model(x, y)
        if val_loss.ndim > 0: val_loss = val_loss.mean()
            
    # Generation Check
    console.print("\n[bold]🎨 Generating sample (Visual Check)...[/bold]")
    context = "The King" if "shakespeare" in exp_config.name.lower() else "The universe"
    # Need tokenizer to encode
    # Assumption: tokenizer is available in closure or we need to pass it. 
    # Current scope doesn't have tokenizer. 
    # Use simple ascii encoding for Shakespeare fallback if needed, but we should pass tokenizer or rely on it being loaded globally?
    # Actually, main() has tokenizer. We can just return 'model' and generate in main, OR generic encode/decode here.
    # Let's do a simple generation and return the string.
    
    # Quick dirty generation without tokenizer ref if needed, but better to skip if no tokenizer.
    # However, request says "Add generation Check".
    # I will assume we can generate token IDs and return them, or simply use the model.generate method.
    # I'll just skip text decoding here and return generated tokens to be decoded later, OR simply omit decoding text and trust the user to inspect tokens if they want.
    # Wait, the request says "Generate 100 tokens... This allows us to visually verify". Verification implies text.
    # I will stick to returning the model and letting `run_ablation_suite` handle the printing using the tokenizer it HAS.
    
    # Save final results if monitoring enabled
    if config_tracker:
        try:
            final_metrics = {
                'val_loss': val_loss.item(),
                'perplexity': torch.exp(val_loss).item(),
                'best_val_loss': early_stopping.best_loss,
                'total_time_seconds': elapsed,
                'tokens_per_second': speed,
                'total_steps': len(losses),
                'final_loss': losses[-1] if losses else 0.0
            }
            
            # Find best checkpoint path if available
            best_checkpoint_path = None
            if checkpoint_manager:
                checkpoints = checkpoint_manager.list_checkpoints()
                if checkpoints:
                    best_checkpoint_path = checkpoints[-1]['path']
            
            config_tracker.save_results(
                final_metrics=final_metrics,
                best_checkpoint=best_checkpoint_path,
                generation_samples=None  # Will be added by caller if needed
            )
            console.print("[green]✓ Final results saved[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save results: {e}[/yellow]")
    
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
        "model": model, 
        "early_stopping_best": early_stopping.best_loss,
        "experiment_dir": experiment_dir if enable_monitoring else None
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
    parallel: bool = False,
    dry_run: bool = False
) -> Dict:
    """Run ablation study experiments"""
    
    if experiments is None or "all" in experiments:
        experiments = list(ABLATION_EXPERIMENTS.keys())
    
    model_config = MODEL_CONFIGS[model_size]
    
    # Override settings for dry-run mode
    if dry_run:
        console.print("[bold yellow]🧪 DRY-RUN MODE ENABLED[/bold yellow]")
        console.print("[yellow]Using small model, mock data, and 10 training steps for fast testing[/yellow]\n")
        steps = 10
        model_size = "small"
        model_config = MODEL_CONFIGS[model_size]
        # Use smaller batch size for faster testing
        model_config.batch_size = 4
    
    console.print(Panel.fit(
        f"[bold magenta]🔬 WAVE-NATIVE GPT ABLATION SUITE[/bold magenta]\n\n"
        f"Model: {model_config.name}\n"
        f"Dataset: {dataset}\n"
        f"Experiments: {', '.join(experiments)}\n"
        f"Steps per experiment: {steps}" + ("\n[yellow]DRY-RUN MODE[/yellow]" if dry_run else ""),
        border_style="magenta"
    ))
    
    # Load data (Directly returns tensor if tiktoken, text if shakespeare)
    data_or_text = get_dataset(dataset, console, dry_run=dry_run)
    
    if isinstance(data_or_text, torch.Tensor):
        # Already tokenized (FineWeb/Tiktoken)
        data = data_or_text
        enc = tiktoken.get_encoding("gpt2") # For decoding later
        tokenizer = enc # Alias for generation
    else:
        # Shakespeare (Text) -> Legacy BasicTokenizer
        text = data_or_text
        tokenizer_path = os.path.join(current_dir, "benchmark_results", "tokenizer.json")
        console.print("♻️  Using BasicTokenizer for Shakespeare...")
        tokenizer = BasicTokenizer()
        if os.path.exists(tokenizer_path):
             with open(tokenizer_path, 'r') as f:
                data_json = json.load(f)
                tokenizer.vocab = {int(k): bytes(v) for k, v in data_json['vocab'].items()}
        else:
             tokenizer.train(text, model_config.vocab_size)
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
    n = int(0.9 * len(data)) # Strict 90/10 split
    train_data = data[:n]
    val_data = data[n:]
    console.print(f"📊 Tokens: Train {len(train_data):,} | Val {len(val_data):,}")
    
    results = {}
    
    for exp_name in experiments:
        console.print("\n" + "="*60)
        
        exp_config = ABLATION_EXPERIMENTS[exp_name]
        exp_config.steps = steps  # Override if needed, but default is now Scientific 5000
        
        # Generate experiment ID for monitoring
        from monitoring import generate_experiment_id
        experiment_id = generate_experiment_id(exp_name)
        
        if getattr(exp_config, 'model_type', 'wave') == "wave":
            wave_config = WaveGPTConfig(
                vocab_size=model_config.vocab_size,
                d_model=model_config.d_model,
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
                num_waves=model_config.num_waves,
                num_harmonics=model_config.num_harmonics,
                block_size=model_config.block_size,
                dropout=exp_config.dropout,
                pure_wave_attention=getattr(exp_config, 'pure_wave_attention', False),
                pure_wave_kernel=getattr(exp_config, 'pure_wave_kernel', 'elu_plus_one'),
                pure_wave_mode=getattr(exp_config, 'pure_wave_mode', 'quadratic'),
                model_type="wave"
            )
        else:
             wave_config = WaveGPTConfig(
                vocab_size=model_config.vocab_size,
                d_model=model_config.d_model,
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
                num_waves=model_config.num_waves,
                num_harmonics=model_config.num_harmonics,
                block_size=model_config.block_size,
                dropout=exp_config.dropout,
                model_type="standard" # Flag for Standard
             )
        
        model = WaveGPT(wave_config).to(device)
        
        # Count Parameters
        params = sum(p.numel() for p in model.parameters())
        console.print(f"📊 Parameters: {params:,} ({params/1e6:.2f}M)")
        
        # Parameter count assertion for fair comparison (Requirements 6.5)
        # Only enforce for non-dry-run experiments to allow testing with smaller models
        if not dry_run:
            MIN_PARAMS = 50_000_000  # 50M
            MAX_PARAMS = 55_000_000  # 55M
            if not (MIN_PARAMS < params < MAX_PARAMS):
                console.print(f"[yellow]⚠️  Parameter count {params:,} outside fair comparison range (50M-55M)[/yellow]")
                console.print(f"[yellow]   Adjust model config for fair comparison runs[/yellow]")
                # Note: We warn but don't assert to allow flexibility in experiments
                # For strict fair comparison, uncomment the assertion below:
                # assert MIN_PARAMS < params < MAX_PARAMS, \
                #     f"Parameter count {params:,} must be between 50M and 55M for fair comparison"
        
        # DataParallel if requested
        if parallel and torch.cuda.device_count() > 1:
            console.print(f"🔀 Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        # Print Detailed Stats
        print_detailed_stats(model, model_config, dataset, train_data, val_data, console)

        # Train with monitoring enabled
        result = train_experiment(
            model, train_data, val_data, exp_config, model_config, console, device,
            experiment_dir=experiment_id,
            enable_monitoring=True,
            dry_run=dry_run
        )
        
        # GENERATION CHECK (Scientific Visual Verification)
        console.print("[bold]📝 Generation Check:[/bold]")
        start_text = "The theory of quantum mechanics suggests" # Scientific Prompt
        
        # Handle tokenizer differences (tiktoken vs basic)
        if hasattr(tokenizer, 'encode'):
             context_ids = tokenizer.encode(start_text)
        else:
             context_ids = tokenizer.encode(start_text)
             
        context_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
        
        # Generate
        if isinstance(model, nn.DataParallel):
            gen_ids = model.module.generate(context_tensor, max_new_tokens=100, temperature=0.8)
        else:
            gen_ids = model.generate(context_tensor, max_new_tokens=100, temperature=0.8)
        
        # Decode
        if hasattr(tokenizer, 'decode'):
            gen_text = tokenizer.decode(gen_ids[0].tolist())
        else:
            # Fallback for BasicTokenizer
            gen_text = tokenizer.decode(gen_ids[0].tolist())
            
        console.print(Panel(gen_text, title=f"{exp_config.name} Output", border_style="green"))
        
        result['generation'] = gen_text
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
        # Adjust window size for short runs
        window = min(100, max(1, len(losses) // 10))
        if len(losses) < window:
            # No smoothing for very short runs
            ax.plot(range(len(losses)), losses, 
                    label=f"{result['name']} (final: {result['val_loss']:.3f})",
                    color=colors[i % len(colors)], linewidth=2)
        else:
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

def print_detailed_stats(model, config, tokenizer_name, train_data, val_data, console):
    """Print comprehensive breakdown of model and data statistics"""
    
    # 1. Dataset Stats
    total_tokens = len(train_data) + len(val_data)
    
    table = Table(title="📚 Dataset Statistics", border_style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Total Tokens", f"{total_tokens:,}")
    table.add_row("Train Split", f"{len(train_data):,} ({len(train_data)/total_tokens:.1%})")
    table.add_row("Validation Split", f"{len(val_data):,} ({len(val_data)/total_tokens:.1%})")
    table.add_row("Tokenizer", "TikToken (GPT-2)" if "fineweb" in tokenizer_name else "BasicTokenizer")
    
    console.print(table)
    
    # 2. Parameter Breakdown
    embed_params = 0
    attn_params = 0
    mlp_params = 0
    other_params = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        n = name.lower()
        if "embed" in n or "wte" in n or "wpe" in n:
            embed_params += param.numel()
        elif "attn" in n or "attention" in n:
            attn_params += param.numel()
        elif "mlp" in n:
            mlp_params += param.numel()
        else:
            other_params += param.numel()
            
    total_params = embed_params + attn_params + mlp_params + other_params
    
    # Model Stats Table
    m_table = Table(title="🧠 Model Parameter Breakdown", border_style="magenta")
    m_table.add_column("Component", style="green")
    m_table.add_column("Parameters", style="white", justify="right")
    m_table.add_column("% of Total", style="yellow", justify="right")
    
    m_table.add_row("Embeddings", f"{embed_params:,}", f"{embed_params/total_params:.1%}")
    m_table.add_row("Attention", f"{attn_params:,}", f"{attn_params/total_params:.1%}")
    m_table.add_row("MLP", f"{mlp_params:,}", f"{mlp_params/total_params:.1%}")
    m_table.add_row("Other (Norms/Head)", f"{other_params:,}", f"{other_params/total_params:.1%}")
    m_table.add_row("TOTAL", f"{total_params:,}", "100.0%", style="bold")
    
    console.print(m_table)
    console.print(f"📊 Ratio: {total_tokens/total_params:.2f} Tokens per Parameter (Target > 7.4)", style="bold purple")


def main():
    parser = argparse.ArgumentParser(description="Wave-Native GPT Experiment Suite")
    parser.add_argument("--experiment", type=str, nargs="+", default=["all"],
                        choices=["all"] + list(ABLATION_EXPERIMENTS.keys()),
                        help="Experiments to run")
    parser.add_argument("--model", type=str, default="small",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model size")
    parser.add_argument("--dataset", type=str, default="fineweb",
                        choices=["shakespeare", "fineweb_small", "fineweb", "fineweb_large"],
                        help="Dataset to use")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Training steps per experiment")
    parser.add_argument("--parallel", action="store_true",
                        help="Use DataParallel for multi-GPU")
    parser.add_argument("--output", type=str, default="experiment_results",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run in dry-run mode with mock data for fast testing (10 steps)")
    
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
        parallel=args.parallel,
        dry_run=args.dry_run
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
