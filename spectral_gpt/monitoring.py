"""
Experiment Monitoring Infrastructure for Spectral GPT

Provides checkpointing, metrics logging, visualization, and configuration tracking
for robust experiment management with interruption recovery.
"""

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Optimizer


# ==========================================
# Experiment ID Generation
# ==========================================

def get_git_hash() -> str:
    """Get current git commit hash (short form)"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def generate_experiment_id(experiment_name: str) -> str:
    """
    Generate unique experiment ID with timestamp and git hash.
    
    Format: {experiment_name}_{timestamp}_{git_hash}
    Example: wave_rgd_qfe_20241209_143022_a3f2b1c4
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Unique experiment ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = get_git_hash()
    return f"{experiment_name}_{timestamp}_{git_hash}"


def create_experiment_directory(base_dir: str, experiment_id: str) -> Dict[str, str]:
    """
    Create directory structure for experiment.
    
    Structure:
        experiments/{experiment_id}/
        ├── checkpoints/
        ├── logs/
        └── visualizations/
    
    Args:
        base_dir: Base directory for all experiments
        experiment_id: Unique experiment identifier
        
    Returns:
        Dictionary mapping directory types to paths
    """
    exp_dir = Path(base_dir) / experiment_id
    
    dirs = {
        'root': str(exp_dir),
        'checkpoints': str(exp_dir / 'checkpoints'),
        'logs': str(exp_dir / 'logs'),
        'visualizations': str(exp_dir / 'visualizations')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


# ==========================================
# CheckpointManager
# ==========================================

class CheckpointManager:
    """
    Manages model checkpointing during training with automatic cleanup and resumption support.
    
    Features:
    - Atomic writes using temporary files to prevent corruption
    - Automatic cleanup of old checkpoints based on retention policy
    - Symlink to latest checkpoint for easy resumption
    - Saves model state, optimizer state, step, loss history, and config
    
    Example:
        >>> manager = CheckpointManager(experiment_dir="experiments/exp_001", 
        ...                             save_interval=1000, keep_last_n=3)
        >>> 
        >>> # During training
        >>> if manager.should_checkpoint(step):
        ...     manager.save_checkpoint(step, model, optimizer, loss_history, config)
        >>> 
        >>> # Resume from interruption
        >>> checkpoint = manager.load_latest_checkpoint()
        >>> if checkpoint:
        ...     model.load_state_dict(checkpoint['model_state_dict'])
        ...     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ...     start_step = checkpoint['step'] + 1
    """
    
    def __init__(self, 
                 experiment_dir: str,
                 save_interval: int = 1000,
                 keep_last_n: int = 3):
        """
        Initialize CheckpointManager.
        
        Args:
            experiment_dir: Root directory for experiment
            save_interval: Steps between checkpoints
            keep_last_n: Number of recent checkpoints to retain
        """
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.latest_symlink = self.checkpoint_dir / 'checkpoint_latest.pt'
    
    def should_checkpoint(self, step: int) -> bool:
        """
        Check if current step requires checkpointing.
        
        Args:
            step: Current training step
            
        Returns:
            True if checkpoint should be saved
        """
        return step > 0 and step % self.save_interval == 0
    
    def save_checkpoint(self,
                       step: int,
                       model: nn.Module,
                       optimizer: Optimizer,
                       loss_history: List[float],
                       config: Dict) -> str:
        """
        Save checkpoint to disk with atomic write.
        
        Uses temporary file and atomic rename to prevent corruption from
        interrupted writes.
        
        Args:
            step: Current training step
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            loss_history: List of training losses
            config: Experiment configuration dictionary
            
        Returns:
            Path to saved checkpoint file
        """
        # Handle DataParallel models
        if isinstance(model, nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint_data = {
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'git_hash': get_git_hash()
        }
        
        # Checkpoint filename
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        temp_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt.tmp'
        
        # Atomic write: save to temp file first
        torch.save(checkpoint_data, temp_path)
        
        # Atomic rename (POSIX systems guarantee atomicity)
        temp_path.replace(checkpoint_path)
        
        # Update symlink to latest checkpoint
        if self.latest_symlink.exists() or self.latest_symlink.is_symlink():
            self.latest_symlink.unlink()
        
        # Create relative symlink
        try:
            self.latest_symlink.symlink_to(checkpoint_path.name)
        except OSError:
            # Fallback for systems that don't support symlinks (Windows)
            # Just copy the file
            import shutil
            shutil.copy2(checkpoint_path, self.latest_symlink)
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """
        Load most recent checkpoint if exists.
        
        Returns:
            Checkpoint dictionary or None if no checkpoint exists
        """
        if not self.latest_symlink.exists():
            return None
        
        try:
            checkpoint = torch.load(self.latest_symlink, map_location='cpu')
            return checkpoint
        except Exception as e:
            print(f"Warning: Failed to load checkpoint from {self.latest_symlink}: {e}")
            
            # Try to find the most recent checkpoint by step number
            checkpoints = sorted(
                self.checkpoint_dir.glob('checkpoint_step_*.pt'),
                key=lambda p: int(p.stem.split('_')[-1]),
                reverse=True
            )
            
            for checkpoint_path in checkpoints:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    print(f"Loaded checkpoint from {checkpoint_path}")
                    return checkpoint
                except Exception:
                    continue
            
            return None
    
    def load_checkpoint(self, step: int) -> Optional[Dict]:
        """
        Load specific checkpoint by step number.
        
        Args:
            step: Step number of checkpoint to load
            
        Returns:
            Checkpoint dictionary or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        
        if not checkpoint_path.exists():
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint
        except Exception as e:
            print(f"Warning: Failed to load checkpoint from {checkpoint_path}: {e}")
            return None
    
    def cleanup_old_checkpoints(self):
        """
        Remove checkpoints beyond retention limit.
        
        Keeps only the N most recent checkpoints based on step number.
        """
        # Get all checkpoint files (excluding symlink and temp files)
        checkpoints = sorted(
            [p for p in self.checkpoint_dir.glob('checkpoint_step_*.pt') 
             if not p.name.endswith('.tmp')],
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True
        )
        
        # Remove old checkpoints beyond retention limit
        for checkpoint_path in checkpoints[self.keep_last_n:]:
            try:
                checkpoint_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete old checkpoint {checkpoint_path}: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List of dictionaries containing checkpoint info
        """
        checkpoints = []
        
        for checkpoint_path in sorted(self.checkpoint_dir.glob('checkpoint_step_*.pt')):
            if checkpoint_path.name.endswith('.tmp'):
                continue
            
            try:
                step = int(checkpoint_path.stem.split('_')[-1])
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
                
                checkpoints.append({
                    'step': step,
                    'path': str(checkpoint_path),
                    'size_mb': size_mb,
                    'modified': mtime.isoformat()
                })
            except Exception:
                continue
        
        return sorted(checkpoints, key=lambda x: x['step'])


# ==========================================
# MetricsLogger
# ==========================================

class MetricsLogger:
    """
    Incrementally logs training metrics with immediate persistence.
    
    Features:
    - JSONL format (one JSON object per line) for easy streaming and parsing
    - Immediate buffer flush after each write to ensure persistence
    - Separate human-readable log file for debugging
    - Tracks step, loss, learning_rate, and model-specific metrics
    
    Example:
        >>> logger = MetricsLogger(log_dir="experiments/exp_001/logs", log_interval=10)
        >>> 
        >>> # During training
        >>> if logger.should_log(step):
        ...     metrics = {
        ...         'loss': 0.5,
        ...         'learning_rate': 0.001,
        ...         'wave_ratio': 0.8,
        ...         'perplexity': 1.65
        ...     }
        ...     logger.log_metrics(step, metrics)
        >>> 
        >>> # Load metrics for analysis
        >>> all_metrics = logger.load_metrics()
        >>> latest_step = logger.get_latest_step()
    """
    
    def __init__(self, log_dir: str, log_interval: int = 10):
        """
        Initialize MetricsLogger.
        
        Args:
            log_dir: Directory for log files
            log_interval: Steps between log entries
        """
        self.log_dir = Path(log_dir)
        self.log_interval = log_interval
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file paths
        self.metrics_file = self.log_dir / 'metrics.jsonl'
        self.training_log = self.log_dir / 'training.log'
        
        # Initialize log files if they don't exist
        if not self.metrics_file.exists():
            self.metrics_file.touch()
        if not self.training_log.exists():
            self.training_log.touch()
    
    def should_log(self, step: int) -> bool:
        """
        Check if current step requires logging.
        
        Args:
            step: Current training step
            
        Returns:
            True if metrics should be logged
        """
        return step % self.log_interval == 0
    
    def log_metrics(self,
                   step: int,
                   metrics: Dict[str, float],
                   flush: bool = True):
        """
        Append metrics to log file.
        
        Writes to both JSONL format (for programmatic analysis) and
        human-readable format (for debugging).
        
        Args:
            step: Current training step
            metrics: Dictionary of metric names to values
            flush: Whether to flush buffer immediately (default: True)
        """
        # Add timestamp and step to metrics
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Write to JSONL file with line buffering (buffering=1)
        with open(self.metrics_file, 'a', buffering=1) as f:
            f.write(json.dumps(log_entry) + '\n')
            if flush:
                f.flush()
        
        # Write to human-readable log
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics_str = ', '.join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in metrics.items()])
        log_line = f"[{timestamp_str}] Step {step}: {metrics_str}\n"
        
        with open(self.training_log, 'a', buffering=1) as f:
            f.write(log_line)
            if flush:
                f.flush()
    
    def load_metrics(self) -> List[Dict]:
        """
        Load all logged metrics from file.
        
        Returns:
            List of metric dictionaries, one per logged step
        """
        if not self.metrics_file.exists():
            return []
        
        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse metrics line: {line[:50]}... Error: {e}")
                        continue
        
        return metrics
    
    def get_latest_step(self) -> int:
        """
        Get the last logged step number.
        
        Returns:
            Last logged step, or 0 if no metrics logged yet
        """
        metrics = self.load_metrics()
        if not metrics:
            return 0
        
        return metrics[-1]['step']
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of logged metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        metrics = self.load_metrics()
        if not metrics:
            return {}
        
        # Extract numeric metrics
        numeric_keys = set()
        for entry in metrics:
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key not in ['step']:
                    numeric_keys.add(key)
        
        summary = {
            'total_steps': len(metrics),
            'first_step': metrics[0]['step'],
            'last_step': metrics[-1]['step']
        }
        
        # Compute statistics for each numeric metric
        for key in numeric_keys:
            values = [entry[key] for entry in metrics if key in entry and isinstance(entry[key], (int, float))]
            if values:
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_max'] = max(values)
                summary[f'{key}_mean'] = sum(values) / len(values)
                summary[f'{key}_final'] = values[-1]
        
        return summary


# ==========================================
# VisualizationManager
# ==========================================

class VisualizationManager:
    """
    Generates and saves visualizations during training.
    
    Features:
    - Timestamped filenames for tracking evolution over training
    - Separate methods for training dynamics vs model internals
    - Matplotlib with dark theme for consistency
    - Error handling for missing model attributes
    - Plots include: loss curves, frequency distributions, phase distributions, 
      harmonics, wave packets, interference patterns
    
    Example:
        >>> viz_manager = VisualizationManager(
        ...     viz_dir="experiments/exp_001/visualizations",
        ...     viz_interval=1000
        ... )
        >>> 
        >>> # During training
        >>> if viz_manager.should_visualize(step):
        ...     viz_manager.generate_training_plots(step, loss_history, metrics)
        ...     viz_manager.generate_model_plots(step, model)
    """
    
    def __init__(self,
                 viz_dir: str,
                 viz_interval: int = 1000):
        """
        Initialize VisualizationManager.
        
        Args:
            viz_dir: Directory for visualization files
            viz_interval: Steps between visualization generation
        """
        self.viz_dir = Path(viz_dir)
        self.viz_interval = viz_interval
        
        # Create visualization directory if it doesn't exist
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        self._setup_matplotlib_style()
    
    def _setup_matplotlib_style(self):
        """Configure matplotlib with dark theme and consistent styling"""
        try:
            import matplotlib.pyplot as plt
            
            # Use dark background style
            plt.style.use('dark_background')
            
            # Set default figure parameters
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['figure.dpi'] = 100
            plt.rcParams['savefig.dpi'] = 150
            plt.rcParams['savefig.bbox'] = 'tight'
            
            # Font settings
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            plt.rcParams['legend.fontsize'] = 9
            
            # Grid settings
            plt.rcParams['grid.alpha'] = 0.3
            plt.rcParams['grid.linestyle'] = '--'
            
        except Exception as e:
            print(f"Warning: Failed to setup matplotlib style: {e}")
    
    def should_visualize(self, step: int) -> bool:
        """
        Check if current step requires visualization.
        
        Args:
            step: Current training step
            
        Returns:
            True if visualizations should be generated
        """
        return step > 0 and step % self.viz_interval == 0
    
    def generate_training_plots(self,
                                step: int,
                                loss_history: List[float],
                                metrics: Dict[str, List[float]]):
        """
        Generate plots of training dynamics.
        
        Creates visualizations for:
        - Loss curves (training and validation if available)
        - Learning rate schedule
        - Other tracked metrics (perplexity, wave_ratio, etc.)
        
        Args:
            step: Current training step
            loss_history: List of loss values over training
            metrics: Dictionary mapping metric names to lists of values
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create figure with subplots
            num_metrics = len(metrics) + 1  # +1 for loss
            ncols = 2
            nrows = (num_metrics + 1) // 2
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
            if nrows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            # Plot 1: Loss curve
            ax = axes[0]
            steps = list(range(len(loss_history)))
            ax.plot(steps, loss_history, linewidth=2, color='cyan', label='Training Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training Loss (Step {step})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Plot other metrics
            plot_idx = 1
            for metric_name, metric_values in metrics.items():
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                steps = list(range(len(metric_values)))
                
                # Choose color based on metric type
                color = 'yellow' if 'lr' in metric_name.lower() else 'magenta'
                if 'perplexity' in metric_name.lower():
                    color = 'green'
                elif 'wave' in metric_name.lower():
                    color = 'orange'
                
                ax.plot(steps, metric_values, linewidth=2, color=color, label=metric_name)
                ax.set_xlabel('Step')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} (Step {step})')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                plot_idx += 1
            
            # Hide unused subplots
            for idx in range(plot_idx, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Training Dynamics - Step {step}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save with timestamped filename
            filename = f'training_dynamics_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved training plots: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to generate training plots: {e}")
    
    def generate_model_plots(self,
                            step: int,
                            model: nn.Module):
        """
        Generate plots of model internals (wave properties).
        
        Creates visualizations for:
        - Frequency distributions
        - Phase distributions
        - Harmonic amplitudes
        - Wave packets
        - Interference patterns
        
        Args:
            step: Current training step
            model: Model to visualize
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Check if model has wave properties
            has_embedding = hasattr(model, 'embedding')
            if not has_embedding:
                # Try to access through module (for DataParallel)
                if hasattr(model, 'module'):
                    model = model.module
                    has_embedding = hasattr(model, 'embedding')
            
            if not has_embedding:
                print(f"Warning: Model does not have 'embedding' attribute, skipping model plots")
                return
            
            embedding = model.embedding
            
            # Check for required wave attributes
            has_freqs = hasattr(embedding, 'base_freqs')
            has_phases = hasattr(embedding, 'phases')
            has_harmonics = hasattr(embedding, 'harmonic_amps')
            
            if not (has_freqs or has_phases or has_harmonics):
                print(f"Warning: Model embedding does not have wave attributes, skipping model plots")
                return
            
            # Generate frequency distribution plot
            if has_freqs:
                self._plot_frequencies(step, embedding)
            
            # Generate phase distribution plot
            if has_phases:
                self._plot_phases(step, embedding)
            
            # Generate harmonics plot
            if has_harmonics:
                self._plot_harmonics(step, embedding)
            
            # Generate wave packets plot (requires all attributes)
            if has_freqs and has_phases and has_harmonics:
                self._plot_wave_packets(step, embedding)
            
        except Exception as e:
            print(f"Warning: Failed to generate model plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_frequencies(self, step: int, embedding):
        """Plot frequency distribution across tokens and waves"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            freqs = embedding.base_freqs.detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Heatmap of frequencies (first 100 tokens)
            n_tokens = min(100, freqs.shape[0])
            im = axes[0].imshow(freqs[:n_tokens].T, aspect='auto', cmap='viridis')
            axes[0].set_xlabel('Token ID')
            axes[0].set_ylabel('Wave Component')
            axes[0].set_title(f'Base Frequencies per Token (Step {step})')
            plt.colorbar(im, ax=axes[0], label='Frequency (Hz)')
            
            # Plot 2: Histogram of all frequencies
            axes[1].hist(freqs.flatten(), bins=50, color='cyan', alpha=0.7, edgecolor='white')
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'Frequency Distribution (Step {step})')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f'frequencies_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved frequency plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot frequencies: {e}")
    
    def _plot_phases(self, step: int, embedding):
        """Plot phase distribution across tokens and waves"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            phases = embedding.phases.detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Heatmap of phases (first 100 tokens)
            n_tokens = min(100, phases.shape[0])
            im = axes[0].imshow(phases[:n_tokens].T, aspect='auto', cmap='twilight')
            axes[0].set_xlabel('Token ID')
            axes[0].set_ylabel('Wave Component')
            axes[0].set_title(f'Token Phases (Step {step})')
            plt.colorbar(im, ax=axes[0], label='Phase (radians)')
            
            # Plot 2: Polar plot of phases (first wave component)
            ax_polar = plt.subplot(122, projection='polar')
            phase0 = phases[:n_tokens, 0]  # First wave component
            colors = plt.cm.viridis(np.linspace(0, 1, n_tokens))
            ax_polar.scatter(phase0, np.ones_like(phase0), c=colors, s=30, alpha=0.6)
            ax_polar.set_title(f'Phase Distribution on Unit Circle\n(First Wave, Step {step})')
            
            plt.tight_layout()
            
            filename = f'phases_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved phase plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot phases: {e}")
    
    def _plot_harmonics(self, step: int, embedding):
        """Plot harmonic amplitude distribution"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            harm_amps = embedding.harmonic_amps.detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Average harmonic profile across tokens
            mean_harmonics = harm_amps.mean(axis=0)  # (waves, harmonics)
            im = axes[0].imshow(mean_harmonics, aspect='auto', cmap='plasma')
            axes[0].set_xlabel('Harmonic (1f, 2f, 3f, ...)')
            axes[0].set_ylabel('Wave Component')
            axes[0].set_title(f'Mean Harmonic Amplitudes (Step {step})')
            plt.colorbar(im, ax=axes[0], label='Amplitude')
            
            # Plot 2: Harmonic distribution histogram
            for h in range(harm_amps.shape[2]):
                axes[1].hist(harm_amps[:, :, h].flatten(), bins=30, 
                           alpha=0.5, label=f'Harmonic {h+1}')
            axes[1].set_xlabel('Amplitude')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'Harmonic Amplitude Distribution (Step {step})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f'harmonics_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved harmonics plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot harmonics: {e}")
    
    def _plot_wave_packets(self, step: int, embedding):
        """Plot wave packets for sample tokens"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            freqs = embedding.base_freqs.detach().cpu().numpy()
            phases = embedding.phases.detach().cpu().numpy()
            harm_amps = embedding.harmonic_amps.detach().cpu().numpy()
            
            # Sample tokens to visualize
            n_tokens = freqs.shape[0]
            sample_tokens = [0, 1, min(10, n_tokens-1), min(50, n_tokens-1), 
                           min(100, n_tokens-1), min(500, n_tokens-1)]
            sample_tokens = [t for t in sample_tokens if t < n_tokens]
            
            fig, axes = plt.subplots(2, 3, figsize=(14, 7))
            axes = axes.flatten()
            
            # Time points for wave visualization
            t = np.linspace(0, 2*np.pi, 200)
            
            for idx, tok_id in enumerate(sample_tokens):
                if idx >= len(axes):
                    break
                
                ax = axes[idx]
                wave_sum = np.zeros_like(t)
                
                # Sum contributions from first 8 waves
                for w in range(min(8, freqs.shape[1])):
                    base_f = freqs[tok_id, w]
                    phase = phases[tok_id, w]
                    
                    # Add harmonics
                    for h in range(harm_amps.shape[2]):
                        amp = harm_amps[tok_id, w, h]
                        freq = base_f * (h + 1)
                        wave_sum += amp * np.cos(freq * t + phase)
                
                ax.plot(t, wave_sum, linewidth=2, color='cyan')
                ax.set_title(f'Token {tok_id}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='white', alpha=0.3, linewidth=0.5)
            
            # Hide unused subplots
            for idx in range(len(sample_tokens), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Wave Packets per Token (Step {step})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f'wave_packets_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved wave packets plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot wave packets: {e}")
    
    def generate_comparison_plots(self,
                                 experiments: List[Dict]):
        """
        Generate comparison plots across multiple experiments.
        
        Args:
            experiments: List of experiment dictionaries containing:
                - name: Experiment name
                - loss_history: List of loss values
                - metrics: Dictionary of metric histories
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not experiments:
                print("Warning: No experiments provided for comparison")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            # Plot 1: Loss comparison
            ax = axes[0]
            for exp in experiments:
                name = exp.get('name', 'Unknown')
                loss_history = exp.get('loss_history', [])
                steps = list(range(len(loss_history)))
                ax.plot(steps, loss_history, linewidth=2, label=name, alpha=0.8)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Learning rate comparison (if available)
            ax = axes[1]
            for exp in experiments:
                name = exp.get('name', 'Unknown')
                metrics = exp.get('metrics', {})
                if 'learning_rate' in metrics:
                    lr_history = metrics['learning_rate']
                    steps = list(range(len(lr_history)))
                    ax.plot(steps, lr_history, linewidth=2, label=name, alpha=0.8)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Plot 3: Perplexity comparison (if available)
            ax = axes[2]
            for exp in experiments:
                name = exp.get('name', 'Unknown')
                metrics = exp.get('metrics', {})
                if 'perplexity' in metrics:
                    ppl_history = metrics['perplexity']
                    steps = list(range(len(ppl_history)))
                    ax.plot(steps, ppl_history, linewidth=2, label=name, alpha=0.8)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Perplexity')
            ax.set_title('Perplexity Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Final metrics bar chart
            ax = axes[3]
            exp_names = [exp.get('name', 'Unknown') for exp in experiments]
            final_losses = [exp.get('loss_history', [0])[-1] if exp.get('loss_history') else 0 
                          for exp in experiments]
            
            x = np.arange(len(exp_names))
            ax.bar(x, final_losses, color='cyan', alpha=0.7, edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(exp_names, rotation=45, ha='right')
            ax.set_ylabel('Final Loss')
            ax.set_title('Final Loss Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Experiment Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = 'experiment_comparison.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved comparison plots: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to generate comparison plots: {e}")
            import traceback
            traceback.print_exc()


# ==========================================
# ConfigTracker
# ==========================================

class ConfigTracker:
    """
    Tracks and saves experiment configuration and metadata.
    
    Features:
    - JSON format for easy parsing and comparison
    - Automatic git hash capture for reproducibility
    - Hardware info includes GPU model, CUDA version, CPU count
    - Results include best validation loss, final perplexity, training time
    
    Example:
        >>> tracker = ConfigTracker(experiment_dir="experiments/exp_001")
        >>> 
        >>> # At experiment start
        >>> tracker.save_config(config, model, dataset_info)
        >>> 
        >>> # At experiment end
        >>> tracker.save_results(final_metrics, best_checkpoint, generation_samples)
        >>> 
        >>> # Load for analysis
        >>> config = tracker.load_config()
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize ConfigTracker.
        
        Args:
            experiment_dir: Root directory for experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.config_file = self.experiment_dir / 'config.json'
        self.results_file = self.experiment_dir / 'results.json'
        
        # Create experiment directory if it doesn't exist
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self,
                   config: Dict,
                   model: Optional[nn.Module] = None,
                   dataset_info: Optional[Dict] = None):
        """
        Save experiment configuration.
        
        Includes:
        - All hyperparameters
        - Model architecture details
        - Dataset information
        - Git commit hash
        - Timestamp
        - Hardware info (GPU, CPU, memory)
        
        Args:
            config: Configuration dictionary
            model: Model to extract architecture info from (optional)
            dataset_info: Dataset information dictionary (optional)
        """
        full_config = {
            'experiment_id': self.experiment_dir.name,
            'timestamp': datetime.now().isoformat(),
            'git_hash': get_git_hash(),
            'config': config,
            'hardware': self._get_hardware_info()
        }
        
        # Add model info if provided
        if model is not None:
            full_config['model'] = self._get_model_info(model)
        
        # Add dataset info if provided
        if dataset_info is not None:
            full_config['dataset'] = dataset_info
        
        # Save to file
        with open(self.config_file, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        print(f"✓ Saved config: {self.config_file}")
    
    def load_config(self) -> Dict:
        """
        Load experiment configuration.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_file.exists():
            return {}
        
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def save_results(self,
                    final_metrics: Dict,
                    best_checkpoint: Optional[str] = None,
                    generation_samples: Optional[List[str]] = None):
        """
        Save final experiment results.
        
        Args:
            final_metrics: Dictionary of final metrics
            best_checkpoint: Path to best checkpoint (optional)
            generation_samples: List of generated text samples (optional)
        """
        results = {
            'experiment_id': self.experiment_dir.name,
            'timestamp': datetime.now().isoformat(),
            'final_metrics': final_metrics
        }
        
        if best_checkpoint is not None:
            results['best_checkpoint'] = best_checkpoint
        
        if generation_samples is not None:
            results['generation_samples'] = generation_samples
        
        # Save to file
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Saved results: {self.results_file}")
    
    def _get_hardware_info(self) -> Dict:
        """Collect hardware information"""
        import platform
        
        hardware_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count()
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            hardware_info['gpu_available'] = True
            hardware_info['gpu_count'] = torch.cuda.device_count()
            hardware_info['gpu_model'] = torch.cuda.get_device_name(0)
            hardware_info['cuda_version'] = torch.version.cuda
        else:
            hardware_info['gpu_available'] = False
        
        return hardware_info
    
    def _get_model_info(self, model: nn.Module) -> Dict:
        """Extract model architecture information"""
        # Handle DataParallel
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        model_info = {
            'type': model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Try to extract common attributes
        for attr in ['d_model', 'num_layers', 'num_heads', 'num_waves', 'num_harmonics', 
                     'vocab_size', 'block_size', 'dropout']:
            if hasattr(model, attr):
                model_info[attr] = getattr(model, attr)
        
        return model_info


# ==========================================
# ResultsParser
# ==========================================

class ResultsParser:
    """
    Parses experiment results from text files into structured JSON format.
    
    Extracts:
    - Experiment configurations (model, optimizer, loss, hyperparameters)
    - Training metrics (step, loss, validation loss, wave ratio)
    - Final results (best validation loss, perplexity, speed)
    - Generation samples
    """
    
    def __init__(self, results_file: str):
        """
        Initialize ResultsParser.
        
        Args:
            results_file: Path to results.txt file
        """
        self.results_file = results_file
        self.experiments = []
        
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse the results file and extract all experiments.
        
        Returns:
            List of experiment dictionaries with structured data
        """
        with open(self.results_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by experiment sections
        # Look for experiment headers like "Standard Transformer (GPT-2)" or "Full Physics (RGD + QFE)"
        experiments = []
        current_experiment = None
        current_metrics = []
        
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Detect experiment header (boxed section)
            if line.startswith('╭─') and i + 1 < len(lines):
                # Next line contains experiment name
                exp_name_line = lines[i + 1]
                if '│' in exp_name_line:
                    exp_name = exp_name_line.strip('│ ').strip()
                    
                    # Skip if this is a generation output box (contains "Output")
                    if 'Output' in exp_name or 'The theory' in exp_name:
                        i += 1
                        continue
                    
                    # Skip if this is just a header box without actual experiment name
                    if '🔬' in exp_name and 'ABLATION' in exp_name:
                        i += 1
                        continue
                    
                    # Save previous experiment if exists
                    if current_experiment is not None:
                        current_experiment['training_metrics'] = current_metrics
                        experiments.append(current_experiment)
                    
                    # Start new experiment
                    current_experiment = {
                        'name': exp_name,
                        'config': {},
                        'training_metrics': [],
                        'final_results': {},
                        'generation_sample': ''
                    }
                    current_metrics = []
            
            # Extract configuration parameters
            elif current_experiment is not None:
                # Parameters line (only capture if we don't already have it)
                if line.startswith('📊 Parameters:'):
                    if 'parameters' not in current_experiment['config']:
                        params_str = line.split(':')[1].strip()
                        # Extract number like "52,892,160 (52.89M)"
                        params_num = params_str.split('(')[0].strip().replace(',', '')
                        current_experiment['config']['parameters'] = int(params_num)
                
                # Steps line
                elif line.startswith('🔄 Steps:'):
                    steps = int(line.split(':')[1].strip())
                    current_experiment['config']['steps'] = steps
                
                # Optimizer line
                elif line.startswith('⚙️  Optimizer:') or line.startswith('⚡ Optimizer:'):
                    optimizer = line.split(':')[1].strip()
                    current_experiment['config']['optimizer'] = optimizer
                
                # Loss line
                elif line.startswith('📉 Loss:') or line.startswith('🌌 Loss:'):
                    loss = line.split(':')[1].strip()
                    current_experiment['config']['loss'] = loss
                
                # Learning rate line
                elif line.startswith('📈 LR:'):
                    lr = line.split(':')[1].strip()
                    current_experiment['config']['learning_rate'] = lr
                
                # Training step metrics
                elif line.startswith('Step ') and '|' in line:
                    # Parse line like: "Step   250 | Train(CE): 7.0259 | Val: 7.0833 | AvgTrain: 8.3495 | R: 0.500"
                    parts = line.split('|')
                    if len(parts) >= 4:
                        try:
                            step_str = parts[0].strip().split()[1]
                            step = int(step_str)
                            
                            train_loss_str = parts[1].strip().split(':')[1].strip()
                            train_loss = float(train_loss_str)
                            
                            val_loss_str = parts[2].strip().split(':')[1].strip()
                            val_loss = float(val_loss_str)
                            
                            avg_train_str = parts[3].strip().split(':')[1].strip()
                            avg_train = float(avg_train_str)
                            
                            wave_ratio = None
                            if len(parts) >= 5 and 'R:' in parts[4]:
                                wave_ratio_str = parts[4].strip().split(':')[1].strip()
                                wave_ratio = float(wave_ratio_str)
                            
                            metric = {
                                'step': step,
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                                'avg_train_loss': avg_train,
                            }
                            if wave_ratio is not None:
                                metric['wave_ratio'] = wave_ratio
                            
                            current_metrics.append(metric)
                        except (ValueError, IndexError):
                            pass
                
                # Best model restoration line
                elif line.startswith('♻️  Restoring best model'):
                    # Extract best validation loss like "(Val: 4.4382)"
                    if '(' in line and 'Val:' in line:
                        val_str = line.split('Val:')[1].split(')')[0].strip()
                        try:
                            current_experiment['final_results']['best_val_loss'] = float(val_str)
                        except ValueError:
                            pass
                
                # Generation sample section
                elif 'Output' in line and '─' in line:
                    # Start of generation sample box
                    sample_lines = []
                    i += 1
                    while i < len(lines) and not lines[i].startswith('╰'):
                        if lines[i].startswith('│'):
                            sample_lines.append(lines[i].strip('│ ').strip())
                        i += 1
                    current_experiment['generation_sample'] = ' '.join(sample_lines)
            
            i += 1
        
        # Save last experiment
        if current_experiment is not None:
            current_experiment['training_metrics'] = current_metrics
            experiments.append(current_experiment)
        
        # Parse ablation study table if present
        ablation_results = self._parse_ablation_table(content)
        if ablation_results:
            # Match ablation results to experiments by name
            for exp in experiments:
                for ablation in ablation_results:
                    if ablation['name'] in exp['name']:
                        exp['final_results'].update({
                            'val_loss': ablation['val_loss'],
                            'perplexity': ablation['perplexity'],
                            'speed_tokens_per_sec': ablation['speed']
                        })
        
        self.experiments = experiments
        return experiments
    
    def _parse_ablation_table(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse the ablation study results table.
        
        Returns:
            List of ablation results
        """
        results = []
        lines = content.split('\n')
        
        # Find the ablation table
        in_table = False
        for line in lines:
            if '🔬 Ablation Study Results' in line:
                in_table = True
                continue
            
            if in_table:
                # Skip table borders and headers
                if line.startswith('┏') or line.startswith('┃') or line.startswith('┡') or line.startswith('└'):
                    continue
                
                # Parse data rows starting with │
                if line.startswith('│'):
                    parts = [p.strip() for p in line.split('│') if p.strip()]
                    if len(parts) >= 5:
                        # Skip header row
                        if 'Experiment' in parts[0] or 'RGD' in parts[1]:
                            continue
                        
                        try:
                            name = parts[0]
                            val_loss = float(parts[3])
                            perplexity = float(parts[4])
                            speed = float(parts[5].replace(',', ''))
                            
                            results.append({
                                'name': name,
                                'val_loss': val_loss,
                                'perplexity': perplexity,
                                'speed': speed
                            })
                        except (ValueError, IndexError):
                            pass
                
                # End of table
                if line.startswith('└'):
                    break
        
        return results
    
    def save_to_json(self, output_file: str):
        """
        Save parsed results to JSON file.
        
        Args:
            output_file: Path to output JSON file
        """
        if not self.experiments:
            self.parse()
        
        output_data = {
            'experiments': self.experiments,
            'parsed_at': datetime.now().isoformat(),
            'source_file': self.results_file
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def get_experiment_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment data by name.
        
        Args:
            name: Experiment name (can be partial match)
            
        Returns:
            Experiment dictionary or None if not found
        """
        if not self.experiments:
            self.parse()
        
        for exp in self.experiments:
            if name.lower() in exp['name'].lower():
                return exp
        return None
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Get all parsed experiments.
        
        Returns:
            List of experiment dictionaries
        """
        if not self.experiments:
            self.parse()
        return self.experiments


# ==========================================
# ResultsAggregator
# ==========================================

class ResultsAggregator:
    """
    Aggregates and compares results from multiple experiments.
    
    Features:
    - Load multiple experiment results
    - Generate comparison tables
    - Statistical significance testing
    - Ablation study analysis
    - Summary visualizations
    """
    
    def __init__(self):
        """Initialize ResultsAggregator."""
        self.experiments = []
        
    def load_from_json(self, json_file: str):
        """
        Load experiments from JSON file.
        
        Args:
            json_file: Path to JSON file with parsed experiments
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
            self.experiments.extend(data.get('experiments', []))
    
    def load_from_parser(self, parser: ResultsParser):
        """
        Load experiments from ResultsParser.
        
        Args:
            parser: ResultsParser instance with parsed experiments
        """
        self.experiments.extend(parser.get_all_experiments())
    
    def add_experiment(self, experiment: Dict[str, Any]):
        """
        Add a single experiment.
        
        Args:
            experiment: Experiment dictionary
        """
        self.experiments.append(experiment)
    
    def generate_comparison_table(self) -> str:
        """
        Generate a comparison table of all experiments.
        
        Returns:
            Formatted table string
        """
        if not self.experiments:
            return "No experiments loaded."
        
        # Build table
        rows = []
        headers = ['Experiment', 'Parameters', 'Val Loss', 'Perplexity', 'Speed (tok/s)']
        
        for exp in self.experiments:
            name = exp['name']
            params = exp['config'].get('parameters', 'N/A')
            if isinstance(params, int):
                params = f"{params:,}"
            
            val_loss = exp['final_results'].get('val_loss', 
                       exp['final_results'].get('best_val_loss', 'N/A'))
            perplexity = exp['final_results'].get('perplexity', 'N/A')
            speed = exp['final_results'].get('speed_tokens_per_sec', 'N/A')
            
            if isinstance(speed, (int, float)):
                speed = f"{speed:,.0f}"
            
            rows.append([name, params, val_loss, perplexity, speed])
        
        # Format table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        def format_row(row, widths):
            return '| ' + ' | '.join(str(item).ljust(width) for item, width in zip(row, widths)) + ' |'
        
        separator = '|-' + '-|-'.join('-' * width for width in col_widths) + '-|'
        
        table = []
        table.append(format_row(headers, col_widths))
        table.append(separator)
        for row in rows:
            table.append(format_row(row, col_widths))
        
        return '\n'.join(table)
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics across all experiments.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.experiments:
            return {}
        
        val_losses = []
        perplexities = []
        speeds = []
        
        for exp in self.experiments:
            val_loss = exp['final_results'].get('val_loss',
                       exp['final_results'].get('best_val_loss'))
            if val_loss is not None and isinstance(val_loss, (int, float)):
                val_losses.append(val_loss)
            
            perplexity = exp['final_results'].get('perplexity')
            if perplexity is not None and isinstance(perplexity, (int, float)):
                perplexities.append(perplexity)
            
            speed = exp['final_results'].get('speed_tokens_per_sec')
            if speed is not None and isinstance(speed, (int, float)):
                speeds.append(speed)
        
        import statistics
        
        summary = {
            'num_experiments': len(self.experiments),
            'val_loss': {
                'mean': statistics.mean(val_losses) if val_losses else None,
                'median': statistics.median(val_losses) if val_losses else None,
                'min': min(val_losses) if val_losses else None,
                'max': max(val_losses) if val_losses else None,
                'stdev': statistics.stdev(val_losses) if len(val_losses) > 1 else None
            },
            'perplexity': {
                'mean': statistics.mean(perplexities) if perplexities else None,
                'median': statistics.median(perplexities) if perplexities else None,
                'min': min(perplexities) if perplexities else None,
                'max': max(perplexities) if perplexities else None,
                'stdev': statistics.stdev(perplexities) if len(perplexities) > 1 else None
            },
            'speed': {
                'mean': statistics.mean(speeds) if speeds else None,
                'median': statistics.median(speeds) if speeds else None,
                'min': min(speeds) if speeds else None,
                'max': max(speeds) if speeds else None,
                'stdev': statistics.stdev(speeds) if len(speeds) > 1 else None
            }
        }
        
        return summary
    
    def compare_experiments(self, exp1_name: str, exp2_name: str) -> Dict[str, Any]:
        """
        Compare two experiments.
        
        Args:
            exp1_name: Name of first experiment (partial match)
            exp2_name: Name of second experiment (partial match)
            
        Returns:
            Comparison dictionary
        """
        exp1 = None
        exp2 = None
        
        for exp in self.experiments:
            if exp1_name.lower() in exp['name'].lower():
                exp1 = exp
            if exp2_name.lower() in exp['name'].lower():
                exp2 = exp
        
        if exp1 is None or exp2 is None:
            return {'error': 'One or both experiments not found'}
        
        comparison = {
            'experiment_1': exp1['name'],
            'experiment_2': exp2['name'],
            'val_loss_diff': None,
            'val_loss_improvement_pct': None,
            'perplexity_diff': None,
            'speed_diff': None
        }
        
        val1 = exp1['final_results'].get('val_loss', exp1['final_results'].get('best_val_loss'))
        val2 = exp2['final_results'].get('val_loss', exp2['final_results'].get('best_val_loss'))
        
        if val1 is not None and val2 is not None:
            comparison['val_loss_diff'] = val2 - val1
            comparison['val_loss_improvement_pct'] = ((val1 - val2) / val1) * 100
        
        perp1 = exp1['final_results'].get('perplexity')
        perp2 = exp2['final_results'].get('perplexity')
        
        if perp1 is not None and perp2 is not None:
            comparison['perplexity_diff'] = perp2 - perp1
        
        speed1 = exp1['final_results'].get('speed_tokens_per_sec')
        speed2 = exp2['final_results'].get('speed_tokens_per_sec')
        
        if speed1 is not None and speed2 is not None:
            comparison['speed_diff'] = speed2 - speed1
        
        return comparison
    
    def generate_ablation_analysis(self) -> Dict[str, Any]:
        """
        Generate ablation study analysis.
        
        Identifies baseline and component contributions.
        
        Returns:
            Ablation analysis dictionary
        """
        if not self.experiments:
            return {}
        
        # Try to identify baseline (standard transformer)
        baseline = None
        for exp in self.experiments:
            if 'standard' in exp['name'].lower() or 'transformer' in exp['name'].lower():
                if 'wave' not in exp['name'].lower() and 'physics' not in exp['name'].lower():
                    baseline = exp
                    break
        
        # Identify component experiments
        rgd_only = None
        qfe_only = None
        full_physics = None
        
        for exp in self.experiments:
            name_lower = exp['name'].lower()
            if 'rgd only' in name_lower:
                rgd_only = exp
            elif 'qfe only' in name_lower:
                qfe_only = exp
            elif 'full physics' in name_lower or ('rgd' in name_lower and 'qfe' in name_lower):
                full_physics = exp
        
        analysis = {
            'baseline': baseline['name'] if baseline else None,
            'components': {}
        }
        
        if baseline:
            baseline_val = baseline['final_results'].get('val_loss', 
                          baseline['final_results'].get('best_val_loss'))
            
            if rgd_only:
                rgd_val = rgd_only['final_results'].get('val_loss',
                         rgd_only['final_results'].get('best_val_loss'))
                if baseline_val and rgd_val:
                    analysis['components']['RGD'] = {
                        'val_loss': rgd_val,
                        'improvement_over_baseline': baseline_val - rgd_val,
                        'improvement_pct': ((baseline_val - rgd_val) / baseline_val) * 100
                    }
            
            if qfe_only:
                qfe_val = qfe_only['final_results'].get('val_loss',
                         qfe_only['final_results'].get('best_val_loss'))
                if baseline_val and qfe_val:
                    analysis['components']['QFE'] = {
                        'val_loss': qfe_val,
                        'improvement_over_baseline': baseline_val - qfe_val,
                        'improvement_pct': ((baseline_val - qfe_val) / baseline_val) * 100
                    }
            
            if full_physics:
                full_val = full_physics['final_results'].get('val_loss',
                          full_physics['final_results'].get('best_val_loss'))
                if baseline_val and full_val:
                    analysis['components']['Full (RGD + QFE)'] = {
                        'val_loss': full_val,
                        'improvement_over_baseline': baseline_val - full_val,
                        'improvement_pct': ((baseline_val - full_val) / baseline_val) * 100
                    }
        
        # Identify top-performing component
        if analysis['components']:
            best_component = max(analysis['components'].items(),
                               key=lambda x: x[1].get('improvement_over_baseline', 0))
            analysis['best_component'] = best_component[0]
            analysis['best_improvement'] = best_component[1].get('improvement_over_baseline')
        
        return analysis
    
    def save_summary(self, output_file: str):
        """
        Save comprehensive summary to JSON file.
        
        Args:
            output_file: Path to output JSON file
        """
        summary = {
            'comparison_table': self.generate_comparison_table(),
            'statistics': self.generate_summary_statistics(),
            'ablation_analysis': self.generate_ablation_analysis(),
            'experiments': self.experiments,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
