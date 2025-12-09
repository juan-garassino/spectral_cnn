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

# Physics-first imports with backward compatibility
try:
    from wave_physics_core import (
        WaveNativeOptimizer,
        WaveCoherenceLoss,
        WaveDiagnostics,
        create_physics_optimizer,
        create_physics_loss
    )
    PHYSICS_CORE_AVAILABLE = True
except ImportError:
    PHYSICS_CORE_AVAILABLE = False
    WaveNativeOptimizer = None
    WaveCoherenceLoss = None
    WaveDiagnostics = None

# Legacy imports for backward compatibility
try:
    from physics_optim import ResonantGradientDescent, QuantumFieldEntanglementLoss
    LEGACY_PHYSICS_AVAILABLE = True
except ImportError:
    LEGACY_PHYSICS_AVAILABLE = False
    ResonantGradientDescent = None
    QuantumFieldEntanglementLoss = None

# Config - SCALED UP for 16GB GPU (using ~12GB now)
CLASSIC_STEPS = 5000  # Steps for Classic Transformer
WAVE_STEPS = 15000    # 3x steps for Wave (to match Classic performance!)
BATCH_SIZE = 32       # Keep same (larger batch = OOM)
BLOCK_SIZE = 256      # Keep same
D_MODEL = 384         # Scaled up from 256 (1.5x wider)
NUM_LAYERS = 8        # Scaled up from 6 (more depth)
NUM_HEADS = 8         # Keep same
NUM_WAVES = 48        # Scaled up from 32 (more wave components)
NUM_HARMONICS = 4     # Harmonics per wave

# Learning rates
CLASSIC_LR = 3e-4     # Standard LR for Classic
WAVE_LR = 6e-4        # 2x higher LR for Wave (faster convergence)
WARMUP_STEPS = 300    # Warmup for stability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ⚡ Physics-Informed Optimization Flags
USE_RGD = True        # WaveNativeOptimizer (physics-first) for Wave model
RGD_STRENGTH = 0.3    # Coherence weight for SVD gradient projection (0=off, 1=full)
RGD_WARMUP = 500      # Steps to gradually enable resonance (legacy compatibility)

USE_QFE = True        # WaveCoherenceLoss (physics-first) for Wave model
QFE_LAMBDA = 0.05     # Weight for phase coherence term (start small!)
QFE_THRESHOLD = 0.01  # Amplitude threshold for phase computation (legacy compatibility)

# Annealing schedule for wave embeddings
ANNEALING_STEPS = 3000  # Steps to decay standard_embed_ratio from 1.0 to 0.0


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
    
    # Check if this is a WaveGPT model (new harmonic structure)
    if not hasattr(model, 'embedding') or not hasattr(model.embedding, 'base_freqs'):
        return  # Not a wave model, skip wave-specific plots
    
    with torch.no_grad():
        # 2. Base Frequency Distribution
        freqs = model.embedding.base_freqs.cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Frequency heatmap
        im = axes[0].imshow(freqs[:100].T, aspect='auto', cmap='viridis')
        axes[0].set_xlabel('Token ID (first 100)')
        axes[0].set_ylabel('Wave Component')
        axes[0].set_title('Base Frequencies per Token')
        plt.colorbar(im, ax=axes[0])
        
        # Frequency histogram
        axes[1].hist(freqs.flatten(), bins=50, color='cyan', alpha=0.7, edgecolor='white')
        axes[1].set_xlabel('Frequency Value')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Frequency Distribution')
        axes[1].axvline(freqs.mean(), color='magenta', linestyle='--', linewidth=2, label=f'Mean: {freqs.mean():.2f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_frequencies.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Token Phase Distribution  
        phases = model.embedding.phases.cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(phases[:100].T, aspect='auto', cmap='twilight')
        ax.set_xlabel('Token ID (first 100)')
        ax.set_ylabel('Wave Component')
        ax.set_title('Token Phases (0 to 2π)')
        plt.colorbar(im, ax=ax, label='Phase (radians)')
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_phases.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Harmonic Amplitude Distribution (NEW!)
        harm_amps = model.embedding.harmonic_amps.cpu().numpy()  # (vocab, waves, harmonics)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Average harmonic profile across tokens
        mean_harmonics = harm_amps.mean(axis=0)  # (waves, harmonics)
        im = axes[0].imshow(mean_harmonics, aspect='auto', cmap='plasma')
        axes[0].set_xlabel('Harmonic (1f, 2f, 3f, 4f)')
        axes[0].set_ylabel('Wave Component')
        axes[0].set_title('Mean Harmonic Amplitude per Wave')
        plt.colorbar(im, ax=axes[0])
        
        # Harmonic decay pattern (should show 1/n falloff for natural signals)
        harm_means = np.abs(harm_amps).mean(axis=(0, 1))  # Mean across vocab and waves
        axes[1].bar(range(1, len(harm_means)+1), harm_means, color='cyan', alpha=0.8, edgecolor='white')
        axes[1].set_xlabel('Harmonic Number')
        axes[1].set_ylabel('Mean |Amplitude|')
        axes[1].set_title('Harmonic Decay Pattern')
        # Add 1/n reference line
        x = np.arange(1, len(harm_means)+1)
        ref = harm_means[0] / x
        axes[1].plot(x, ref, 'r--', label='1/n decay (natural sound)', linewidth=2)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_harmonics.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Sample Wave Packets for specific tokens (with harmonics!)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        sample_tokens = [0, 1, 10, 50, 100, 500]
        t = np.linspace(0, 2*np.pi, 300)
        
        for idx, (ax, tok_id) in enumerate(zip(axes.flat, sample_tokens)):
            if tok_id < len(freqs):
                wave = np.zeros_like(t)
                for w in range(min(8, freqs.shape[1])):  # First 8 waves
                    base_f = freqs[tok_id, w]
                    phase = phases[tok_id, w]
                    for h in range(harm_amps.shape[2]):  # All harmonics
                        amp = harm_amps[tok_id, w, h]
                        freq = base_f * (h + 1)  # Harmonic: 1f, 2f, 3f...
                        wave += amp * np.sin(freq * t + phase)
                ax.plot(t, wave, color='cyan', linewidth=1.5)
                ax.fill_between(t, wave, alpha=0.3, color='cyan')
                ax.set_title(f'Token {tok_id}', fontsize=10)
                ax.set_xlabel('t')
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='white', alpha=0.3)
        
        plt.suptitle('Wave Packets per Token (with Harmonics)', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_wave_packets.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 6. Attention Phase Shifts (if available)
        if hasattr(model.blocks[0].attn, 'temperature'):
            temps = []
            for block in model.blocks:
                temps.append(block.attn.temperature.item())
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(temps)), temps, color='magenta', alpha=0.8)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Learned Temperature')
            ax.set_title('Attention Temperature per Layer')
            ax.axhline(np.sqrt(NUM_WAVES), color='cyan', linestyle='--', label=f'Initial: √{NUM_WAVES}')
            ax.legend()
            plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_attention_temps.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 7. POLAR PHASE PLOT - tokens as points on unit circle
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})
        
        # First wave component phases for first 100 tokens
        phase0 = phases[:100, 0]  # First wave component
        amp0 = np.abs(harm_amps[:100, 0, 0])  # Amplitude of first harmonic
        amp0_norm = amp0 / amp0.max() if amp0.max() > 0 else amp0
        
        colors = plt.cm.viridis(np.linspace(0, 1, 100))
        axes[0].scatter(phase0, np.ones_like(phase0), c=colors, s=50, alpha=0.8)
        axes[0].set_title('Token Phases on Unit Circle\n(First Wave Component)')
        axes[0].set_rticks([])
        
        # Phase vs Token ID spiral
        all_phases = phases[:100, :8].mean(axis=1)  # Average phase across first 8 waves
        radii = np.linspace(0.3, 1, 100)
        axes[1].scatter(all_phases, radii, c=colors, s=30, alpha=0.8)
        axes[1].set_title('Phase Spiral by Token ID\n(Avg across waves)')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_polar_phases.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 8. COMPLEX PLANE - sin/cos as real/imaginary
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Sample tokens in complex plane
        sample_ids = [0, 1, 10, 50, 100, 500]
        t_sample = 0  # Fixed time point
        
        for tok_id in sample_ids:
            if tok_id < len(freqs):
                # Compute complex representation: e^(i*phase) * amplitude
                real_parts = []
                imag_parts = []
                for w in range(min(8, freqs.shape[1])):
                    for h in range(harm_amps.shape[2]):
                        phase = phases[tok_id, w]
                        amp = harm_amps[tok_id, w, h]
                        real_parts.append(amp * np.cos(phase))
                        imag_parts.append(amp * np.sin(phase))
                
                # Plot as scatter
                axes[0].scatter(real_parts, imag_parts, alpha=0.5, label=f'Token {tok_id}', s=20)
        
        axes[0].axhline(0, color='white', alpha=0.3)
        axes[0].axvline(0, color='white', alpha=0.3)
        axes[0].set_xlabel('Real (cos)')
        axes[0].set_ylabel('Imaginary (sin)')
        axes[0].set_title('Token Waves in Complex Plane')
        axes[0].legend(fontsize=8)
        axes[0].set_aspect('equal')
        
        # Phase-Frequency scatter for all tokens
        freq_flat = freqs[:500].flatten()
        phase_flat = phases[:500].flatten()
        axes[1].scatter(freq_flat, phase_flat, alpha=0.1, s=5, c='cyan')
        axes[1].set_xlabel('Base Frequency')
        axes[1].set_ylabel('Phase (radians)')
        axes[1].set_title('Frequency vs Phase Distribution')
        axes[1].axhline(np.pi, color='magenta', linestyle='--', alpha=0.5, label='π')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_complex_plane.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 9. TOKEN SPECTROGRAM - frequency spectrum per token
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Build spectrogram: each row is a token, each column is a frequency bin
        num_tokens = min(200, len(freqs))
        freq_bins = np.linspace(0, 10, 100)  # Frequency range
        spectrogram = np.zeros((num_tokens, len(freq_bins)))
        
        for tok_id in range(num_tokens):
            for w in range(freqs.shape[1]):
                base_f = freqs[tok_id, w]
                for h in range(harm_amps.shape[2]):
                    f_actual = base_f * (h + 1)
                    amp = np.abs(harm_amps[tok_id, w, h])
                    # Find nearest bin
                    if 0 <= f_actual < 10:
                        bin_idx = int(f_actual / 10 * len(freq_bins))
                        bin_idx = min(bin_idx, len(freq_bins) - 1)
                        spectrogram[tok_id, bin_idx] += amp
        
        im = ax.imshow(spectrogram.T, aspect='auto', cmap='magma', origin='lower',
                       extent=[0, num_tokens, 0, 10])
        ax.set_xlabel('Token ID')
        ax.set_ylabel('Frequency')
        ax.set_title('Token Spectrogram (Frequency Content per Token)')
        plt.colorbar(im, ax=ax, label='Amplitude')
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_spectrogram.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 10. INTERFERENCE PATTERN - how two tokens interact
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        t = np.linspace(0, 4*np.pi, 500)
        
        token_pairs = [(0, 1), (10, 11), (50, 100)]
        
        for idx, (tok_a, tok_b) in enumerate(token_pairs):
            if tok_a < len(freqs) and tok_b < len(freqs):
                # Generate waves for each token
                wave_a = np.zeros_like(t)
                wave_b = np.zeros_like(t)
                
                for w in range(min(4, freqs.shape[1])):
                    for h in range(harm_amps.shape[2]):
                        f_a = freqs[tok_a, w] * (h + 1)
                        f_b = freqs[tok_b, w] * (h + 1)
                        p_a = phases[tok_a, w]
                        p_b = phases[tok_b, w]
                        a_a = harm_amps[tok_a, w, h]
                        a_b = harm_amps[tok_b, w, h]
                        wave_a += a_a * np.sin(f_a * t + p_a)
                        wave_b += a_b * np.sin(f_b * t + p_b)
                
                # Individual waves
                axes[0, idx].plot(t, wave_a, 'c-', alpha=0.7, label=f'Token {tok_a}')
                axes[0, idx].plot(t, wave_b, 'm-', alpha=0.7, label=f'Token {tok_b}')
                axes[0, idx].set_title(f'Tokens {tok_a} & {tok_b}')
                axes[0, idx].legend(fontsize=8)
                axes[0, idx].set_xlabel('t')
                axes[0, idx].grid(alpha=0.3)
                
                # Interference (sum)
                interference = wave_a + wave_b
                axes[1, idx].fill_between(t, interference, alpha=0.5, color='yellow')
                axes[1, idx].plot(t, interference, 'y-', linewidth=1)
                axes[1, idx].set_title(f'Interference (sum)')
                axes[1, idx].set_xlabel('t')
                axes[1, idx].grid(alpha=0.3)
                axes[1, idx].axhline(0, color='white', alpha=0.3)
        
        plt.suptitle('Wave Interference Patterns Between Token Pairs', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_interference.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 11. 3D WAVE SURFACE - tokens × time × amplitude
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        t_3d = np.linspace(0, 2*np.pi, 50)
        token_ids_3d = np.arange(0, 50)
        T, TOK = np.meshgrid(t_3d, token_ids_3d)
        Z = np.zeros_like(T)
        
        for i, tok_id in enumerate(token_ids_3d):
            for w in range(min(4, freqs.shape[1])):
                f = freqs[tok_id, w]
                p = phases[tok_id, w]
                a = harm_amps[tok_id, w, 0]
                Z[i, :] += a * np.sin(f * t_3d + p)
        
        surf = ax.plot_surface(TOK, T, Z, cmap='viridis', alpha=0.8, edgecolor='none')
        ax.set_xlabel('Token ID')
        ax.set_ylabel('Time')
        ax.set_zlabel('Wave Amplitude')
        ax.set_title('Token Wave Landscape')
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_wave_surface.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 12. FFT SPECTRUM ANALYSIS (Diagnostic Plot)
        # Shows harmonic peaks in the frequency domain
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Compute FFT of embeddings
        embeddings_flat = model.embedding.simple_embed.weight.data.cpu().numpy() if hasattr(model.embedding, 'simple_embed') else None
        if embeddings_flat is not None:
            fft_result = np.fft.rfft(embeddings_flat, axis=-1)
            magnitudes = np.abs(fft_result)
            avg_spectrum = magnitudes.mean(axis=0)
            
            # Plot average spectrum
            axes[0].plot(avg_spectrum, color='cyan', linewidth=1.5)
            axes[0].fill_between(range(len(avg_spectrum)), avg_spectrum, alpha=0.3, color='cyan')
            axes[0].set_xlabel('Frequency Index')
            axes[0].set_ylabel('Magnitude')
            axes[0].set_title('FFT Spectrum (Harmonic Peak Detection)')
            axes[0].grid(True, alpha=0.3)
            
            # Find and mark peaks
            threshold = avg_spectrum.mean() + 0.5 * avg_spectrum.std()
            peaks = []
            for i in range(1, len(avg_spectrum) - 1):
                if avg_spectrum[i] > avg_spectrum[i-1] and avg_spectrum[i] > avg_spectrum[i+1]:
                    if avg_spectrum[i] > threshold:
                        peaks.append(i)
                        axes[0].axvline(x=i, color='magenta', linestyle='--', alpha=0.5)
            
            # Spectrum heatmap per token
            im = axes[1].imshow(magnitudes[:100].T, aspect='auto', cmap='magma', origin='lower')
            axes[1].set_xlabel('Token ID (first 100)')
            axes[1].set_ylabel('Frequency Index')
            axes[1].set_title('Per-Token Frequency Spectrum')
            plt.colorbar(im, ax=axes[1], label='Magnitude')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_fft_spectrum.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 13. AUTOCORRELATION PLOT (Interference Fringes Detection)
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Compute autocorrelation of wave patterns
        sample_waves = np.zeros((100, 300))
        t = np.linspace(0, 4*np.pi, 300)
        for tok_id in range(100):
            for w in range(min(4, freqs.shape[1])):
                for h in range(harm_amps.shape[2]):
                    f = freqs[tok_id, w] * (h + 1)
                    p = phases[tok_id, w]
                    a = harm_amps[tok_id, w, h]
                    sample_waves[tok_id] += a * np.sin(f * t + p)
        
        # Compute average autocorrelation
        autocorr_sum = np.zeros(300)
        for wave in sample_waves:
            wave_centered = wave - wave.mean()
            autocorr = np.correlate(wave_centered, wave_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-8)
            autocorr_sum += autocorr
        autocorr_avg = autocorr_sum / len(sample_waves)
        
        ax.plot(autocorr_avg, color='yellow', linewidth=1.5)
        ax.fill_between(range(len(autocorr_avg)), autocorr_avg, alpha=0.3, color='yellow')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Autocorrelation (Interference Fringe Detection)')
        ax.axhline(0, color='white', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_autocorrelation.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 14. TRAJECTORY STABILITY PLOT
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot norm evolution across tokens
        norms = np.linalg.norm(sample_waves, axis=1)
        axes[0].plot(norms, color='green', linewidth=1.5)
        axes[0].fill_between(range(len(norms)), norms, alpha=0.3, color='green')
        axes[0].set_xlabel('Token ID')
        axes[0].set_ylabel('Wave Norm')
        axes[0].set_title('Wave Norm per Token (Trajectory Stability)')
        axes[0].axhline(norms.mean(), color='magenta', linestyle='--', label=f'Mean: {norms.mean():.2f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot norm distribution
        axes[1].hist(norms, bins=30, color='green', alpha=0.7, edgecolor='white')
        axes[1].set_xlabel('Wave Norm')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Norm Distribution (Bounded = Stable)')
        axes[1].axvline(norms.mean(), color='magenta', linestyle='--', linewidth=2, label=f'Mean: {norms.mean():.2f}')
        axes[1].axvline(norms.mean() + 2*norms.std(), color='red', linestyle=':', label=f'+2σ: {norms.mean() + 2*norms.std():.2f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace(' ', '_')}_trajectory_stability.png", dpi=150, bbox_inches='tight')
        plt.close()

def run_wave_benchmark(config_name, model, train_data, console, tokenizer, num_steps=5000, lr=3e-4, 
                        use_radam=False, use_rgd=False, use_qfe=False, experiment_dir=None, enable_monitoring=True):
    """Benchmark a single model with configurable steps, LR, optimizer, and loss"""
    console.print(Panel(f"[bold cyan]{config_name}[/bold cyan]", title="🧪 Model Config", border_style="cyan"))
    
    params = sum(p.numel() for p in model.parameters())
    console.print(f"📊 Parameters: [bold]{params:,}[/bold] ({params/1e6:.2f}M)")
    console.print(f"🔄 Training Steps: [bold]{num_steps}[/bold]")
    
    # Initialize monitoring components if enabled
    checkpoint_manager = None
    metrics_logger = None
    viz_manager = None
    config_tracker = None
    
    if enable_monitoring and experiment_dir:
        import sys
        import os
        # Add current directory to path for monitoring import
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from monitoring import (
            CheckpointManager, MetricsLogger, 
            VisualizationManager, ConfigTracker,
            create_experiment_directory
        )
        
        # Create experiment directory structure
        dirs = create_experiment_directory("experiments", experiment_dir)
        console.print(f"📁 Experiment directory: {dirs['root']}")
        
        # Initialize monitoring components
        checkpoint_manager = CheckpointManager(
            experiment_dir=dirs['root'],
            save_interval=500,  # More frequent for benchmark
            keep_last_n=3
        )
        
        metrics_logger = MetricsLogger(
            log_dir=dirs['logs'],
            log_interval=10
        )
        
        viz_manager = VisualizationManager(
            viz_dir=dirs['visualizations'],
            viz_interval=500  # More frequent for benchmark
        )
        
        config_tracker = ConfigTracker(
            experiment_dir=dirs['root']
        )
        
        # Save initial configuration
        config_dict = {
            'model_name': config_name,
            'num_steps': num_steps,
            'learning_rate': lr,
            'optimizer': 'RGD' if use_rgd else ('RAdam' if use_radam else 'AdamW'),
            'loss_function': 'QFE' if use_qfe else 'CrossEntropy',
            'batch_size': BATCH_SIZE,
            'block_size': BLOCK_SIZE,
            'd_model': D_MODEL,
            'num_layers': NUM_LAYERS,
            'num_heads': NUM_HEADS,
            'num_waves': NUM_WAVES,
            'num_harmonics': NUM_HARMONICS
        }
        
        dataset_info = {
            'dataset': 'tiny_shakespeare',
            'train_tokens': len(train_data),
            'split': 0.9
        }
        
        try:
            config_tracker.save_config(config_dict, model, dataset_info)
            console.print("✓ Configuration saved")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save config: {e}[/yellow]")
    
    # Select optimizer
    if use_rgd:
        # Use physics-first WaveNativeOptimizer if available
        if PHYSICS_CORE_AVAILABLE and WaveNativeOptimizer is not None:
            optimizer_name = "⚡WaveNative"
            optimizer = WaveNativeOptimizer(
                model.parameters(), 
                lr=lr, 
                coherence_weight=RGD_STRENGTH,
                damping=0.1,
                weight_decay=0.01
            )
            console.print(f"⚙️  Optimizer: [bold magenta]{optimizer_name}[/bold magenta] (coherence={RGD_STRENGTH}, damping=0.1)")
        elif LEGACY_PHYSICS_AVAILABLE and ResonantGradientDescent is not None:
            # Fall back to legacy ResonantGradientDescent
            optimizer_name = "⚡RGD (legacy)"
            optimizer = ResonantGradientDescent(
                model.parameters(), 
                lr=lr, 
                resonance_strength=RGD_STRENGTH,
                warmup_steps=RGD_WARMUP,
                weight_decay=0.01
            )
            console.print(f"⚙️  Optimizer: [bold magenta]{optimizer_name}[/bold magenta] (strength={RGD_STRENGTH}, warmup={RGD_WARMUP})")
        else:
            # No physics optimizer available, fall back to AdamW
            optimizer_name = "AdamW (fallback)"
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            console.print(f"⚙️  Optimizer: [bold yellow]{optimizer_name}[/bold yellow] (physics optimizers unavailable)")
    elif use_radam:
        optimizer_name = "RAdam"
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=0.01)
        console.print(f"⚙️  Optimizer: [bold]{optimizer_name}[/bold]")
    else:
        optimizer_name = "AdamW"
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        console.print(f"⚙️  Optimizer: [bold]{optimizer_name}[/bold]")
    
    console.print(f"📈 Learning Rate: [bold]{lr:.0e}[/bold] (cosine decay, {WARMUP_STEPS} warmup)")
    
    # Select loss function
    if use_qfe:
        # Use physics-first WaveCoherenceLoss if available
        if PHYSICS_CORE_AVAILABLE and WaveCoherenceLoss is not None:
            loss_fn = WaveCoherenceLoss(
                lambda_phase=QFE_LAMBDA,
                lambda_energy=QFE_LAMBDA,
                lambda_harmonic=QFE_LAMBDA
            )
            console.print(f"🌌 Loss: [bold magenta]WaveCoherence[/bold magenta] (λ_phase={QFE_LAMBDA}, λ_energy={QFE_LAMBDA})")
        elif LEGACY_PHYSICS_AVAILABLE and QuantumFieldEntanglementLoss is not None:
            # Fall back to legacy QuantumFieldEntanglementLoss
            loss_fn = QuantumFieldEntanglementLoss(
                lambda_coherence=QFE_LAMBDA,
                amplitude_threshold=QFE_THRESHOLD
            )
            console.print(f"🌌 Loss: [bold magenta]QFE (legacy)[/bold magenta] (λ={QFE_LAMBDA}, threshold={QFE_THRESHOLD})")
        else:
            # No physics loss available, fall back to None (use model's CE)
            loss_fn = None
            console.print(f"📉 Loss: [bold yellow]Cross-Entropy (fallback)[/bold yellow] (physics losses unavailable)")
    else:
        loss_fn = None  # Use model's built-in CE loss
        console.print(f"📉 Loss: [bold]Cross-Entropy[/bold]")
    
    # Cosine schedule with warmup
    def get_lr(step):
        if step < WARMUP_STEPS:
            return lr * step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / (num_steps - WARMUP_STEPS)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    model.train()
    losses = []
    coherence_losses = []
    learning_rates = []
    
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
        task = progress.add_task(f"Training {config_name}", total=num_steps, loss=0.0)
        
        for step in range(num_steps):
            # Update learning rate
            current_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            total_tokens += x.numel()
            
            # Compute annealing ratio for wave embeddings
            # Decays linearly from 1.0 to 0.0 over ANNEALING_STEPS
            standard_embed_ratio = max(0.0, 1.0 - step / ANNEALING_STEPS)
            
            # Forward pass with annealing support
            # Check if model supports standard_embed_ratio parameter
            if hasattr(model, 'embedding') and hasattr(model.embedding, 'forward'):
                try:
                    logits, ce_loss = model(x, y, standard_embed_ratio=standard_embed_ratio)
                except TypeError:
                    # Model doesn't support standard_embed_ratio, use default
                    logits, ce_loss = model(x, y)
            else:
                logits, ce_loss = model(x, y)
            
            # Compute loss (WaveCoherenceLoss or standard CE)
            if use_qfe and loss_fn is not None:
                # Check if using new WaveCoherenceLoss (returns dict) or legacy QFE
                if PHYSICS_CORE_AVAILABLE and isinstance(loss_fn, WaveCoherenceLoss):
                    loss_dict = loss_fn(logits, y)
                    loss = loss_dict['total']
                    coherence_losses.append(loss_dict['coherence'].item())
                else:
                    # Legacy QFE loss
                    loss_dict = loss_fn(logits, y, return_components=True)
                    loss = loss_dict['total']
                    coherence_losses.append(loss_dict['coherence'].item())
            else:
                loss = ce_loss
            
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
            learning_rates.append(current_lr)
            progress.update(task, advance=1, loss=loss.item())
            
            # Log metrics if monitoring enabled
            if metrics_logger and metrics_logger.should_log(step):
                try:
                    metrics_dict = {
                        'loss': loss.item(),
                        'learning_rate': current_lr,
                        'tokens_per_sec': total_tokens / (time.perf_counter() - start_time) if (time.perf_counter() - start_time) > 0 else 0,
                        'standard_embed_ratio': standard_embed_ratio
                    }
                    
                    if use_qfe and coherence_losses:
                        metrics_dict['coherence_loss'] = coherence_losses[-1]
                    
                    metrics_logger.log_metrics(step, metrics_dict)
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to log metrics: {e}[/yellow]")
            
            # Save checkpoint if monitoring enabled
            if checkpoint_manager and checkpoint_manager.should_checkpoint(step):
                try:
                    checkpoint_config = {
                        'model_name': config_name,
                        'step': step,
                        'total_steps': num_steps
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
            
            # Generate visualizations if monitoring enabled
            if viz_manager and viz_manager.should_visualize(step):
                try:
                    viz_metrics = {
                        'learning_rate': learning_rates
                    }
                    if coherence_losses:
                        viz_metrics['coherence_loss'] = coherence_losses
                    
                    viz_manager.generate_training_plots(step, losses, viz_metrics)
                    viz_manager.generate_model_plots(step, model)
                    console.print(f"[green]✓ Visualizations generated at step {step}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to generate visualizations: {e}[/yellow]")
            
    # Log every 100 steps
            if (step + 1) % 100 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                console.print(f"Step {step+1:4d} | Loss: {loss.item():.4f} | Avg: {avg:.4f}")
    
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
    
    # Run WaveDiagnostics to verify wave signatures (physics-first validation)
    diagnostics_results = None
    if PHYSICS_CORE_AVAILABLE and WaveDiagnostics is not None:
        console.print("\n🔬 Running Wave Diagnostics...")
        try:
            diagnostics = WaveDiagnostics(model)
            
            # Analyze spectrum for harmonic peaks
            has_harmonics, spectrum_metrics = diagnostics.analyze_spectrum()
            console.print(f"  📊 Harmonic Peaks: {'[green]✓[/green]' if has_harmonics else '[yellow]✗[/yellow]'}")
            console.print(f"     Num peaks: {spectrum_metrics.get('num_peaks', 0)}")
            if spectrum_metrics.get('harmonic_ratios'):
                console.print(f"     Harmonic ratios: {spectrum_metrics['harmonic_ratios'][:3]}")
            
            # Visualize interference for periodic fringes
            has_fringes, interference_metrics = diagnostics.visualize_interference()
            console.print(f"  🌊 Interference Fringes: {'[green]✓[/green]' if has_fringes else '[yellow]✗[/yellow]'}")
            if interference_metrics.get('fringe_period', 0) > 0:
                console.print(f"     Fringe period: {interference_metrics['fringe_period']:.2f}")
            
            # Analyze trajectories for stable orbits
            sample_input = torch.randint(0, model.config.vocab_size, (1, 32), device=DEVICE)
            is_stable, trajectory_metrics = diagnostics.analyze_trajectories(sample_input)
            console.print(f"  🎯 Trajectory Stability: {'[green]✓[/green]' if is_stable else '[yellow]✗[/yellow]'}")
            console.print(f"     Lyapunov exponent: {trajectory_metrics.get('lyapunov_exponent', 'N/A'):.4f}")
            console.print(f"     Norm ratio: {trajectory_metrics.get('norm_ratio', 'N/A'):.2f}")
            
            diagnostics_results = {
                'has_harmonics': has_harmonics,
                'spectrum_metrics': spectrum_metrics,
                'has_fringes': has_fringes,
                'interference_metrics': interference_metrics,
                'is_stable': is_stable,
                'trajectory_metrics': trajectory_metrics
            }
            
            console.print("[green]✓ Wave diagnostics complete[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Wave diagnostics failed: {e}[/yellow]")
    
    # Save final results if monitoring enabled
    if config_tracker:
        try:
            final_metrics = {
                'val_loss': val_loss.item(),
                'perplexity': torch.exp(val_loss).item(),
                'total_time_seconds': elapsed,
                'tokens_per_second': speed,
                'peak_memory_mb': peak_mem,
                'total_steps': num_steps,
                'final_loss': losses[-1] if losses else 0.0
            }
            
            # Add diagnostics results if available
            if diagnostics_results:
                final_metrics['diagnostics'] = {
                    'has_harmonics': diagnostics_results.get('has_harmonics', False),
                    'has_fringes': diagnostics_results.get('has_fringes', False),
                    'is_stable': diagnostics_results.get('is_stable', False),
                    'lyapunov_exponent': diagnostics_results.get('trajectory_metrics', {}).get('lyapunov_exponent', None)
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
                generation_samples=[text[:200]]
            )
            console.print("[green]✓ Final results saved[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save results: {e}[/yellow]")
    
    return {
        "model": config_name,
        "params": params,
        "speed": speed,
        "val_loss": val_loss.item(),
        "perplexity": torch.exp(val_loss).item(),
        "time": elapsed,
        "peak_memory": peak_mem,
        "losses": losses,
        "model_ref": model,
        "experiment_dir": experiment_dir if enable_monitoring else None,
        "diagnostics": diagnostics_results
    }


def main():
    console = Console()
    console.print(Panel.fit(
        "[bold magenta]🌊 WAVE-NATIVE GPT BENCHMARK (PUSH TO MATCH!)[/bold magenta]\n\n"
        f"d_model={D_MODEL} | layers={NUM_LAYERS} | heads={NUM_HEADS} | waves={NUM_WAVES}×{NUM_HARMONICS}H\n"
        f"batch={BATCH_SIZE} | context={BLOCK_SIZE}\n"
        f"Classic: {CLASSIC_STEPS} steps @ {CLASSIC_LR:.0e} (AdamW + CE)\n"
        f"Wave:    {WAVE_STEPS} steps @ {WAVE_LR:.0e} (⚡RGD + 🌌QFE)",
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
    
    # Generate experiment ID for monitoring
    from monitoring import generate_experiment_id
    classic_exp_id = generate_experiment_id("classic_transformer")
    
    classic_config = GPTConfig(
        vocab_size=1024, d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, d_ff=D_MODEL*4, block_size=BLOCK_SIZE,
        dropout=0.1, layer_type="attention", weight_type="standard",
        num_waves=12, num_harmonics=5
    )
    classic_model = SpectralGPT(classic_config).to(DEVICE)
    
    res = run_wave_benchmark("Classic Transformer", classic_model, train_data, console, tokenizer, 
                             num_steps=CLASSIC_STEPS, lr=CLASSIC_LR, use_radam=False,
                             experiment_dir=classic_exp_id, enable_monitoring=True)
    if res:
        results.append(res)
    
    del classic_model
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    # ========================================
    # Model 2: Wave-Native GPT (PHYSICS-INFORMED!)
    # ========================================
    console.print("\n" + "="*60)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Generate experiment ID for monitoring
    wave_exp_id = generate_experiment_id("wave_native_gpt")
    
    wave_config = WaveGPTConfig(
        vocab_size=1024, d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, num_waves=NUM_WAVES, num_harmonics=NUM_HARMONICS,
        block_size=BLOCK_SIZE, dropout=0.1
    )
    wave_model = WaveGPT(wave_config).to(DEVICE)
    
    # Wave gets physics-informed optimization: RGD + QFE!
    res = run_wave_benchmark(
        "Wave-Native GPT 🌊⚡", wave_model, train_data, console, tokenizer, 
        num_steps=WAVE_STEPS, lr=WAVE_LR, 
        use_radam=False, use_rgd=USE_RGD, use_qfe=USE_QFE,
        experiment_dir=wave_exp_id, enable_monitoring=True
    )
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
    
    for r in results:
        table.add_row(
            r["model"],
            f"{r['params']/1e6:.2f}M",
            f"{r['speed']:,.0f}",
            f"{r['perplexity']:.2f}",
            f"{r['val_loss']:.4f}"
        )
    
    console.print(table)
    
    # ========================================
    # Save Models, Tokenizer, and Create ZIP
    # ========================================
    import json
    import shutil
    
    # Create organized folder structure
    output_dir = os.path.join(current_dir, "benchmark_results")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    console.print("\n💾 Saving models and tokenizer...")
    
    # Save each model
    for r in results:
        if r and 'model_ref' in r:
            model_name = r['model'].replace(' ', '_').replace('🌊', 'wave')
            model_path = os.path.join(models_dir, f"{model_name}.pt")
            
            # Save model state dict and config
            save_dict = {
                'state_dict': r['model_ref'].state_dict(),
                'config': r['model_ref'].config.__dict__ if hasattr(r['model_ref'], 'config') else {},
                'val_loss': r['val_loss'],
                'perplexity': r['perplexity'],
                'params': r['params']
            }
            torch.save(save_dict, model_path)
            console.print(f"  ✅ Saved {model_name} → {model_path}")
    
    # Save tokenizer (copy to output folder)
    tokenizer_out = os.path.join(output_dir, "tokenizer.json")
    console.print(f"  ✅ Tokenizer at {tokenizer_out}")
    
    # Save benchmark config
    config_path = os.path.join(output_dir, "benchmark_config.json")
    config_data = {
        "classic_steps": CLASSIC_STEPS,
        "wave_steps": WAVE_STEPS,
        "classic_lr": CLASSIC_LR,
        "wave_lr": WAVE_LR,
        "batch_size": BATCH_SIZE,
        "block_size": BLOCK_SIZE,
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "num_waves": NUM_WAVES,
        "num_harmonics": NUM_HARMONICS,
        "warmup_steps": WARMUP_STEPS,
        "results": [
            {
                "model": r["model"],
                "params": r["params"],
                "speed": r["speed"],
                "val_loss": r["val_loss"],
                "perplexity": r["perplexity"],
                "time_seconds": r["time"]
            }
            for r in results if r
        ]
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    console.print(f"  ✅ Saved config → {config_path}")
    
    # Create ZIP file for easy download
    console.print("\n📦 Creating ZIP archive for download...")
    zip_path = os.path.join(current_dir, "wave_gpt_benchmark_results")
    shutil.make_archive(zip_path, 'zip', output_dir)
    zip_file = f"{zip_path}.zip"
    
    # Get zip size
    zip_size = os.path.getsize(zip_file) / (1024 * 1024)  # MB
    
    console.print(f"\n[bold green]📦 ZIP READY FOR DOWNLOAD:[/bold green]")
    console.print(f"   [cyan]{zip_file}[/cyan]")
    console.print(f"   Size: {zip_size:.1f} MB")
    
    console.print(f"\n📁 Folder structure:")
    console.print("   benchmark_results/")
    console.print("   ├── models/")
    console.print("   │   ├── Classic_Transformer.pt")
    console.print("   │   └── Wave-Native_GPT_wave.pt")
    console.print("   ├── wave_gpt_plots/")
    console.print("   │   ├── *_learning_curve.png")
    console.print("   │   ├── *_frequencies.png")
    console.print("   │   ├── *_phases.png")
    console.print("   │   ├── *_harmonics.png")
    console.print("   │   ├── *_wave_packets.png")
    console.print("   │   ├── *_polar_phases.png")
    console.print("   │   ├── *_complex_plane.png")
    console.print("   │   ├── *_spectrogram.png")
    console.print("   │   ├── *_interference.png")
    console.print("   │   ├── *_wave_surface.png")
    console.print("   │   └── comparison_learning_curves.png")
    console.print("   ├── tokenizer.json")
    console.print("   └── benchmark_config.json")
    
    console.print("\n[bold green]✅ Benchmark Complete![/bold green]")
    console.print("\n💡 To download from Colab:")
    console.print("   from google.colab import files")
    console.print(f"   files.download('{zip_file}')")


if __name__ == "__main__":
    main()

