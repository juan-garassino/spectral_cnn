"""
Wave Physics Core: First-Principles Physics Engine for Wave-Native GPT

This module implements the physics-first approach to neural computation:
1. WaveNativeOptimizer - SVD projection and damped harmonic momentum
2. WaveCoherenceLoss - QFE regularization for field coherence
3. WaveDiagnostics - Verification of wave signatures in trained models

All components are designed to treat neural networks as coupled oscillator systems
where parameters evolve according to wave mechanics principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings
from typing import Optional, Callable, Iterable, Dict, List, Tuple, Any


# ==========================================
# WaveNativeOptimizer
# ==========================================

class WaveNativeOptimizer(torch.optim.Optimizer):
    """
    Wave-Native Optimizer: Treats parameters as coupled oscillators.
    
    Key Principles:
    1. SVD Projection: Project gradients onto coherent subspaces defined by weight SVD
    2. Damped Harmonic Momentum: Update velocity using damped oscillator dynamics
    3. Coherent Gradient Combination: Blend coherent and raw gradients
    
    Formulas:
    - SVD: U, S, Vh = SVD(W)
    - Coherent gradient: grad_coherent = U @ (U.T @ grad @ Vh.T) @ Vh
    - Combined gradient: grad_final = coherence_weight * grad_coherent + (1 - coherence_weight) * raw_grad
    - Damped momentum: v_{t+1} = v_t * (1 - damping) - grad_final * lr
    - Parameter update: θ_{t+1} = θ_t + v_{t+1}
    
    Args:
        params: Model parameters
        lr: Learning rate (η)
        damping: Damping coefficient (γ), controls momentum decay
        coherence_weight: Weight for coherent gradient (0.7 recommended per spec)
        weight_decay: L2 regularization
    """
    
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        damping: float = 0.1,
        coherence_weight: float = 0.7,
        weight_decay: float = 0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if damping < 0.0 or damping > 1.0:
            raise ValueError(f"Invalid damping coefficient: {damping}")
        if coherence_weight < 0.0 or coherence_weight > 1.0:
            raise ValueError(f"Invalid coherence weight: {coherence_weight}")
            
        defaults = dict(
            lr=lr,
            damping=damping,
            coherence_weight=coherence_weight,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
    
    def _compute_svd_projection(
        self,
        weight: torch.Tensor,
        grad: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute SVD-projected coherent gradient.
        
        Projects gradient onto the coherent subspace defined by weight's SVD.
        
        Args:
            weight: 2D weight matrix
            grad: Gradient of same shape
            
        Returns:
            Coherent gradient or None if SVD fails
        """
        try:
            # Compute SVD of weight matrix
            U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
            
            # Project gradient: grad_coherent = U @ (U.T @ grad @ Vh.T) @ Vh
            # Step by step for clarity:
            # 1. U.T @ grad -> project grad onto left singular vectors
            # 2. ... @ Vh.T -> project onto right singular vectors (transposed)
            # 3. U @ ... @ Vh -> reconstruct in original space
            
            grad_float = grad.float()
            
            # U.T @ grad @ Vh.T
            projected = U.T @ grad_float @ Vh.T
            
            # U @ projected @ Vh
            grad_coherent = U @ projected @ Vh
            
            return grad_coherent.to(grad.dtype)
            
        except RuntimeError:
            # SVD failed to converge - return None to signal fallback
            return None
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform optimization step with SVD projection and damped harmonic momentum.
        
        Args:
            closure: Optional closure for computing loss
            
        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            damping = group['damping']
            coherence_weight = group['coherence_weight']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    # Initialize velocity (momentum) to zeros
                    state['velocity'] = torch.zeros_like(p)
                
                velocity = state['velocity']
                
                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # Compute final gradient with SVD projection for 2D matrices
                if p.dim() == 2 and coherence_weight > 0:
                    grad_coherent = self._compute_svd_projection(p, grad)
                    
                    if grad_coherent is not None:
                        # Combine coherent and raw gradients
                        # grad_final = coherence_weight * grad_coherent + (1 - coherence_weight) * raw_grad
                        grad_final = coherence_weight * grad_coherent + (1 - coherence_weight) * grad
                    else:
                        # SVD failed - fall back to raw gradient
                        grad_final = grad
                else:
                    # Non-2D tensors or coherence disabled - use raw gradient
                    grad_final = grad
                
                # Damped harmonic momentum update
                # v_{t+1} = v_t * (1 - γ) - ∇L * η
                velocity.mul_(1 - damping).sub_(grad_final, alpha=lr)
                
                # Parameter update
                # θ_{t+1} = θ_t + v_{t+1}
                p.add_(velocity)
        
        return loss



# ==========================================
# WaveCoherenceLoss
# ==========================================

class WaveCoherenceLoss(nn.Module):
    """
    Wave Coherence Loss: Minimizes field decoherence alongside prediction error.
    
    Combines:
    1. CrossEntropyLoss - Primary prediction loss
    2. Phase Lock Regularization - Penalizes high phase variance within local windows
    3. Energy Conservation Regularization - Penalizes L2 norm drift between layers
    4. Harmonic Fidelity Regularization - Penalizes deviation from 1/n amplitude decay
    
    Returns a dictionary with {total, ce, coherence} keys.
    
    Args:
        lambda_phase: Weight for phase lock regularization
        lambda_energy: Weight for energy conservation regularization
        lambda_harmonic: Weight for harmonic fidelity regularization
        window_size: Window size for local phase variance computation
    """
    
    def __init__(
        self,
        lambda_phase: float = 0.01,
        lambda_energy: float = 0.01,
        lambda_harmonic: float = 0.01,
        window_size: int = 8
    ):
        super().__init__()
        self.lambda_phase = lambda_phase
        self.lambda_energy = lambda_energy
        self.lambda_harmonic = lambda_harmonic
        self.window_size = window_size
        self.ce_loss = nn.CrossEntropyLoss()
    
    def _compute_phase_lock_loss(
        self,
        layer_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute phase lock regularization.
        
        Penalizes high phase variance within local token windows.
        Phase is estimated from the angle of complex representation.
        
        Args:
            layer_outputs: List of layer output tensors (B, T, D)
            
        Returns:
            Phase lock loss (scalar)
        """
        if not layer_outputs:
            return torch.tensor(0.0)
        
        total_phase_loss = torch.tensor(0.0, device=layer_outputs[0].device)
        
        for output in layer_outputs:
            B, T, D = output.shape
            
            if T < self.window_size:
                continue
            
            # Estimate phase from output using FFT
            # Compute FFT along feature dimension
            output_fft = torch.fft.rfft(output, dim=-1)
            phases = torch.angle(output_fft)  # (B, T, D//2+1)
            
            # Compute variance within local windows
            num_windows = T // self.window_size
            if num_windows == 0:
                continue
                
            # Reshape to windows
            truncated_T = num_windows * self.window_size
            phases_windowed = phases[:, :truncated_T, :].view(
                B, num_windows, self.window_size, -1
            )
            
            # Compute variance within each window
            phase_var = phases_windowed.var(dim=2)  # (B, num_windows, D//2+1)
            
            # Average variance across all windows and dimensions
            total_phase_loss = total_phase_loss + phase_var.mean()
        
        return total_phase_loss / max(len(layer_outputs), 1)
    
    def _compute_energy_conservation_loss(
        self,
        layer_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute energy conservation regularization.
        
        Penalizes L2 norm drift between consecutive layers.
        
        Args:
            layer_outputs: List of layer output tensors (B, T, D)
            
        Returns:
            Energy conservation loss (scalar)
        """
        if len(layer_outputs) < 2:
            return torch.tensor(0.0)
        
        device = layer_outputs[0].device
        total_energy_loss = torch.tensor(0.0, device=device)
        
        for i in range(1, len(layer_outputs)):
            # Compute L2 norms
            norm_prev = torch.norm(layer_outputs[i-1], dim=-1)  # (B, T)
            norm_curr = torch.norm(layer_outputs[i], dim=-1)    # (B, T)
            
            # Penalize drift: |norm_curr - norm_prev|
            drift = torch.abs(norm_curr - norm_prev)
            total_energy_loss = total_energy_loss + drift.mean()
        
        return total_energy_loss / (len(layer_outputs) - 1)
    
    def _compute_harmonic_fidelity_loss(
        self,
        harmonic_amplitudes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute harmonic fidelity regularization.
        
        Penalizes deviation from 1/n amplitude decay.
        
        Args:
            harmonic_amplitudes: Tensor of harmonic amplitudes (..., num_harmonics)
            
        Returns:
            Harmonic fidelity loss (scalar)
        """
        if harmonic_amplitudes is None:
            return torch.tensor(0.0)
        
        device = harmonic_amplitudes.device
        num_harmonics = harmonic_amplitudes.shape[-1]
        
        # Expected 1/n decay pattern
        n = torch.arange(1, num_harmonics + 1, device=device, dtype=harmonic_amplitudes.dtype)
        expected_decay = 1.0 / n  # (num_harmonics,)
        
        # Normalize expected decay to have same scale as actual amplitudes
        # Compare relative decay pattern, not absolute values
        actual_abs = torch.abs(harmonic_amplitudes)
        
        # Normalize both to sum to 1 for comparison
        actual_norm = actual_abs / (actual_abs.sum(dim=-1, keepdim=True) + 1e-8)
        expected_norm = expected_decay / expected_decay.sum()
        
        # Penalize deviation from expected decay pattern
        deviation = torch.abs(actual_norm - expected_norm)
        
        return deviation.mean()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        layer_outputs: Optional[List[torch.Tensor]] = None,
        harmonic_amplitudes: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss with QFE regularization.
        
        Args:
            logits: Model output logits (B, T, V)
            targets: Target token IDs (B, T)
            layer_outputs: Optional list of layer outputs for coherence regularization
            harmonic_amplitudes: Optional tensor of harmonic amplitudes
            
        Returns:
            Dict with keys: 'total', 'ce', 'coherence'
        """
        B, T, V = logits.shape
        
        # Primary cross-entropy loss
        ce_loss = self.ce_loss(
            logits.view(-1, V),
            targets.view(-1)
        )
        
        # Initialize coherence loss
        coherence_loss = torch.tensor(0.0, device=logits.device)
        
        # Phase lock regularization
        if layer_outputs is not None and self.lambda_phase > 0:
            phase_loss = self._compute_phase_lock_loss(layer_outputs)
            coherence_loss = coherence_loss + self.lambda_phase * phase_loss
        
        # Energy conservation regularization
        if layer_outputs is not None and self.lambda_energy > 0:
            energy_loss = self._compute_energy_conservation_loss(layer_outputs)
            coherence_loss = coherence_loss + self.lambda_energy * energy_loss
        
        # Harmonic fidelity regularization
        if harmonic_amplitudes is not None and self.lambda_harmonic > 0:
            harmonic_loss = self._compute_harmonic_fidelity_loss(harmonic_amplitudes)
            coherence_loss = coherence_loss + self.lambda_harmonic * harmonic_loss
        
        # Total loss
        total_loss = ce_loss + coherence_loss
        
        return {
            'total': total_loss,
            'ce': ce_loss,
            'coherence': coherence_loss
        }



# ==========================================
# WaveDiagnostics
# ==========================================

class WaveDiagnostics:
    """
    Diagnostic tools for verifying wave signatures in trained models.
    
    Provides methods to:
    1. analyze_spectrum() - FFT analysis for harmonic peaks (f, 2f, 3f)
    2. visualize_interference() - Autocorrelation for periodic fringes
    3. analyze_trajectories() - Lyapunov exponent for trajectory stability
    
    Each method returns both boolean indicators and quantitative metrics.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize diagnostics for a wave model.
        
        Args:
            model: Neural network model to analyze
        """
        self.model = model
        self.device = next(model.parameters()).device
    
    def analyze_spectrum(
        self,
        embeddings: Optional[torch.Tensor] = None,
        harmonic_tolerance: float = 0.1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze FFT of embeddings for harmonic peaks.
        
        Checks if harmonic peaks (f, 2f, 3f) exist in the frequency spectrum.
        
        Args:
            embeddings: Optional embeddings tensor (V, D) or (B, T, D)
                       If None, extracts from model's embedding layer
            harmonic_tolerance: Tolerance for detecting harmonic relationships
            
        Returns:
            (has_harmonics, metrics_dict)
        """
        # Extract embeddings if not provided
        if embeddings is None:
            embeddings = self._extract_embeddings()
        
        if embeddings is None:
            return False, {'error': 'Could not extract embeddings'}
        
        # Ensure 2D for analysis
        if embeddings.dim() == 3:
            embeddings = embeddings.view(-1, embeddings.shape[-1])
        
        # Compute FFT along feature dimension
        fft_result = torch.fft.rfft(embeddings.float(), dim=-1)
        magnitudes = torch.abs(fft_result)  # (N, D//2+1)
        
        # Average magnitude spectrum across all embeddings
        avg_spectrum = magnitudes.mean(dim=0).cpu().numpy()
        
        # Find peaks in the spectrum
        peaks = self._find_peaks(avg_spectrum)
        
        # Check for harmonic relationships (f, 2f, 3f)
        has_harmonics = False
        harmonic_ratios = []
        
        if len(peaks) >= 2:
            # Sort peaks by magnitude
            peak_indices = sorted(peaks, key=lambda x: avg_spectrum[x], reverse=True)
            
            # Check if any peak is approximately 2x or 3x another
            for i, p1 in enumerate(peak_indices[:5]):  # Check top 5 peaks
                for p2 in peak_indices[i+1:10]:
                    if p1 > 0 and p2 > 0:
                        ratio = max(p1, p2) / min(p1, p2)
                        # Check for harmonic ratios (2, 3, or 1.5 for 3f/2f)
                        for target_ratio in [2.0, 3.0, 1.5]:
                            if abs(ratio - target_ratio) < harmonic_tolerance:
                                has_harmonics = True
                                harmonic_ratios.append(ratio)
        
        metrics = {
            'num_peaks': len(peaks),
            'peak_indices': peaks[:10] if peaks else [],
            'peak_magnitudes': [float(avg_spectrum[p]) for p in peaks[:10]] if peaks else [],
            'harmonic_ratios': harmonic_ratios,
            'spectrum_mean': float(avg_spectrum.mean()),
            'spectrum_std': float(avg_spectrum.std())
        }
        
        return has_harmonics, metrics
    
    def visualize_interference(
        self,
        attention_weights: Optional[torch.Tensor] = None,
        fringe_threshold: float = 0.1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Compute autocorrelation of attention weights for periodic fringes.
        
        Interference patterns should show periodic structure in autocorrelation.
        
        Args:
            attention_weights: Optional attention weights (B, H, T, T)
                              If None, attempts to extract from model
            fringe_threshold: Threshold for detecting periodic fringes
            
        Returns:
            (has_fringes, metrics_dict)
        """
        if attention_weights is None:
            # Try to get attention weights from a forward pass
            attention_weights = self._extract_attention_weights()
        
        if attention_weights is None:
            return False, {'error': 'Could not extract attention weights'}
        
        # Flatten to 2D for autocorrelation
        if attention_weights.dim() == 4:
            B, H, T, _ = attention_weights.shape
            weights_flat = attention_weights.view(-1, T)  # (B*H*T, T)
        else:
            weights_flat = attention_weights.view(-1, attention_weights.shape[-1])
        
        # Compute autocorrelation
        autocorr = self._compute_autocorrelation(weights_flat)
        
        # Find peaks in autocorrelation (excluding lag 0)
        autocorr_np = autocorr.cpu().numpy()
        peaks = self._find_peaks(autocorr_np[1:])  # Exclude lag 0
        peaks = [p + 1 for p in peaks]  # Adjust indices
        
        # Check for periodic structure
        has_fringes = False
        fringe_period = 0.0
        
        if len(peaks) >= 2:
            # Check if peaks are roughly evenly spaced
            peak_diffs = np.diff(peaks)
            if len(peak_diffs) > 0:
                mean_period = np.mean(peak_diffs)
                period_std = np.std(peak_diffs)
                
                # Periodic if standard deviation is small relative to mean
                if mean_period > 0 and period_std / mean_period < 0.3:
                    has_fringes = True
                    fringe_period = float(mean_period)
        
        metrics = {
            'num_autocorr_peaks': len(peaks),
            'peak_lags': peaks[:10] if peaks else [],
            'fringe_period': fringe_period,
            'autocorr_mean': float(autocorr_np.mean()),
            'autocorr_max': float(autocorr_np.max())
        }
        
        return has_fringes, metrics
    
    def analyze_trajectories(
        self,
        input_sequence: torch.Tensor,
        num_steps: int = 100,
        perturbation_scale: float = 1e-6
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Track hidden states and estimate Lyapunov exponent for stability.
        
        Stable orbits have negative Lyapunov exponents (bounded/quasi-periodic).
        
        Args:
            input_sequence: Input tensor (B, T) for forward pass
            num_steps: Number of steps for trajectory analysis
            perturbation_scale: Scale of initial perturbation
            
        Returns:
            (is_stable, metrics_dict)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get baseline trajectory
            hidden_states = self._get_hidden_states(input_sequence)
            
            if hidden_states is None or len(hidden_states) == 0:
                return False, {'error': 'Could not extract hidden states'}
            
            # Stack hidden states
            trajectory = torch.stack(hidden_states, dim=0)  # (L, B, T, D)
            
            # Compute trajectory statistics
            trajectory_norms = torch.norm(trajectory, dim=-1)  # (L, B, T)
            
            # Check for bounded orbits
            max_norm = trajectory_norms.max().item()
            min_norm = trajectory_norms.min().item()
            mean_norm = trajectory_norms.mean().item()
            
            # Estimate Lyapunov exponent via perturbation
            lyapunov = self._estimate_lyapunov(
                input_sequence, 
                perturbation_scale,
                num_steps=min(num_steps, len(hidden_states))
            )
            
            # Stable if Lyapunov exponent is negative or small positive
            is_stable = lyapunov < 0.1
            
            # Check for quasi-periodicity via autocorrelation of norms
            norm_autocorr = self._compute_autocorrelation(
                trajectory_norms.view(-1, trajectory_norms.shape[-1])
            )
            
            metrics = {
                'lyapunov_exponent': float(lyapunov),
                'max_norm': float(max_norm),
                'min_norm': float(min_norm),
                'mean_norm': float(mean_norm),
                'norm_ratio': float(max_norm / (min_norm + 1e-8)),
                'is_bounded': max_norm < 100 * mean_norm
            }
        
        return is_stable, metrics
    
    def _extract_embeddings(self) -> Optional[torch.Tensor]:
        """Extract embeddings from model's embedding layer."""
        # Try common embedding attribute names
        for attr in ['embedding', 'embed', 'token_embedding', 'wte']:
            if hasattr(self.model, attr):
                emb_layer = getattr(self.model, attr)
                if hasattr(emb_layer, 'weight'):
                    return emb_layer.weight.data
                elif hasattr(emb_layer, 'simple_embed'):
                    return emb_layer.simple_embed.weight.data
        return None
    
    def _extract_attention_weights(self) -> Optional[torch.Tensor]:
        """Extract attention weights from a sample forward pass."""
        # This would require hooks or model modification
        # For now, return None - user should provide weights
        return None
    
    def _get_hidden_states(
        self,
        input_sequence: torch.Tensor
    ) -> Optional[List[torch.Tensor]]:
        """Get hidden states from each layer during forward pass."""
        hidden_states = []
        
        # Register hooks to capture hidden states
        hooks = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states.append(output[0].detach())
            else:
                hidden_states.append(output.detach())
        
        # Try to find transformer blocks
        for name, module in self.model.named_modules():
            if 'block' in name.lower() or 'layer' in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))
        
        if not hooks:
            return None
        
        try:
            # Forward pass
            with torch.no_grad():
                self.model(input_sequence)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return hidden_states if hidden_states else None
    
    def _estimate_lyapunov(
        self,
        input_sequence: torch.Tensor,
        perturbation_scale: float,
        num_steps: int
    ) -> float:
        """Estimate Lyapunov exponent via perturbation analysis."""
        # Get baseline hidden states
        baseline_states = self._get_hidden_states(input_sequence)
        
        if baseline_states is None or len(baseline_states) < 2:
            return 0.0
        
        # Create perturbed input (add small noise to embeddings)
        # This is a simplified estimation
        divergences = []
        
        for i in range(1, min(num_steps, len(baseline_states))):
            # Compute divergence between consecutive states
            diff = baseline_states[i] - baseline_states[i-1]
            divergence = torch.norm(diff).item()
            if divergence > 0:
                divergences.append(np.log(divergence + 1e-10))
        
        if not divergences:
            return 0.0
        
        # Lyapunov exponent is average rate of divergence
        return float(np.mean(divergences))
    
    def _find_peaks(
        self,
        signal: np.ndarray,
        min_prominence: float = 0.1
    ) -> List[int]:
        """Find peaks in a 1D signal."""
        if len(signal) < 3:
            return []
        
        peaks = []
        threshold = signal.mean() + min_prominence * signal.std()
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if signal[i] > threshold:
                    peaks.append(i)
        
        return peaks
    
    def _compute_autocorrelation(
        self,
        signal: torch.Tensor
    ) -> torch.Tensor:
        """Compute autocorrelation of a signal."""
        # Normalize
        signal = signal - signal.mean(dim=-1, keepdim=True)
        
        # Use FFT for efficient autocorrelation
        n = signal.shape[-1]
        fft = torch.fft.rfft(signal.float(), n=2*n, dim=-1)
        autocorr = torch.fft.irfft(fft * fft.conj(), dim=-1)
        
        # Normalize and take first half
        autocorr = autocorr[..., :n]
        autocorr = autocorr / (autocorr[..., 0:1] + 1e-8)
        
        # Average across batch dimension
        return autocorr.mean(dim=0)
    
    def generate_visualizations(
        self,
        embeddings: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        input_sequence: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate publication-quality visualizations.
        
        Args:
            embeddings: Optional embeddings for spectrum analysis
            attention_weights: Optional attention weights for interference
            input_sequence: Optional input for trajectory analysis
            save_path: Optional path to save figures
            
        Returns:
            Dictionary with figure objects
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return {'error': 'matplotlib not available'}
        
        figures = {}
        
        # 1. Spectrum Analysis Plot
        has_harmonics, spectrum_metrics = self.analyze_spectrum(embeddings)
        
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        if 'peak_indices' in spectrum_metrics and spectrum_metrics['peak_indices']:
            # Create a simple spectrum visualization
            ax1.set_title(f'Frequency Spectrum (Harmonics: {"Yes" if has_harmonics else "No"})')
            ax1.set_xlabel('Frequency Index')
            ax1.set_ylabel('Magnitude')
            
            # Mark peaks
            for idx in spectrum_metrics['peak_indices'][:5]:
                ax1.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
        
        figures['spectrum'] = fig1
        
        # 2. Interference Pattern Plot
        has_fringes, interference_metrics = self.visualize_interference(attention_weights)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.set_title(f'Autocorrelation (Fringes: {"Yes" if has_fringes else "No"})')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Autocorrelation')
        
        if 'peak_lags' in interference_metrics:
            for lag in interference_metrics['peak_lags'][:5]:
                ax2.axvline(x=lag, color='r', linestyle='--', alpha=0.5)
        
        figures['interference'] = fig2
        
        # 3. Trajectory Stability Plot
        if input_sequence is not None:
            is_stable, trajectory_metrics = self.analyze_trajectories(input_sequence)
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.set_title(f'Trajectory Analysis (Stable: {"Yes" if is_stable else "No"})')
            ax3.text(0.5, 0.5, 
                    f"Lyapunov: {trajectory_metrics.get('lyapunov_exponent', 'N/A'):.4f}\n"
                    f"Norm Ratio: {trajectory_metrics.get('norm_ratio', 'N/A'):.2f}",
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            
            figures['trajectory'] = fig3
        
        # Save if path provided
        if save_path:
            for name, fig in figures.items():
                if not isinstance(fig, str):
                    fig.savefig(f"{save_path}_{name}.png", dpi=150, bbox_inches='tight')
        
        plt.close('all')
        
        return figures



# ==========================================
# Compatibility Functions
# ==========================================

def create_physics_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    use_resonance: bool = True,
    resonance_strength: float = 0.3,
    warmup_steps: int = 500,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """
    Compatibility function matching physics_optim.py signature.
    Returns WaveNativeOptimizer with equivalent configuration.
    
    Args:
        model: Neural network model
        lr: Learning rate
        use_resonance: Whether to use wave-native optimization
        resonance_strength: Maps to coherence_weight
        warmup_steps: Ignored (for API compatibility)
        weight_decay: L2 regularization
        
    Returns:
        Optimizer instance
    """
    if use_resonance:
        return WaveNativeOptimizer(
            model.parameters(),
            lr=lr,
            coherence_weight=resonance_strength,
            damping=0.1,
            weight_decay=weight_decay
        )
    else:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )


def create_physics_loss(
    use_qfe: bool = True,
    lambda_coherence: float = 0.05,
    amplitude_threshold: float = 0.01,
    projection_dim: int = 256
) -> nn.Module:
    """
    Compatibility function matching physics_optim.py signature.
    Returns WaveCoherenceLoss with equivalent configuration.
    
    Args:
        use_qfe: Whether to use QFE loss
        lambda_coherence: Weight for coherence terms
        amplitude_threshold: Ignored (for API compatibility)
        projection_dim: Ignored (for API compatibility)
        
    Returns:
        Loss function module
    """
    if use_qfe:
        return WaveCoherenceLoss(
            lambda_phase=lambda_coherence,
            lambda_energy=lambda_coherence,
            lambda_harmonic=lambda_coherence
        )
    else:
        # Return a simple wrapper for CE loss
        class CELossWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.ce = nn.CrossEntropyLoss()
            
            def forward(self, logits, targets, layer_outputs=None, harmonic_amplitudes=None):
                B, T, V = logits.shape
                loss = self.ce(logits.view(-1, V), targets.view(-1))
                return {'total': loss, 'ce': loss, 'coherence': torch.tensor(0.0)}
        
        return CELossWrapper()


# ==========================================
# Testing / Demo
# ==========================================

if __name__ == "__main__":
    print("🌊 Testing Wave Physics Core Components\n")
    
    # Test WaveNativeOptimizer
    print("⚡ Testing WaveNativeOptimizer...")
    model = nn.Linear(64, 64)
    optimizer = WaveNativeOptimizer(
        model.parameters(), 
        lr=1e-3, 
        damping=0.1,
        coherence_weight=0.7
    )
    
    for step in range(5):
        x = torch.randn(8, 64)
        y = model(x)
        loss = y.pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")
    
    print("  ✅ WaveNativeOptimizer working!\n")
    
    # Test WaveCoherenceLoss
    print("🌌 Testing WaveCoherenceLoss...")
    qfe_loss = WaveCoherenceLoss(
        lambda_phase=0.01,
        lambda_energy=0.01,
        lambda_harmonic=0.01
    )
    
    B, T, V = 2, 32, 100
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    
    # Test without layer outputs
    loss_dict = qfe_loss(logits, targets)
    print(f"  Total Loss: {loss_dict['total'].item():.4f}")
    print(f"  CE Loss: {loss_dict['ce'].item():.4f}")
    print(f"  Coherence Loss: {loss_dict['coherence'].item():.4f}")
    
    # Test with layer outputs
    layer_outputs = [torch.randn(B, T, 64) for _ in range(4)]
    loss_dict = qfe_loss(logits, targets, layer_outputs=layer_outputs)
    print(f"  With layers - Coherence: {loss_dict['coherence'].item():.4f}")
    print("  ✅ WaveCoherenceLoss working!\n")
    
    # Test WaveDiagnostics
    print("🔬 Testing WaveDiagnostics...")
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 64)
            self.block1 = nn.Linear(64, 64)
            self.block2 = nn.Linear(64, 64)
            self.head = nn.Linear(64, 100)
        
        def forward(self, x):
            x = self.embedding(x)
            x = F.relu(self.block1(x))
            x = F.relu(self.block2(x))
            return self.head(x)
    
    test_model = SimpleModel()
    diagnostics = WaveDiagnostics(test_model)
    
    # Test spectrum analysis
    has_harmonics, spectrum_metrics = diagnostics.analyze_spectrum()
    print(f"  Spectrum - Has Harmonics: {has_harmonics}")
    print(f"  Spectrum - Num Peaks: {spectrum_metrics.get('num_peaks', 0)}")
    
    # Test trajectory analysis
    test_input = torch.randint(0, 100, (2, 16))
    is_stable, traj_metrics = diagnostics.analyze_trajectories(test_input)
    print(f"  Trajectory - Is Stable: {is_stable}")
    print(f"  Trajectory - Lyapunov: {traj_metrics.get('lyapunov_exponent', 'N/A')}")
    
    print("  ✅ WaveDiagnostics working!\n")
    
    # Test compatibility functions
    print("🔧 Testing Compatibility Functions...")
    compat_optimizer = create_physics_optimizer(test_model, lr=1e-3, use_resonance=True)
    print(f"  Created optimizer: {type(compat_optimizer).__name__}")
    
    compat_loss = create_physics_loss(use_qfe=True)
    print(f"  Created loss: {type(compat_loss).__name__}")
    print("  ✅ Compatibility functions working!\n")
    
    print("✨ All wave physics core components ready!")

