"""
Physics-Informed Optimization for Wave-Native GPT

Contains:
1. Resonant Gradient Descent (RGD) - Optimizer that aligns gradient spectrum with weight spectrum
2. Quantum Field Entanglement Loss (QFE) - Phase coherence loss in frequency domain

Both designed for wave-based neural networks with physics-first principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import math
from typing import Optional, Callable, Iterable, Tuple


# ==========================================
# Resonant Gradient Descent (RGD) Optimizer
# ==========================================

class ResonantGradientDescent(torch.optim.Optimizer):
    """
    Resonant Gradient Descent (RGD): Physics-informed optimizer for wave-based networks.
    
    Key Idea: Update weights more aggressively at frequencies where both the weight 
    AND gradient have significant magnitude (resonance condition).
    
    Mechanism:
    1. Transform weight W and gradient G to frequency domain via FFT
    2. Compute resonance factor ρ_k = f(|W_k|, |G_k|)
    3. Apply gated update: ΔW_k = -η * G_k * ρ_k
    4. Transform back to spatial domain
    
    Hybrid Warmup: To prevent the "bootstrap problem" (frequencies that start small
    stay small), we use α scheduling: ρ = α*resonance + (1-α)*uniform
    where α goes from 0→1 over warmup_steps.
    
    Args:
        params: Model parameters
        lr: Learning rate
        resonance_strength: How strongly to apply resonance gating (0=off, 1=full)
        warmup_steps: Steps over which to gradually enable resonance
        weight_decay: L2 regularization
        eps: Small constant for numerical stability
    """
    
    def __init__(
        self, 
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        resonance_strength: float = 0.5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        eps: float = 1e-8
    ):
        defaults = dict(
            lr=lr, 
            betas=betas,
            resonance_strength=resonance_strength,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            eps=eps
        )
        super().__init__(params, defaults)
        self.step_count = 0
        
    def _compute_resonance_factor(
        self, 
        weight_fft: torch.Tensor, 
        grad_fft: torch.Tensor,
        alpha: float,
        eps: float
    ) -> torch.Tensor:
        """
        Compute frequency-dependent resonance factor.
        
        Resonance is high when both weight and gradient have significant magnitude
        at a given frequency.
        
        Args:
            weight_fft: FFT of weights (complex)
            grad_fft: FFT of gradients (complex)
            alpha: Warmup factor (0=uniform, 1=full resonance)
            eps: Numerical stability
            
        Returns:
            Resonance factor ρ for each frequency component
        """
        # Magnitudes
        W_mag = torch.abs(weight_fft)
        G_mag = torch.abs(grad_fft)
        
        # Normalize magnitudes to [0, 1] range
        W_norm = W_mag / (W_mag.max() + eps)
        G_norm = G_mag / (G_mag.max() + eps)
        
        # Resonance factor: geometric mean of normalized magnitudes
        # High when BOTH are significant
        resonance = torch.sqrt(W_norm * G_norm + eps)
        
        # Hybrid: blend resonance with uniform (1.0) based on alpha
        # Early training: uniform updates (explore all frequencies)
        # Late training: resonance-gated updates (focus on active frequencies)
        rho = alpha * resonance + (1 - alpha) * torch.ones_like(resonance)
        
        return rho
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step with resonance gating."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.step_count += 1
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            resonance_strength = group['resonance_strength']
            warmup_steps = group['warmup_steps']
            weight_decay = group['weight_decay']
            eps = group['eps']
            
            # Compute warmup alpha: 0 → 1 over warmup_steps
            if warmup_steps > 0:
                alpha = min(1.0, self.step_count / warmup_steps) * resonance_strength
            else:
                alpha = resonance_strength
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Weight decay (AdamW style - before momentum)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Update momentum terms (Adam)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Corrected estimates
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Adam-style denominator
                denom = torch.sqrt(exp_avg_sq_corrected) + eps
                
                # Base update (Adam direction)
                update = exp_avg_corrected / denom
                
                # Apply resonance gating for 2D+ tensors (matrices, etc.)
                if p.dim() >= 2 and alpha > 0:
                    # Reshape to 2D for FFT
                    original_shape = p.shape
                    if p.dim() > 2:
                        p_2d = p.view(p.shape[0], -1)
                        update_2d = update.view(update.shape[0], -1)
                    else:
                        p_2d = p
                        update_2d = update
                    
                    # FFT of weights and update
                    weight_fft = fft.fft2(p_2d.float())
                    update_fft = fft.fft2(update_2d.float())
                    
                    # Compute resonance factor
                    rho = self._compute_resonance_factor(weight_fft, update_fft, alpha, eps)
                    
                    # Apply resonance gating in frequency domain
                    gated_fft = update_fft * rho
                    
                    # Transform back
                    gated_update = fft.ifft2(gated_fft).real
                    
                    # Reshape back
                    if p.dim() > 2:
                        gated_update = gated_update.view(original_shape)
                    
                    update = gated_update.to(update.dtype)
                
                # Apply update
                p.add_(update, alpha=-lr)
        
        return loss


# ==========================================
# Quantum Field Entanglement Loss (QFE)
# ==========================================

class QuantumFieldEntanglementLoss(nn.Module):
    """
    Quantum Field Entanglement Loss: Phase coherence in frequency domain.
    
    L_QFE = L_error + λ * L_coherence
    
    where L_coherence measures phase alignment between output and target
    in the frequency domain, weighted by amplitude.
    
    Key Insight: Standard CE loss treats tokens as independent symbols.
    QFE Loss treats sequences as wave fields that should be phase-locked.
    
    Amplitude Gating: To prevent phase singularities (undefined gradients
    when amplitude → 0), we only compute phase loss where both output 
    and target amplitudes exceed a threshold.
    
    Args:
        lambda_coherence: Weight for coherence term (start small: 0.01-0.1)
        amplitude_threshold: Minimum amplitude to compute phase loss
        temperature: Softmax temperature for logits → distribution
    """
    
    def __init__(
        self,
        lambda_coherence: float = 0.05,
        amplitude_threshold: float = 0.01,
        temperature: float = 1.0
    ):
        super().__init__()
        self.lambda_coherence = lambda_coherence
        self.amplitude_threshold = amplitude_threshold
        self.temperature = temperature
        
    def compute_spectral_coherence(
        self, 
        output: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phase coherence loss between output and target sequences.
        
        Args:
            output: Model output logits (B, T, V) or probabilities
            target: Target token IDs (B, T) - will be converted to one-hot
            
        Returns:
            Coherence loss (scalar)
        """
        B, T, V = output.shape
        
        # Convert output to probabilities
        probs = F.softmax(output / self.temperature, dim=-1)  # (B, T, V)
        
        # Convert target to one-hot
        target_onehot = F.one_hot(target, num_classes=V).float()  # (B, T, V)
        
        # Compute FFT along sequence dimension (T)
        output_fft = fft.rfft(probs, dim=1)  # (B, T//2+1, V)
        target_fft = fft.rfft(target_onehot, dim=1)  # (B, T//2+1, V)
        
        # Extract amplitude and phase
        A_out = torch.abs(output_fft)
        A_target = torch.abs(target_fft)
        
        phi_out = torch.angle(output_fft)
        phi_target = torch.angle(target_fft)
        
        # Phase difference
        phase_diff = phi_out - phi_target
        
        # Phase coherence penalty: 1 - cos(Δφ)
        # = 0 when phases match, = 2 when phases are opposite
        phase_penalty = 1 - torch.cos(phase_diff)
        
        # Amplitude weighting: focus on frequencies that matter
        magnitude_weight = A_out * A_target
        
        # Amplitude gating: only compute phase loss where both amplitudes are significant
        # This prevents phase singularities and gradient explosions
        amplitude_mask = (A_out > self.amplitude_threshold) & (A_target > self.amplitude_threshold)
        
        # Weighted phase loss with gating
        weighted_phase_loss = magnitude_weight * phase_penalty * amplitude_mask.float()
        
        # Normalize by number of valid frequencies
        num_valid = amplitude_mask.float().sum() + 1e-8
        coherence_loss = weighted_phase_loss.sum() / num_valid
        
        return coherence_loss
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined QFE loss.
        
        L = L_CE + λ * L_coherence
        
        Args:
            logits: Model output logits (B, T, V)
            targets: Target token IDs (B, T)
            return_components: If True, return dict with all components
            
        Returns:
            Total loss (or dict if return_components=True)
        """
        # Standard cross-entropy loss
        B, T, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            reduction='mean'
        )
        
        # Spectral coherence loss
        coherence_loss = self.compute_spectral_coherence(logits, targets)
        
        # Combined loss
        total_loss = ce_loss + self.lambda_coherence * coherence_loss
        
        if return_components:
            return {
                'total': total_loss,
                'ce': ce_loss,
                'coherence': coherence_loss,
                'lambda': self.lambda_coherence
            }
        
        return total_loss


# ==========================================
# Convenience Functions
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
    Create optimizer with optional resonance gating.
    
    Args:
        model: Neural network model
        lr: Learning rate
        use_resonance: Whether to use RGD (True) or standard AdamW (False)
        resonance_strength: How strongly to gate by resonance (0 to 1)
        warmup_steps: Steps to gradually enable resonance
        weight_decay: L2 regularization
        
    Returns:
        Optimizer instance
    """
    if use_resonance:
        return ResonantGradientDescent(
            model.parameters(),
            lr=lr,
            resonance_strength=resonance_strength,
            warmup_steps=warmup_steps,
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
    amplitude_threshold: float = 0.01
) -> nn.Module:
    """
    Create loss function with optional QFE coherence term.
    
    Args:
        use_qfe: Whether to use QFE loss (True) or standard CE (False)
        lambda_coherence: Weight for coherence term
        amplitude_threshold: Minimum amplitude for phase computation
        
    Returns:
        Loss function (callable)
    """
    if use_qfe:
        return QuantumFieldEntanglementLoss(
            lambda_coherence=lambda_coherence,
            amplitude_threshold=amplitude_threshold
        )
    else:
        # Return a simple wrapper for CE loss
        class CELossWrapper(nn.Module):
            def forward(self, logits, targets, return_components=False):
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
                if return_components:
                    return {'total': loss, 'ce': loss, 'coherence': torch.tensor(0.0)}
                return loss
        return CELossWrapper()


# ==========================================
# Testing / Demo
# ==========================================

if __name__ == "__main__":
    print("🔬 Testing Physics-Informed Optimization\n")
    
    # Test RGD
    print("⚡ Testing Resonant Gradient Descent...")
    model = nn.Linear(64, 64)
    optimizer = ResonantGradientDescent(model.parameters(), lr=1e-3, warmup_steps=100)
    
    for step in range(5):
        x = torch.randn(8, 64)
        y = model(x)
        loss = y.pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")
    
    print("  ✅ RGD working!\n")
    
    # Test QFE Loss
    print("🌌 Testing Quantum Field Entanglement Loss...")
    qfe_loss = QuantumFieldEntanglementLoss(lambda_coherence=0.1)
    
    B, T, V = 2, 32, 100
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    
    loss_dict = qfe_loss(logits, targets, return_components=True)
    print(f"  Total Loss: {loss_dict['total'].item():.4f}")
    print(f"  CE Loss: {loss_dict['ce'].item():.4f}")
    print(f"  Coherence Loss: {loss_dict['coherence'].item():.4f}")
    print("  ✅ QFE Loss working!\n")
    
    print("✨ All physics-informed components ready!")
