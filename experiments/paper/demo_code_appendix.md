# Appendix: Code Examples

This appendix provides key implementation details for reproducibility.

## A.1 Wave Packet Embedding

The core innovation of Spectral GPT is representing tokens as wave packets with learnable frequencies, phases, and harmonic amplitudes.

```python
# WavePacketEmbedding (class)
# Embed tokens as wave packets with HARMONICS.

class WavePacketEmbedding(nn.Module):
    """
    Embed tokens as wave packets with HARMONICS.
    
    Each token has:
    - Base frequencies (fundamental pitches)
    - Harmonic series (1f, 2f, 3f, 4f...) like a musical instrument
    - Learnable amplitude per harmonic (timbre)
    
    This creates richer superpositions that can represent complex patterns.
    
    Training Strategy: 
    - Start with small wave contribution + standard embedding (for gradient flow)
    - Gradually increase wave contribution as training progresses
    """

    def __init__(self, vocab_size, d_model, num_waves=16, num_harmonics=4):
        super().__init__()
        self.d_model = d_model
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # Base frequencies per token (fundamental pitch)
        # Different tokens resonate at different frequencies
        self.base_freqs = nn.Parameter(
            torch.linspace(0.5, 5.0, num_waves).unsqueeze(0).expand(vocab_size, -1).clone() +
            torch.randn(vocab_size, num_waves) * 0.1
        )
        
        # Harmonic multipliers: 1, 2, 3, 4... (physics-based, fixed)
        self.register_buffer('harmonic_mults', torch.arange(1, num_harmonics + 1).float())
        
        # Learnable amplitude per harmonic per wave per token
        # Initialize with meaningful scale for gradient flow
        self.harmonic_amps = nn.Parameter(
            torch.randn(vocab_size, num_waves, num_harmonics) * 0.5 / math.sqrt(num_harmonics * num_waves)
        )
        
        # Phases: where in the wave cycle does this token start?
        self.phases = nn.Parameter(torch.rand(vocab_size, num_waves) * 2 * math.pi)
        
        # Project wave state to d_model dimension
        # num_waves * num_harmonics * 2 (sin + cos)
        wave_dim = num_waves * num_harmonics * 2
        self.wave_to_embed = nn.Linear(wave_dim, d_model)
        
        # Positional wave modulation
        self.pos_freq = nn.Parameter(torch.randn(1, 1, num_waves) * 0.1)
        
        # Standard embedding as gradient bootstrap (will be phased out)
        self.simple_embed = nn.Embedding(vocab_size, d_model)
        # Start with 50% wave, 50% standard - wave will take over during training
        self.wave_ratio = nn.Parameter(torch.tensor(0.5))
        
        # LayerNorm for final output stability
        self.ln = nn.LayerNorm(d_model)

    def __init__(self, vocab_size, d_model, num_waves=16, num_harmonics=4):
        super().__init__()
        self.d_model = d_model
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # Base frequencies per token (fundamental pitch)
        # Different tokens resonate at different frequencies
        self.base_freqs = nn.Parameter(
            torch.linspace(0.5, 5.0, num_waves).unsqueeze(0).expand(vocab_size, -1).clone() +
            torch.randn(vocab_size, num_waves) * 0.1
        )
        
        # Harmonic multipliers: 1, 2, 3, 4... (physics-based, fixed)
        self.register_buffer('harmonic_mults', torch.arange(1, num_harmonics + 1).float())
        
        # Learnable amplitude per harmonic per wave per token
        # Initialize with meaningful scale for gradient flow
        self.harmonic_amps = nn.Parameter(
            torch.randn(vocab_size, num_waves, num_harmonics) * 0.5 / math.sqrt(num_harmonics * num_waves)
        )
        
        # Phases: where in the wave cycle does this token start?
    # ... (truncated)
```

## A.2 Interference Attention

Instead of dot-product attention, we use wave interference to compute attention weights based on phase relationships.

## A.3 Resonant Gradient Descent (RGD)

Physics-informed optimizer that filters gradients in the frequency domain, applying stronger updates at resonant frequencies.

```python
# ResonantGradientDescent (class)
# Resonant Gradient Descent (RGD): Physics-informed optimizer for wave-based networks.

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
    # ... (truncated)
```

## A.4 Quantum Field Entanglement (QFE) Loss

Coherence loss that enforces phase alignment between predictions and targets in the frequency domain.

```python
# QuantumFieldEntanglementLoss (class)
# Quantum Field Entanglement Loss: Phase coherence in frequency domain.

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
        temperature: float = 1.0,
        projection_dim: int = 256
    ):
        super().__init__()
        self.lambda_coherence = lambda_coherence
        self.amplitude_threshold = amplitude_threshold
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.register_buffer('proj_matrix', None) # Lazy init

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
    # ... (truncated)
```

## A.5 High-Level API Usage

Example of using the Spectral GPT API for training and inference.

```python
# High-Level API Usage Example

import torch
from spectral_gpt.wave_gpt import WaveGPT, WaveGPTConfig
from spectral_gpt.physics_optim import ResonantGradientDescent, QuantumFieldEntanglementLoss

# 1. Configure the model
config = WaveGPTConfig(
    vocab_size=50257,
    block_size=256,
    d_model=384,
    num_layers=6,
    num_heads=6,
    num_waves=48,          # Number of wave packets per token
    num_harmonics=4,       # Harmonics per wave (1f, 2f, 3f, 4f)
    dropout=0.1
)

# 2. Create the model
model = WaveGPT(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3. Set up physics-informed optimization
optimizer = ResonantGradientDescent(
    model.parameters(),
    lr=3e-4,
    resonance_strength=0.3,  # Frequency-domain gradient filtering
    warmup_steps=500
)

# 4. Set up coherence loss
qfe_loss = QuantumFieldEntanglementLoss(
    lambda_coherence=0.1  # Weight for phase coherence term
)

# 5. Training loop
model.train()
for batch in dataloader:
    input_ids, targets = batch
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss with phase coherence
    loss = qfe_loss(logits, targets)
    
    # Backward pass with resonant gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# 6. Generation
model.eval()
prompt = torch.tensor([[1, 2, 3]])  # Token IDs
generated = model.generate(prompt, max_new_tokens=100, temperature=0.8)

```

