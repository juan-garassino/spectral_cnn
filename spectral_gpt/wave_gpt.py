"""
Wave-Native GPT: Language Modeling via Continuous Wave Functions

Philosophy: "Everything is a mass on a spring"
- Tokens are embedded as wave packets (frequency, phase, amplitude)
- Computation happens via wave interference
- Discretization only at output (measurement/collapse)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# ==========================================
# Configuration
# ==========================================

@dataclass
class WaveGPTConfig:
    vocab_size: int
    d_model: int           # Embedding dimension (also wave dimension)
    num_layers: int
    num_heads: int
    num_waves: int         # Number of wave components per token
    num_harmonics: int = 4 # Harmonics per wave (1f, 2f, 3f, 4f)
    block_size: int = 256  # Context length
    dropout: float = 0.1
    pure_wave_attention: bool = False  # True = NO SOFTMAX, pure interference
    pure_wave_kernel: str = "elu_plus_one" # Kernel for pure wave: 'elu_plus_one', 'sigmoid', 'exp'
    pure_wave_mode: str = "quadratic"      # 'quadratic' (N^2, exact kernel) or 'linear' (N, decomposable)
    model_type: str = "wave"               # "wave" or "standard"
    use_interference_attention: bool = False  # True = physics-based interference attention (Req 2.1-2.5)
    use_wave_embeddings: bool = True       # True = WavePacketEmbedding, False = StandardEmbedding (Req 7.5)

# ==========================================
# Standard Transformer Components (The Control)
# ==========================================

class StandardEmbedding(nn.Module):
    """Standard Token + Positional Embedding"""
    def __init__(self, vocab_size, d_model, block_size, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
        
        tok_emb = self.token_embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding(pos) # (1, T, C)
        
        x = self.dropout(tok_emb + pos_emb)
        return x

class StandardCausalSelfAttention(nn.Module):
    """Vanilla Multi-Head Attention with Causal Mask"""
    def __init__(self, d_model, num_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = num_heads
        self.n_embd = d_model
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class StandardBlock(nn.Module):
    """Transformer Block: LN -> Attn -> LN -> MLP"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = StandardCausalSelfAttention(config.d_model, config.num_heads, config.block_size, config.dropout)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x



# ==========================================
# Wave Packet Embedding
# ==========================================

class WavePacketEmbedding(nn.Module):
    """
    Physics-aware token embeddings based on mass-frequency relationships.
    
    Tokens are quantized resonant modes following natural physical laws:
    - Token mass follows Zipfian distribution: Mass(i) = 1/(i+1)
    - Base frequency: ω_0 = 1.0 / sqrt(Mass) (heavy tokens = low freq, light tokens = high freq)
    - Harmonic frequencies: ω_n = n * ω_0 (initialized as integer multiples, but LEARNABLE)
    - Harmonic amplitudes: A_n = 1/n (initialized with power law decay, but LEARNABLE)
    
    Training Strategy:
    - Supports annealing via standard_embed_ratio parameter
    - out = (1-r)*wave + r*standard where r decays from 1.0 to 0.0
    
    Key change from original: frequencies and amplitudes are now LEARNABLE parameters
    initialized with physics-based values, allowing the model to adapt while starting
    from a physically meaningful prior.
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """
    def __init__(self, vocab_size, d_model, num_waves=16, num_harmonics=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # === Requirement 1.1: Zipfian mass distribution ===
        # Mass(i) = 1/(i+1) - stored as buffer (reference only)
        token_indices = torch.arange(vocab_size, dtype=torch.float32)
        masses = 1.0 / (token_indices + 1.0)  # Zipfian: Mass(i) = 1/(i+1)
        self.register_buffer('masses', masses)  # (vocab_size,)
        
        # === Requirement 1.2: MULTI-SCALE frequency spectrum ===
        # Each token gets frequencies at DIFFERENT SCALES for multi-resolution attention:
        # - Low freq (0.01-0.1): peaks every 60-600 tokens → document-level context
        # - Mid freq (0.1-1.0): peaks every 6-60 tokens → paragraph-level context  
        # - High freq (1.0-10.0): peaks every 0.6-6 tokens → local/word-level context
        #
        # This is like wavelets - different frequencies capture different scales!
        
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio for spacing
        
        # Create LOG-SPACED frequencies from 0.01 to 10.0 (3 orders of magnitude)
        # This ensures we have low, medium, AND high frequencies
        freq_min, freq_max = 0.02, 5.0  # Cycles per position
        log_freqs = torch.logspace(math.log10(freq_min), math.log10(freq_max), num_waves)
        
        # Add small token-dependent offset using mass (common tokens slightly lower freq)
        # This gives each token a unique "fingerprint" while keeping the multi-scale structure
        raw_center_freq = 1.0 / torch.sqrt(masses)  # (vocab_size,)
        token_offset = torch.log1p(raw_center_freq) / 10.0  # Small offset [0.07, 0.54]
        
        # Each token gets the full spectrum, slightly shifted by its mass
        # Shape: (vocab_size, num_waves)
        init_base_frequencies = log_freqs.unsqueeze(0) * (1.0 + token_offset.unsqueeze(1) * 0.1)
        self.base_freqs = nn.Parameter(init_base_frequencies.clone())  # LEARNABLE!
        
        # === Requirement 1.3: Harmonic quantization ===
        # Harmonic multipliers - keep as buffer (structural, not learned)
        harmonic_mults = torch.arange(1, num_harmonics + 1, dtype=torch.float32)  # [1, 2, 3, 4, ...]
        self.register_buffer('harmonic_mults', harmonic_mults)
        
        # === Requirement 1.4: Power law amplitude decay ===
        # LEARNABLE amplitudes initialized with physics prior: A_n = 1/n
        # But the model can learn what harmonics matter!
        init_harmonic_amplitudes = 1.0 / harmonic_mults  # [1, 0.5, 0.333, 0.25, ...]
        init_harmonic_amplitudes = init_harmonic_amplitudes.view(1, 1, num_harmonics).expand(vocab_size, num_waves, -1).clone()
        self.harmonic_amps = nn.Parameter(init_harmonic_amplitudes)  # NOW LEARNABLE!
        
        # === Phases: Random initialization with structure ===
        # Golden angle creates too rigid a pattern. Use random init with some structure:
        # - Random base phase per token (allows learning unique token "signatures")
        # - Small wave-dependent offset (breaks symmetry between waves)
        
        golden_angle = 2 * math.pi / (phi ** 2)
        
        # Random phase per token, plus small structured offset per wave
        random_token_phase = torch.rand(vocab_size, 1) * 2 * math.pi
        wave_offset = torch.arange(num_waves, dtype=torch.float32).unsqueeze(0) * golden_angle * 0.1
        
        # Add small random noise to break any remaining patterns
        noise = torch.randn(vocab_size, num_waves) * 0.3
        
        init_phases = (random_token_phase + wave_offset + noise) % (2 * math.pi)
        self.phases = nn.Parameter(init_phases)  # LEARNABLE!
        
        # Project wave state to d_model dimension
        # num_waves * num_harmonics * 2 (sin + cos) + num_waves (phase) + num_waves (freq)
        wave_dim = num_waves * num_harmonics * 2 + num_waves * 2
        self.wave_to_embed = nn.Linear(wave_dim, d_model)
        
        # Initialize wave_to_embed with slightly larger weights for phase/freq pathways
        # This ensures gradients flow back to phases and frequencies
        # NOTE: 10x was too aggressive and caused instability, using 3x instead
        with torch.no_grad():
            sin_cos_dim = num_waves * num_harmonics * 2
            self.wave_to_embed.weight[:, sin_cos_dim:] *= 3.0  # 3x larger weights (was 10x)
        
        # Learnable scale for phase contribution (starts smaller for stability)
        self.phase_scale = nn.Parameter(torch.tensor(0.5))  # Start at 0.5, not 1.0
        
        # Positional wave modulation
        self.pos_freq = nn.Parameter(torch.randn(1, 1, num_waves) * 0.1)
        
        # === Requirement 1.5: Standard embedding for annealing ===
        self.simple_embed = nn.Embedding(vocab_size, d_model)
        
        # Use learned scaling instead of LayerNorm (LN kills gradients to phases/freqs)
        # RMSNorm-style: just scale by learned parameter, no mean subtraction
        self.output_scale = nn.Parameter(torch.ones(d_model))
        
    def forward(self, token_ids, standard_embed_ratio=0.0):
        """
        Compute wave packet embeddings with optional annealing.
        
        Args:
            token_ids: (B, T) tensor of token indices
            standard_embed_ratio: Mixing ratio for standard embeddings
                                  0.0 = pure wave, 1.0 = pure standard
                                  Requirement 1.5: out = (1-r)*wave + r*standard
        
        Returns:
            (B, T, d_model) wave packet embeddings
        """
        B, T = token_ids.shape
        device = token_ids.device
        
        # Get wave parameters for each token
        base_f = self.base_freqs[token_ids]    # (B, T, num_waves)
        phases = self.phases[token_ids]         # (B, T, num_waves)
        harm_a = self.harmonic_amps[token_ids]  # (B, T, num_waves, num_harmonics)
        
        # Positional modulation
        positions = torch.arange(T, device=device).float().view(1, T, 1)
        pos_phase = positions * self.pos_freq
        
        # Generate harmonics: ω_n = n * ω_0 (Requirement 1.3)
        # base_f: (B, T, W) -> expand to (B, T, W, H)
        freqs = base_f.unsqueeze(-1) * self.harmonic_mults  # (B, T, W, H)
        
        # Phase applies to all harmonics
        wave_phase = freqs * 2 * math.pi + phases.unsqueeze(-1) + pos_phase.unsqueeze(-1)
        
        # Weighted sum of sin/cos harmonics with 1/n amplitude decay (Requirement 1.4)
        sin_waves = harm_a * torch.sin(wave_phase)  # (B, T, W, H)
        cos_waves = harm_a * torch.cos(wave_phase)  # (B, T, W, H)
        
        # === DIRECT PHASE PATHWAY for stronger gradients ===
        # The sin/cos pathway dilutes phase gradients. Add direct phase info.
        # Normalize phases to [-1, 1] range and scale for stability
        phase_direct = self.phase_scale * ((phases / math.pi) - 1.0)  # (B, T, num_waves)
        
        # Also add frequency info directly (normalized to similar scale as phases)
        freq_direct = torch.log1p(base_f) / 5.0  # Normalize to ~[0, 1] range
        
        # Flatten: (B, T, W*H*2 + W + W)
        wave_state = torch.cat([
            sin_waves.reshape(B, T, -1),
            cos_waves.reshape(B, T, -1),
            phase_direct,  # Direct phase pathway!
            freq_direct    # Direct frequency pathway!
        ], dim=-1)
        
        # Project to embedding dimension
        wave_embed = self.wave_to_embed(wave_state)  # (B, T, d_model)
        
        # === Requirement 1.5: Embedding annealing ===
        # out = (1-r)*wave + r*standard
        if standard_embed_ratio > 0.0:
            simple_embed = self.simple_embed(token_ids)
            r = standard_embed_ratio
            embeddings = (1.0 - r) * wave_embed + r * simple_embed
        else:
            embeddings = wave_embed
        
        # Scale output (RMSNorm-style, preserves gradients unlike LayerNorm)
        rms = embeddings.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        embeddings = embeddings / rms * self.output_scale
        
        return embeddings
    
    def get_token_mass(self, token_id):
        """Get the mass for a specific token (Requirement 1.1)."""
        return self.masses[token_id]
    
    def get_base_frequency(self, token_id):
        """Get the base frequency for a specific token (Requirement 1.2)."""
        return self.base_freqs[token_id, 0]  # Same for all waves
    
    def get_harmonic_frequencies(self, token_id):
        """Get all harmonic frequencies for a token (Requirement 1.3)."""
        base_freq = self.base_freqs[token_id, 0]
        return base_freq * self.harmonic_mults
    
    def get_harmonic_amplitudes(self):
        """Get the harmonic amplitude decay pattern (Requirement 1.4)."""
        return 1.0 / self.harmonic_mults


# ==========================================
# Wave Interference Attention (Legacy - Hybrid)
# ==========================================

class WaveInterferenceAttention(nn.Module):
    """
    Attention via wave interference instead of dot product.
    
    Key insight: When waves meet, they interfere:
    - Constructive: waves in phase → amplify (high attention)
    - Destructive: waves out of phase → cancel (low attention)
    
    This is natural for waves and avoids the discrete dot product.
    """
    def __init__(self, d_model, num_heads, num_waves=16, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_waves = num_waves
        
        # Project to wave space for Q, K, V
        # Each becomes a wave state
        self.q_proj = nn.Linear(d_model, num_heads * num_waves)
        self.k_proj = nn.Linear(d_model, num_heads * num_waves)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Phase modulation for Q and K (learned)
        self.q_phase = nn.Parameter(torch.zeros(1, num_heads, 1, num_waves))
        self.k_phase = nn.Parameter(torch.zeros(1, num_heads, 1, num_waves))
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for attention sharpness
        self.temperature = nn.Parameter(torch.ones(1) * (num_waves ** 0.5))
        
    def forward(self, x):
        """Wave interference attention"""
        B, T, C = x.shape
        
        # Project to wave space
        q_waves = self.q_proj(x).view(B, T, self.num_heads, self.num_waves)  # (B, T, H, W)
        k_waves = self.k_proj(x).view(B, T, self.num_heads, self.num_waves)  # (B, T, H, W)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)         # (B, T, H, D)
        
        # Transpose for attention: (B, H, T, ...)
        q_waves = q_waves.transpose(1, 2)  # (B, H, T, W)
        k_waves = k_waves.transpose(1, 2)  # (B, H, T, W)
        v = v.transpose(1, 2)              # (B, H, T, D)
        
        # Add learnable phase shifts
        q_waves = q_waves + self.q_phase
        k_waves = k_waves + self.k_phase
        
        # Wave interference: compute via cosine similarity of wave states
        # When waves have same phase → cos(0) = 1 (constructive)
        # When waves are out of phase → cos(π) = -1 (destructive)
        
        # Normalize waves
        q_norm = F.normalize(q_waves, dim=-1)
        k_norm = F.normalize(k_waves, dim=-1)
        
        # Interference pattern: dot product of normalized waves
        # This gives cosine similarity = wave interference strength
        interference = torch.matmul(q_norm, k_norm.transpose(-2, -1)) / self.temperature  # (B, H, T, T)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        interference = interference.masked_fill(causal_mask, float('-inf'))
        
        # Softmax to get attention weights (interference probabilities)
        attn = F.softmax(interference, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values (wave superposition of values)
        out = torch.matmul(attn, v)  # (B, H, T, D)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        
        return out


# ==========================================
# Physics-Based Interference Attention (Memory-Efficient)
# ==========================================

class InterferenceAttention(nn.Module):
    """
    Memory-efficient physics-based attention via wave interference.
    
    Core physics: I = A_q² + A_k² + 2*A_q*A_k*cos(Δω*Δt + Δφ)
    
    MEMORY OPTIMIZATION (CRITICAL FOR TRAINING):
    The naive implementation creates (B, H, T, T, W) tensors = O(T²W) memory.
    With T=256, H=8, W=48, B=4: 4*8*256*256*48 = 100M floats = 400MB per layer!
    
    This version uses PHASOR DECOMPOSITION to achieve O(T²) memory:
    
    Key insight: sum_w[A_q*A_k*cos(θ_q - θ_k)] = Re(phasor_q · phasor_k*)
    where phasor = sum_w[A * e^(iθ)]
    
    By pre-computing phasor sums over W, we avoid the T×T×W explosion.
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_waves: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_waves = num_waves
        
        # === Requirement 2.1: Frequency/Phase/Amplitude projections ===
        self.freq_proj = nn.Linear(d_model, num_heads * num_waves)
        self.phase_proj = nn.Linear(d_model, num_heads * num_waves)
        self.amp_proj = nn.Linear(d_model, num_heads * num_waves)
        
        # Value projection
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8
        
        # Learnable temperature for attention sharpness
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wave interference attention - simplified for stable optimization.
        
        Instead of full interference formula (which has poor gradient properties),
        we use amplitude-weighted phase similarity:
        
        score[q,k] = sum_w(A_q * A_k * cos(θ_q - θ_k))
        
        This is like dot-product attention but in wave space:
        - Amplitude = magnitude (like vector length)
        - Phase = direction (like vector angle)
        - cos(Δθ) = alignment (like cosine similarity)
        """
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype
        
        # === Project to wave parameters (B, H, T, W) ===
        freq = self.freq_proj(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        phase_0 = self.phase_proj(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        amp = self.amp_proj(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        
        # Positive amplitudes (physical constraint)
        amp = F.softplus(amp) + self.eps  # (B, H, T, W)
        
        # Value projection
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        
        # === Phase evolution: θ(t) = ω*t + φ_0 ===
        t_pos = torch.arange(T, device=device, dtype=dtype).view(1, 1, T, 1)  # (1, 1, T, 1)
        theta = freq * t_pos + phase_0  # (B, H, T, W) - evolved phase at each position
        
        # === SIMPLIFIED INTERFERENCE: Amplitude-weighted phase similarity ===
        # score[q,k] = Re(phasor_q · phasor_k*) where phasor = A * e^(iθ)
        # This is analogous to Q @ K.T but in wave space
        
        # Compute phasors: A * e^(iθ)
        phasor = amp * torch.exp(1j * theta.to(torch.complex64))  # (B, H, T, W) complex
        
        # Attention scores = Re(phasor_q @ phasor_k^H) / sqrt(W)
        scores = torch.matmul(phasor, phasor.conj().transpose(-2, -1)).real  # (B, H, T, T)
        scores = scores / (self.num_waves ** 0.5)  # Scale like standard attention
        
        # Temperature scaling
        scores = scores * self.temperature
        
        # === Causal masking ===
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax for stable probability distribution
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        out = torch.matmul(attn_weights, v)  # (B, H, T, D)
        
        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        
        return out
    
    def get_interference_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw interference scores for visualization."""
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype
        
        freq = self.freq_proj(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        phase_0 = self.phase_proj(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        amp = self.amp_proj(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        amp = F.softplus(amp) + self.eps
        
        t_pos = torch.arange(T, device=device, dtype=dtype).view(1, 1, T, 1)
        theta = freq * t_pos + phase_0
        
        phasor = amp * torch.exp(1j * theta.to(torch.complex64))
        scores = torch.matmul(phasor, phasor.conj().transpose(-2, -1)).real
        scores = scores / (self.num_waves ** 0.5)
        
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return scores.masked_fill(causal_mask, 0.0)
    
    def get_interference_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized attention weights."""
        scores = self.get_interference_pattern(x)
        T = scores.size(-1)
        device = scores.device
        
        scores = scores * self.temperature
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        return F.softmax(scores, dim=-1)


# ==========================================
# PURE Wave Attention (NO SOFTMAX!)
# ==========================================

class PureWaveAttention(nn.Module):
    """
    TRUE wave interference attention - NO SOFTMAX, NO DOT PRODUCT.
    
    This is the pure wave paradigm:
    - Attention weights come directly from wave interference
    - Negative values = destructive interference = SUPPRESSION
    - No softmax normalization - pure field dynamics
    
    Key differences from standard attention:
    1. No softmax: interference can be negative (destructive)
    2. No Q/K dot product: uses phase-based interference
    3. Bounded naturally: cosine similarity is in [-1, 1]
    """
    def __init__(self, d_model, num_heads, num_waves=16, dropout=0.1, kernel='elu_plus_one', mode='quadratic'):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_waves = num_waves
        self.kernel = kernel
        self.mode = mode
        
        # Wave projections - map to frequency/phase space
        self.q_freq = nn.Linear(d_model, num_heads * num_waves)
        self.k_freq = nn.Linear(d_model, num_heads * num_waves)
        self.q_phase = nn.Linear(d_model, num_heads * num_waves)
        self.k_phase = nn.Linear(d_model, num_heads * num_waves)
        
        # Value projection (still needed to carry information)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Learnable interference strength per head
        self.interference_scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Pure wave interference attention.
        """
        B, T, C = x.shape
        
        # Project to wave parameters
        q_f = self.q_freq(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        k_f = self.k_freq(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        q_p = self.q_phase(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        k_p = self.k_phase(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute wave components
        t_pos = torch.arange(T, device=x.device).float().view(1, 1, T, 1)
        q_waves = torch.sin(q_f * t_pos + q_p)
        k_waves = torch.sin(k_f * t_pos + k_p)
        
        # Normalize
        q_norm = F.normalize(q_waves, dim=-1)
        k_norm = F.normalize(k_waves, dim=-1)
        
        # Scale q/k to have variance 1 before kernel
        # interference = q @ k * sqrt(W). Equivalent to q*W^0.25 @ k*W^0.25? 
        # For linear mode, we need to distribute the scaling.
        scale_factor = (self.num_waves ** 0.25)
        q_norm = q_norm * scale_factor
        k_norm = k_norm * scale_factor
        
        if self.mode == 'linear':
            # === Linear Attention: O(N) ===
            # Decompose kernel: phi(Q) @ phi(K)^T
            
            # 1. Apply kernel to Q and K directly
            if self.kernel == 'elu_plus_one':
                q_prime = F.elu(q_norm) + 1.0
                k_prime = F.elu(k_norm) + 1.0
            elif self.kernel == 'sigmoid':
                q_prime = torch.sigmoid(q_norm)
                k_prime = torch.sigmoid(k_norm)
            elif self.kernel == 'exp':
                q_prime = torch.exp(q_norm)
                k_prime = torch.exp(k_norm)
            else: # identity (not recommended for linear mode as cumsum will drift)
                q_prime = F.elu(q_norm) + 1.0
                k_prime = F.elu(k_norm) + 1.0
            
            # 2. Compute KV state = cumsum(K' * V)
            # Result: (B, H, T, W, D)
            kv = torch.einsum('bhtw, bhtd -> bhtwd', k_prime, v)
            S = torch.cumsum(kv, dim=2)
            
            # 3. Compute Z state (normalization denominator)
            # Result: (B, H, T, W)
            Z = torch.cumsum(k_prime, dim=2)
            
            # 4. Numerator = Q' . S (sum over W)
            numerator = torch.einsum('bhtw, bhtwd -> bhtd', q_prime, S)
            
            # 5. Denominator = Q' . Z (sum over W)
            denominator = torch.einsum('bhtw, bhtw -> bht', q_prime, Z)
            denominator = denominator.unsqueeze(-1)
            
            out = numerator / (denominator + 1e-6)
            
        else:
            # === Quadratic Attention: O(N^2) ===
            # Explicit interference matrix calculation
            
            # Interference = (q @ k) 
            # Note: q_norm, k_norm already scaled by W^0.25. Product is scaled by W^0.5.
            interference = torch.matmul(q_norm, k_norm.transpose(-2, -1))
            
            # Scale per head (learnable)
            interference = interference * self.interference_scale
            
            # Masking
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            
            # Activation
            if self.kernel == 'elu_plus_one':
                interference = interference.masked_fill(causal_mask, -float('inf'))
                attn = F.elu(interference) + 1.0
            elif self.kernel == 'sigmoid':
                interference = interference.masked_fill(causal_mask, -float('inf'))
                attn = torch.sigmoid(interference)
            elif self.kernel == 'exp':
                interference = interference.masked_fill(causal_mask, -float('inf'))
                attn = torch.exp(interference)
            else:
                interference = interference.masked_fill(causal_mask, 0.0)
                attn = interference
            
            # Normalize by number of attended positions (approx Z)
            num_attended = torch.arange(1, T + 1, device=x.device).float().view(1, 1, T, 1)
            attn = attn / num_attended.sqrt()
            
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        
        return out


# ==========================================
# Wave MLP (Resonance Network)
# ==========================================

class WaveResonanceMLP(nn.Module):
    """
    MLP that preserves wave nature using harmonic activations.
    
    Instead of ReLU (which clips), use periodic activations
    that maintain the oscillatory structure.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Hybrid activation: GELU for expressivity + sin for wave nature
        # Pure sin traps gradients; pure GELU loses wave character
        h = self.fc1(x)
        h = F.gelu(h) + 0.1 * torch.sin(h)  # GELU dominant, sin adds harmonics
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h


# ==========================================
# Wave Block
# ==========================================

class WaveBlock(nn.Module):
    """Single transformer block operating in wave space"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Choose attention type based on config
        if getattr(config, 'use_interference_attention', False):
            # Physics-based interference attention (Requirements 2.1-2.5)
            self.attn = InterferenceAttention(
                config.d_model, config.num_heads,
                config.num_waves, config.dropout
            )
        elif getattr(config, 'pure_wave_attention', False):
            # PURE wave attention - NO SOFTMAX!
            self.attn = PureWaveAttention(
                config.d_model, config.num_heads, 
                config.num_waves, config.dropout,
                kernel=getattr(config, 'pure_wave_kernel', 'elu_plus_one')
            )
        else:
            # Hybrid wave attention (with softmax)
            self.attn = WaveInterferenceAttention(
                config.d_model, config.num_heads, 
                config.num_waves, config.dropout
            )
        
        self.mlp = WaveResonanceMLP(
            config.d_model, config.d_model * 4, config.dropout
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ==========================================
# Collapse Head (Wave → Token)
# ==========================================

class CollapseHead(nn.Module):
    """
    Wave function collapse: continuous wave state → discrete token.
    
    Like quantum measurement: the continuous superposition
    "collapses" to a probability distribution over tokens.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.ln(x)
        logits = self.proj(x)  # Wave collapse → token probabilities
        return logits


# ==========================================
# Wave-Native GPT
# ==========================================

# Note: WaveGPTConfig is defined as a dataclass at the top of the file.
# This placeholder class is kept for backward compatibility with code that
# instantiates WaveGPTConfig with positional arguments.
class WaveGPTConfigCompat:
    """Compatibility class for WaveGPTConfig with positional arguments."""
    def __init__(self, vocab_size, d_model, num_heads, num_waves, dropout, num_harmonics, num_layers, block_size, model_type="wave", pure_wave_attention=False, pure_wave_kernel='elu_plus_one', pure_wave_mode='quadratic', use_interference_attention=False, use_wave_embeddings=True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_waves = num_waves
        self.dropout = dropout
        self.num_harmonics = num_harmonics
        self.num_layers = num_layers
        self.block_size = block_size
        self.model_type = model_type
        self.pure_wave_attention = pure_wave_attention
        self.pure_wave_kernel = pure_wave_kernel
        self.pure_wave_mode = pure_wave_mode
        self.use_interference_attention = use_interference_attention
        self.use_wave_embeddings = use_wave_embeddings

# Placeholder for StandardEmbedding and StandardBlock
# These classes were referenced in the diff but not provided.
# For the code to be syntactically correct, I'll add minimal placeholder definitions.
class StandardEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, dropout):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        token_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(pos)
        x = self.dropout(token_emb + pos_emb)
        return x

class StandardBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=self._generate_square_subsequent_mask(x.size(1)).to(x.device))[0]
        x = x + self.mlp(self.ln2(x))
        return x
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class WaveGPT(nn.Module):
    """
    Wave-Native GPT with independently toggleable physics components.
    
    Component Independence (Requirement 7.5):
    - use_wave_embeddings: Toggle WavePacketEmbedding vs StandardEmbedding
    - use_interference_attention: Toggle physics-based InterferenceAttention
    - pure_wave_attention: Toggle PureWaveAttention (no softmax)
    - model_type: "wave" uses WaveBlock, "standard" uses StandardBlock
    
    Each component can be toggled independently without breaking model functionality.
    """
    def __init__(self, config: WaveGPTConfig):
        super().__init__()
        self.config = config
        
        # Determine embedding type (Requirement 7.5: independent toggle)
        use_wave_embed = getattr(config, 'use_wave_embeddings', True)
        
        if config.model_type == "standard":
            # --- Standard Transformer ---
            # Standard model always uses standard embeddings
            self.embedding = StandardEmbedding(
                config.vocab_size, config.d_model, config.block_size, config.dropout
            )
            self.blocks = nn.ModuleList([
                StandardBlock(config) for _ in range(config.num_layers)
            ])
            self.ln_f = nn.LayerNorm(config.d_model) # Final LN
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            
        else:
            # --- Wave-Native Transformer ---
            # Embedding type can be toggled independently (Requirement 7.5)
            if use_wave_embed:
                self.embedding = WavePacketEmbedding(
                    config.vocab_size, config.d_model, 
                    config.num_waves, config.num_harmonics
                )
            else:
                # Use standard embedding with wave attention blocks
                self.embedding = StandardEmbedding(
                    config.vocab_size, config.d_model, config.block_size, config.dropout
                )
            
            self.blocks = nn.ModuleList([
                WaveBlock(config) for _ in range(config.num_layers)
            ])
            self.ln_f = nn.LayerNorm(config.d_model)
            self.head = CollapseHead(config.d_model, config.vocab_size)
        
        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, standard_embed_ratio=0.0):
        """
        Forward pass with optional embedding annealing.
        
        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices (optional)
            standard_embed_ratio: Mixing ratio for standard embeddings (0.0 = pure wave, 1.0 = pure standard)
                                  Used for annealing during training (Requirement 1.5)
        
        Returns:
            logits: (B, T, vocab_size) output logits
            loss: scalar loss if targets provided, else None
        """
        B, T = idx.shape
        
        # 1. Embedding (with optional annealing for wave embeddings)
        # Check if embedding is WavePacketEmbedding (supports standard_embed_ratio)
        if isinstance(self.embedding, WavePacketEmbedding):
            # WavePacketEmbedding supports standard_embed_ratio for annealing
            x = self.embedding(idx, standard_embed_ratio=standard_embed_ratio)
        else:
            # StandardEmbedding doesn't use standard_embed_ratio
            x = self.embedding(idx)
        
        # 2. Blocks
        for block in self.blocks:
            x = block(x)
            
        # 3. Final LN
        x = self.ln_f(x)
        
        # 4. Collapse/Head
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens one by one
        """
        for _ in range(max_new_tokens):
            # Crop to block size if needed
            if idx.size(1) > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size:]
            else:
                idx_cond = idx
                
            # Forward
            logits, _ = self(idx_cond)
            
            # Select last step
            logits = logits[:, -1, :] / temperature
            
            # Top-K sampling (optional)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx # Return full sequence

