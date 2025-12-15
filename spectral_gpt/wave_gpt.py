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
    model_type: str = "wave"               # "wave", "standard", or "pure_wave"
    use_interference_attention: bool = False  # True = physics-based interference attention (Req 2.1-2.5)
    use_wave_embeddings: bool = True       # True = WavePacketEmbedding, False = StandardEmbedding (Req 7.5)
    pure_wave_mode_v2: bool = False        # True = PURE WAVE-TO-WAVE (no embeddings anywhere!)

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


class PureWaveBlock(nn.Module):
    """
    Pure Wave-Native Transformer Block.
    
    NO EMBEDDINGS! Everything operates on wave parameters:
    - Input: (frequencies, phases, amplitudes)
    - Attention: Wave interference between wave parameters
    - MLP: Wave parameter transformations
    - Output: (frequencies, phases, amplitudes)
    
    This is truly wave-native computation!
    """
    def __init__(self, config):
        super().__init__()
        
        # Wave-native attention
        self.wave_attn = InterferenceAttention(
            num_waves=config.num_waves,
            num_harmonics=config.num_harmonics,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Wave-native MLP (operates on wave parameters)
        self.wave_mlp = WaveNativeMLP(
            num_waves=config.num_waves,
            num_harmonics=config.num_harmonics,
            dropout=config.dropout
        )
        
        # Wave normalization (instead of LayerNorm)
        self.wave_norm1 = WaveNormalization(config.num_waves, config.num_harmonics)
        self.wave_norm2 = WaveNormalization(config.num_waves, config.num_harmonics)
        
    def forward(self, wave_freqs, wave_phases, wave_amps):
        """
        Pure wave computation - no embeddings anywhere!
        
        Args:
            wave_freqs: (B, T, num_waves)
            wave_phases: (B, T, num_waves)  
            wave_amps: (B, T, num_waves, num_harmonics)
            
        Returns:
            Same shapes but contextualized through wave physics
        """
        # Normalize waves
        norm_freqs, norm_phases, norm_amps = self.wave_norm1(wave_freqs, wave_phases, wave_amps)
        
        # Wave interference attention
        attn_freqs, attn_phases, attn_amps = self.wave_attn(norm_freqs, norm_phases, norm_amps)
        
        # Residual connection in wave space
        res_freqs = wave_freqs + attn_freqs
        res_phases = wave_phases + attn_phases  
        res_amps = wave_amps + attn_amps
        
        # Normalize again
        norm_freqs2, norm_phases2, norm_amps2 = self.wave_norm2(res_freqs, res_phases, res_amps)
        
        # Wave MLP
        mlp_freqs, mlp_phases, mlp_amps = self.wave_mlp(norm_freqs2, norm_phases2, norm_amps2)
        
        # Final residual connection
        output_freqs = res_freqs + mlp_freqs
        output_phases = res_phases + mlp_phases
        output_amps = res_amps + mlp_amps
        
        return output_freqs, output_phases, output_amps


class WaveNativeMLP(nn.Module):
    """MLP that operates directly on wave parameters"""
    def __init__(self, num_waves, num_harmonics, dropout=0.1):
        super().__init__()
        
        # Separate MLPs for each wave parameter type
        self.freq_mlp = nn.Sequential(
            nn.Linear(num_waves, 4 * num_waves),
            nn.GELU(),
            nn.Linear(4 * num_waves, num_waves),
            nn.Dropout(dropout)
        )
        
        self.phase_mlp = nn.Sequential(
            nn.Linear(num_waves, 4 * num_waves),
            nn.GELU(), 
            nn.Linear(4 * num_waves, num_waves),
            nn.Dropout(dropout)
        )
        
        amp_dim = num_waves * num_harmonics
        self.amp_mlp = nn.Sequential(
            nn.Linear(amp_dim, 4 * amp_dim),
            nn.GELU(),
            nn.Linear(4 * amp_dim, amp_dim),
            nn.Dropout(dropout)
        )
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
    def forward(self, wave_freqs, wave_phases, wave_amps):
        B, T, W = wave_freqs.shape
        H = wave_amps.shape[-1]
        
        # Transform each wave parameter type
        new_freqs = self.freq_mlp(wave_freqs)
        new_phases = self.phase_mlp(wave_phases)
        
        # Flatten amplitudes for MLP
        amps_flat = wave_amps.view(B, T, -1)
        new_amps_flat = self.amp_mlp(amps_flat)
        new_amps = new_amps_flat.view(B, T, W, H)
        
        return new_freqs, new_phases, new_amps


class WaveNormalization(nn.Module):
    """Normalization for wave parameters (instead of LayerNorm)"""
    def __init__(self, num_waves, num_harmonics):
        super().__init__()
        
        # Learnable scales for each parameter type
        self.freq_scale = nn.Parameter(torch.ones(num_waves))
        self.phase_scale = nn.Parameter(torch.ones(num_waves))
        self.amp_scale = nn.Parameter(torch.ones(num_waves, num_harmonics))
        
        self.eps = 1e-8
        
    def forward(self, wave_freqs, wave_phases, wave_amps):
        # RMS normalization for each parameter type
        freq_rms = wave_freqs.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        norm_freqs = wave_freqs / freq_rms * self.freq_scale
        
        phase_rms = wave_phases.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        norm_phases = wave_phases / phase_rms * self.phase_scale
        
        amp_rms = wave_amps.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=self.eps)
        norm_amps = wave_amps / amp_rms * self.amp_scale
        
        return norm_freqs, norm_phases, norm_amps



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
        # Create a "sea of low frequencies with spikes for local attention"
        # 
        # Strategy: Most waves are LOW frequency (global context), but some are HIGH (local spikes)
        # Distribution: 70% low freq, 20% mid freq, 10% high freq
        # This creates the desired "sea with spikes" pattern
        
        # Define frequency bands for multi-scale attention:
        # - Low freq (0.1-1.0): peaks every 6-60 tokens → global/document context  
        # - Mid freq (1.0-5.0): peaks every 1-6 tokens → paragraph context
        # - High freq (5.0-20.0): peaks every 0.3-1 tokens → local/word spikes
        
        n_low = int(0.7 * num_waves)    # 70% low frequencies (the "sea")
        n_mid = int(0.2 * num_waves)    # 20% mid frequencies  
        n_high = num_waves - n_low - n_mid  # 10% high frequencies (the "spikes")
        
        # Generate frequency bands
        low_freqs = torch.linspace(0.1, 1.0, n_low)      # Sea of low frequencies
        mid_freqs = torch.linspace(1.0, 5.0, n_mid)      # Medium frequencies
        high_freqs = torch.linspace(5.0, 20.0, n_high)   # Local attention spikes
        
        # Combine into full spectrum (sea + spikes)
        base_spectrum = torch.cat([low_freqs, mid_freqs, high_freqs])
        
        # Shuffle to mix low/mid/high across wave components (breaks patterns)
        perm = torch.randperm(num_waves)
        base_spectrum = base_spectrum[perm]
        
        # Add small token-dependent variation (each token gets slightly different spectrum)
        # Common tokens (low mass) get slightly lower frequencies → more global attention
        # Rare tokens (high mass) get slightly higher frequencies → more local attention
        raw_center_freq = 1.0 / torch.sqrt(masses)  # (vocab_size,)
        token_variation = (raw_center_freq - raw_center_freq.mean()) / raw_center_freq.std()
        token_variation = token_variation * 0.1  # Small variation ±10%
        
        # Each token gets the base spectrum with small individual variation
        # Shape: (vocab_size, num_waves)
        init_base_frequencies = base_spectrum.unsqueeze(0) * (1.0 + token_variation.unsqueeze(1))
        self.base_freqs = nn.Parameter(init_base_frequencies.clone())  # LEARNABLE!
        
        # === Requirement 1.3: Harmonic quantization ===
        # Harmonic multipliers - keep as buffer (structural, not learned)
        harmonic_mults = torch.arange(1, num_harmonics + 1, dtype=torch.float32)  # [1, 2, 3, 4, ...]
        self.register_buffer('harmonic_mults', harmonic_mults)
        
        # === Requirement 1.4: Power law amplitude decay ===
        # LEARNABLE amplitudes initialized with physics prior: A_n = 1/n
        # But add some randomness to create more interesting wave shapes
        base_amplitudes = 1.0 / harmonic_mults  # [1, 0.5, 0.333, 0.25, ...]
        
        # Add small random variations to break uniformity and create unique wave shapes
        # Each token-wave combination gets slightly different harmonic content
        random_variations = torch.randn(vocab_size, num_waves, num_harmonics) * 0.2
        init_harmonic_amplitudes = base_amplitudes.view(1, 1, -1) * (1.0 + random_variations)
        
        # Ensure amplitudes stay positive and reasonable
        init_harmonic_amplitudes = torch.clamp(init_harmonic_amplitudes, min=0.1, max=2.0)
        
        self.harmonic_amps = nn.Parameter(init_harmonic_amplitudes)  # NOW LEARNABLE!
        
        # Debug: Print frequency distribution
        print(f"🌊 Frequency spectrum initialized:")
        print(f"   Low freq (sea): {low_freqs.min():.3f} - {low_freqs.max():.3f} Hz ({n_low} waves)")
        print(f"   Mid freq: {mid_freqs.min():.3f} - {mid_freqs.max():.3f} Hz ({n_mid} waves)")  
        print(f"   High freq (spikes): {high_freqs.min():.3f} - {high_freqs.max():.3f} Hz ({n_high} waves)")
        print(f"   Total range: {base_spectrum.min():.3f} - {base_spectrum.max():.3f} Hz")
        
        # === Phases: Random initialization with structure ===
        # Golden angle creates too rigid a pattern. Use random init with some structure:
        # - Random base phase per token (allows learning unique token "signatures")
        # - Small wave-dependent offset (breaks symmetry between waves)
        
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
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
        
        # === PHASE 2: TEMPORAL EVOLUTION ===
        # Position in sequence = Time t
        # Phase evolves according to wave equation: φ(t) = ω*t + φ_0
        # This is pure physics - no hardcoded envelopes!
        wave_phase = freqs * positions.unsqueeze(-1) + phases.unsqueeze(-1) + pos_phase.unsqueeze(-1)
        
        # === WAVE PACKET GENERATION ===
        # The token is "struck" and generates oscillations at its natural frequencies
        # No artificial envelopes - the wave shape emerges from harmonic superposition
        sin_waves = harm_a * torch.sin(wave_phase)  # (B, T, W, H)
        cos_waves = harm_a * envelope * torch.cos(wave_phase)  # (B, T, W, H)
        
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
        
        # === PURE WAVE OUTPUT ===
        # Instead of projecting to embedding space, return raw wave parameters!
        # This enables pure wave-to-wave computation throughout the network
        
        if standard_embed_ratio > 0.0:
            # For annealing, we still need embeddings temporarily
            simple_embed = self.simple_embed(token_ids)
            r = standard_embed_ratio
            embeddings = (1.0 - r) * wave_embed + r * simple_embed
            
            # Scale output (RMSNorm-style, preserves gradients unlike LayerNorm)
            rms = embeddings.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
            embeddings = embeddings / rms * self.output_scale
            
            return embeddings
        else:
            # Pure wave mode: return wave parameters directly!
            return base_f, phases, harm_a  # (freqs, phases, amplitudes)
    
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
    Pure Wave-to-Wave Interference Attention.
    
    NO EMBEDDING SPACE! Direct wave parameter transformations.
    
    Input: Wave parameters (frequencies, phases, amplitudes)
    Output: Transformed wave parameters after interference
    
    This is truly wave-native - every operation has clear physical meaning:
    - Wave parameter transformations (learned resonance coupling)
    - Direct wave interference computation
    - Wave superposition for output
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    def __init__(
        self,
        num_waves: int = 16,
        num_harmonics: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        self.num_heads = num_heads
        
        # === DIRECT WAVE-TO-WAVE TRANSFORMATIONS ===
        # Transform input wave parameters to attention wave parameters
        self.freq_transform = nn.Linear(num_waves, num_heads * num_waves)
        self.phase_transform = nn.Linear(num_waves, num_heads * num_waves)
        self.amp_transform = nn.Linear(num_waves * num_harmonics, num_heads * num_waves)
        
        # Wave-to-wave value transformation (no embedding!)
        self.value_freq_transform = nn.Linear(num_waves, num_waves)
        self.value_phase_transform = nn.Linear(num_waves, num_waves)
        self.value_amp_transform = nn.Linear(num_waves * num_harmonics, num_waves * num_harmonics)
        
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8
        
        # Learnable wave physics parameters
        self.temperature = nn.Parameter(torch.ones(1))
        self.interference_strength = nn.Parameter(torch.tensor(1.0))
        
        # Initialize transformations to preserve multi-scale structure
        with torch.no_grad():
            # Initialize frequency transform to spread across scales
            freq_init = torch.linspace(0.1, 5.0, num_waves).repeat(num_heads)
            self.freq_transform.bias.data = freq_init
        
    def forward(self, wave_freqs: torch.Tensor, wave_phases: torch.Tensor, wave_amps: torch.Tensor) -> tuple:
        """
        Pure Wave-to-Wave Interference Attention.
        
        Input: Raw wave parameters from embedding layer
        - wave_freqs: (B, T, num_waves) - base frequencies per token
        - wave_phases: (B, T, num_waves) - phases per token  
        - wave_amps: (B, T, num_waves, num_harmonics) - harmonic amplitudes
        
        Output: Transformed wave parameters after attention
        - Same shapes as input but contextualized through wave interference
        
        NO EMBEDDING SPACE - pure wave physics!
        """
        B, T, W = wave_freqs.shape
        H = wave_amps.shape[-1]  # num_harmonics
        device = wave_freqs.device
        dtype = wave_freqs.dtype
        
        # === WAVE-TO-WAVE TRANSFORMATIONS ===
        # Transform input waves to attention space waves
        attn_freqs = F.softplus(self.freq_transform(wave_freqs)) + self.eps  # (B, T, H*W)
        attn_freqs = attn_freqs.view(B, T, self.num_heads, self.num_waves).transpose(1, 2)  # (B, H, T, W)
        
        attn_phases = self.phase_transform(wave_phases).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        
        # Flatten amplitudes for transformation
        wave_amps_flat = wave_amps.view(B, T, -1)  # (B, T, W*H)
        attn_amps = F.softplus(self.amp_transform(wave_amps_flat)) + self.eps
        attn_amps = attn_amps.view(B, T, self.num_heads, self.num_waves).transpose(1, 2)  # (B, H, T, W)
        
        # === VALUE WAVE TRANSFORMATIONS ===
        # Transform input waves for values (wave-to-wave, no embedding!)
        value_freqs = F.softplus(self.value_freq_transform(wave_freqs)) + self.eps  # (B, T, W)
        value_phases = self.value_phase_transform(wave_phases)  # (B, T, W)
        value_amps = F.softplus(self.value_amp_transform(wave_amps_flat)) + self.eps
        value_amps = value_amps.view(B, T, W, H)  # (B, T, W, H)
        
        # === WAVE INTERFERENCE (Pure Physics!) ===
        # Position indices for phase evolution
        positions = torch.arange(T, device=device, dtype=dtype)
        
        # Phase at each position: θ(t) = ω*t + φ_0
        # Different frequencies create different phase evolution rates!
        theta = freq * positions.view(1, 1, T, 1) + phase  # (B, H, T, W)
        
        # Create complex phasors: A * e^(iθ)
        # This encodes both amplitude and phase in one complex number
        phasor = amp * torch.exp(1j * theta.to(torch.complex64))  # (B, H, T, W)
        
        # === INTERFERENCE PATTERN ===
        # When query phasor meets key phasor, they interfere:
        # score[q,k] = Re(sum_w(phasor_q[w] * conj(phasor_k[w])))
        #            = sum_w(A_q * A_k * cos(θ_q - θ_k))
        #
        # This NATURALLY creates:
        # - High scores when phases align (constructive interference)
        # - Low scores when phases oppose (destructive interference)
        # - BEATING patterns when frequencies differ!
        #   Beat period = 2π / |ω_q - ω_k| → learned from data!
        
        # === FULL INTERFERENCE EQUATION ===
        # I = A_Q² + A_K² + 2*A_Q*A_K*cos(Δφ)
        # 
        # The phasor dot product gives us: sum_w(A_q * A_k * cos(θ_q - θ_k))
        # This is the interference term: 2*A_Q*A_K*cos(Δφ) (summed over waves)
        
        interference_term = torch.matmul(phasor, phasor.conj().transpose(-2, -1)).real  # (B, H, T, T)
        
        # Compute amplitude squared terms for full intensity
        amp_sq = (amp ** 2).sum(dim=-1)  # (B, H, T) - sum over waves
        amp_q_sq = amp_sq.unsqueeze(-1)  # (B, H, T, 1)
        amp_k_sq = amp_sq.unsqueeze(-2)  # (B, H, 1, T)
        
        # Full interference intensity: I = A_Q² + A_K² + 2*A_Q*A_K*cos(Δφ)
        intensity = amp_q_sq + amp_k_sq + 2 * interference_term  # (B, H, T, T)
        
        # === PHYSICS-BASED NORMALIZATION ===
        # Normalize by maximum potential energy: (A_Q + A_K)²
        # This gives transmission coefficient in [0, 1] based on resonance
        amp_sum = amp.sum(dim=-1)  # (B, H, T) - total amplitude per position
        amp_q = amp_sum.unsqueeze(-1)  # (B, H, T, 1)
        amp_k = amp_sum.unsqueeze(-2)  # (B, H, 1, T)
        max_energy = (amp_q + amp_k) ** 2 + self.eps  # (B, H, T, T)
        
        # Transmission coefficient: how much energy passes through
        transmission = intensity / max_energy  # (B, H, T, T), range ~ [0, 1]
        
        # Scale by learned parameters
        scores = self.interference_strength * transmission * self.temperature
        
        # === Causal masking ===
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, 0.0)  # Zero out future (not -inf!)
        
        # === NO SOFTMAX! ===
        # Multiple contexts can be fully active simultaneously
        # Just normalize to ensure stability
        row_sum = scores.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        attn_weights = scores / row_sum  # Normalize by row sum
        attn_weights = self.dropout(attn_weights)
        
        # Store for visualization
        self.last_attention_weights = attn_weights.detach()
        self.last_interference = interference.detach()
        self.last_frequencies = freq.detach()
        self.last_phases = phase.detach()
        self.last_amplitudes = amp.detach()
        
        # === WAVE SUPERPOSITION (Instead of matrix multiply with values) ===
        # Apply attention weights to wave parameters directly!
        
        # Expand value waves for multi-head attention
        value_freqs_expanded = value_freqs.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, T, W)
        value_phases_expanded = value_phases.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, T, W)
        value_amps_expanded = value_amps.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)  # (B, H, T, W, H)
        
        # Wave superposition: weighted combination of wave parameters
        output_freqs = torch.matmul(attn_weights, value_freqs_expanded)  # (B, H, T, W)
        output_phases = torch.matmul(attn_weights, value_phases_expanded)  # (B, H, T, W)
        
        # For amplitudes, we need to handle the extra harmonic dimension
        value_amps_flat = value_amps_expanded.view(B, self.num_heads, T, -1)  # (B, H, T, W*H)
        output_amps_flat = torch.matmul(attn_weights, value_amps_flat)  # (B, H, T, W*H)
        output_amps = output_amps_flat.view(B, self.num_heads, T, W, H)  # (B, H, T, W, H)
        
        # === MULTI-HEAD WAVE FUSION ===
        # Combine multiple attention heads back to single wave representation
        final_freqs = output_freqs.mean(dim=1)  # (B, T, W) - average across heads
        final_phases = output_phases.mean(dim=1)  # (B, T, W)
        final_amps = output_amps.mean(dim=1)  # (B, T, W, H)
        
        return final_freqs, final_phases, final_amps
    
    def get_interference_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw wave interference scores for visualization."""
        with torch.no_grad():
            _ = self.forward(x)
        
        if hasattr(self, 'last_interference'):
            return self.last_interference
        else:
            return torch.zeros(x.size(0), self.num_heads, x.size(1), x.size(1), device=x.device)
    
    def get_interference_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized attention weights (after softmax)."""
        with torch.no_grad():
            _ = self.forward(x)
        
        if hasattr(self, 'last_attention_weights'):
            return self.last_attention_weights
        else:
            T = x.size(1)
            return torch.zeros(x.size(0), self.num_heads, T, T, device=x.device)
    
    def get_wave_components(self, x: torch.Tensor) -> dict:
        """Get learned wave components for visualization."""
        with torch.no_grad():
            _ = self.forward(x)
        
        return {
            'frequencies': self.last_frequencies if hasattr(self, 'last_frequencies') else None,
            'phases': self.last_phases if hasattr(self, 'last_phases') else None,
            'amplitudes': self.last_amplitudes if hasattr(self, 'last_amplitudes') else None,
            'interference': self.last_interference if hasattr(self, 'last_interference') else None,
            'attention_weights': self.last_attention_weights if hasattr(self, 'last_attention_weights') else None,
        }
    
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


# ==========================================
# PURE WAVE GPT - Wave-to-Wave Throughout!
# ==========================================

class PureWaveGPT(nn.Module):
    """
    Pure Wave-Native GPT: Wave-to-Wave computation throughout.
    
    NO EMBEDDING SPACE! Everything is waves:
    - Token → Wave excitation (frequencies, phases, amplitudes)
    - Wave → Wave attention (interference physics)
    - Wave → Wave MLP (resonance filtering)
    - Wave → Logits (measurement/collapse)
    
    All patterns emerge from learned wave physics:
    - Decay and re-emergence from frequency beating
    - Attention patterns from phase interference
    - Context from wave superposition
    """
    
    def __init__(self, config: WaveGPTConfig):
        super().__init__()
        self.config = config
        
        # === WAVE EXCITATION (Token → Wave) ===
        self.wave_excitation = PureWaveExcitation(
            vocab_size=config.vocab_size,
            num_waves=config.num_waves,
            num_harmonics=config.num_harmonics,
            block_size=config.block_size
        )
        
        # === WAVE TRANSFORMER LAYERS ===
        self.wave_layers = nn.ModuleList([
            PureWaveLayer(
                num_waves=config.num_waves,
                num_harmonics=config.num_harmonics,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # === WAVE COLLAPSE (Wave → Logits) ===
        self.wave_collapse = WaveCollapse(
            vocab_size=config.vocab_size,
            num_waves=config.num_waves,
            num_harmonics=config.num_harmonics
        )
        
    def forward(self, token_ids, targets=None):
        """
        Enhanced pure wave forward pass with wave regularization.
        
        Token → Wave → Wave → ... → Wave → Logits
        """
        B, T = token_ids.shape
        
        # 1. EXCITATION: Token → Wave parameters
        wave_state = self.wave_excitation(token_ids)
        # wave_state = (freqs, phases, amps) - pure wave representation!
        
        # Store initial wave state for regularization
        initial_wave_state = WaveState(
            wave_state.freqs.clone(),
            wave_state.phases.clone(), 
            wave_state.amps.clone()
        )
        
        # 2. WAVE LAYERS: Wave → Wave transformations
        for layer in self.wave_layers:
            wave_state = layer(wave_state)
        
        # 3. COLLAPSE: Wave → Logits (measurement)
        logits = self.wave_collapse(wave_state)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Primary loss: cross-entropy
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # === WAVE PHYSICS REGULARIZATION ===
            wave_reg_loss = 0.0
            
            # 1. Frequency stability: prevent frequencies from becoming too extreme
            freq_stability = torch.mean((wave_state.freqs - initial_wave_state.freqs) ** 2) * 0.01
            
            # 2. Phase coherence: encourage smooth phase evolution
            if T > 1:
                phase_diff = wave_state.phases[:, 1:] - wave_state.phases[:, :-1]
                phase_coherence = torch.mean(torch.sin(phase_diff) ** 2) * 0.005  # Penalize abrupt phase jumps
            else:
                phase_coherence = 0.0
            
            # 3. Amplitude conservation: prevent amplitude explosion/collapse
            amp_conservation = torch.mean((wave_state.amps.mean() - initial_wave_state.amps.mean()) ** 2) * 0.01
            
            # 4. Wave coupling regularization: encourage meaningful wave interactions
            if hasattr(self.wave_excitation, 'wave_coupling'):
                coupling_reg = torch.mean(self.wave_excitation.wave_coupling ** 2) * 0.001
            else:
                coupling_reg = 0.0
            
            wave_reg_loss = freq_stability + phase_coherence + amp_conservation + coupling_reg
            
            # Total loss
            loss = ce_loss + wave_reg_loss
            
            # Store loss components for monitoring
            if hasattr(self, 'loss_components'):
                self.loss_components = {
                    'ce_loss': ce_loss.item(),
                    'freq_stability': freq_stability.item() if isinstance(freq_stability, torch.Tensor) else freq_stability,
                    'phase_coherence': phase_coherence.item() if isinstance(phase_coherence, torch.Tensor) else phase_coherence,
                    'amp_conservation': amp_conservation.item() if isinstance(amp_conservation, torch.Tensor) else amp_conservation,
                    'coupling_reg': coupling_reg.item() if isinstance(coupling_reg, torch.Tensor) else coupling_reg,
                    'total_wave_reg': wave_reg_loss.item() if isinstance(wave_reg_loss, torch.Tensor) else wave_reg_loss
                }
        
        return logits, loss
    
    def get_wave_state(self, token_ids, layer_idx=None):
        """Get wave state at a specific layer for visualization."""
        wave_state = self.wave_excitation(token_ids)
        
        if layer_idx is None:
            return wave_state
        
        for i, layer in enumerate(self.wave_layers):
            wave_state = layer(wave_state)
            if i == layer_idx:
                return wave_state
        
        return wave_state
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens one by one for PureWaveGPT
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


class PureWaveExcitation(nn.Module):
    """
    Token → Wave Excitation.
    
    EVERY TOKEN gets the FULL wave spectrum - ALL frequencies, ALL harmonics!
    
    Key principle: Token 0 and Token 50256 both have:
    - Low frequencies (sentence-level patterns)
    - Mid frequencies (word-level patterns)  
    - High frequencies (phoneme-level patterns)
    - All harmonics (1f, 2f, 3f, 4f...)
    
    The ONLY difference between tokens is their LEARNED wave parameters.
    Position encoding comes from temporal phase evolution φ(t) = ω*t + φ₀
    
    Everything is LEARNABLE:
    - Frequencies per token per wave
    - Phases per token per wave
    - Amplitudes per token per wave per harmonic
    """
    
    def __init__(self, vocab_size, num_waves, num_harmonics, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        self.block_size = block_size
        
        # === EVERY TOKEN GETS FULL MULTI-SCALE SPECTRUM ===
        # Initialize with physics-based prior, but FULLY LEARNABLE
        
        # Create the base spectrum covering ALL linguistic scales:
        # - Sentence scale: 0.005-0.02 Hz (periods 50-200 tokens)
        # - Phrase scale: 0.02-0.05 Hz (periods 20-50 tokens)
        # - Word scale: 0.05-0.2 Hz (periods 5-20 tokens)
        # - Morpheme scale: 0.2-0.5 Hz (periods 2-5 tokens)
        # - Phoneme scale: 0.5-2.0 Hz (periods 0.5-2 tokens)
        
        # Log-spaced frequencies cover all scales naturally
        base_spectrum = torch.logspace(math.log10(0.005), math.log10(2.0), num_waves)
        
        # EVERY token gets this FULL spectrum as initialization
        # Shape: (vocab_size, num_waves) - each token has ALL frequencies
        init_freqs = base_spectrum.unsqueeze(0).expand(vocab_size, -1).clone()
        
        # Add small per-token variation for diversity (but all tokens still have all scales!)
        token_indices = torch.arange(vocab_size, dtype=torch.float32)
        token_variation = torch.randn(vocab_size, num_waves) * 0.1
        init_freqs = init_freqs * (1.0 + token_variation)
        
        # Ensure positive frequencies
        init_freqs = torch.clamp(init_freqs, min=0.001)
        
        self.base_freqs = nn.Parameter(init_freqs)  # (vocab_size, num_waves) - FULLY LEARNABLE!
        
        print(f"🌊 FULL SPECTRUM for EVERY token:")
        print(f"   Each token has {num_waves} waves spanning {base_spectrum.min():.4f} - {base_spectrum.max():.3f} Hz")
        print(f"   Sentence-level: ~{base_spectrum[0]:.4f} Hz (period ~{1/base_spectrum[0]:.0f} tokens)")
        print(f"   Phoneme-level: ~{base_spectrum[-1]:.3f} Hz (period ~{1/base_spectrum[-1]:.1f} tokens)")
        print(f"   Total learnable frequencies: {vocab_size} × {num_waves} = {vocab_size * num_waves:,}")
        
        # === HARMONIC MULTIPLIERS ===
        # Each wave has harmonics: f, 2f, 3f, 4f...
        harmonic_mults = torch.arange(1, num_harmonics + 1, dtype=torch.float32)
        self.register_buffer('harmonic_mults', harmonic_mults)
        
        # === PHASES - FULLY LEARNABLE per token per wave ===
        # Random initialization - the model learns what phases work best
        init_phases = torch.rand(vocab_size, num_waves) * 2 * math.pi
        self.phases = nn.Parameter(init_phases)  # (vocab_size, num_waves) - FULLY LEARNABLE!
        
        print(f"   Total learnable phases: {vocab_size} × {num_waves} = {vocab_size * num_waves:,}")
        
        # === AMPLITUDES - FULLY LEARNABLE per token per wave per harmonic ===
        # Initialize with 1/n decay prior (physics-based) but fully learnable
        base_amps = 1.0 / harmonic_mults  # [1, 0.5, 0.33, 0.25]
        
        # EVERY token gets ALL harmonics for ALL waves
        init_amps = base_amps.view(1, 1, -1).expand(vocab_size, num_waves, -1).clone()
        
        # Add per-token variation
        amp_variation = torch.randn(vocab_size, num_waves, num_harmonics) * 0.2
        init_amps = init_amps * (1.0 + amp_variation)
        init_amps = torch.clamp(init_amps, min=0.01, max=2.0)
        
        self.amplitudes = nn.Parameter(init_amps)  # (vocab_size, num_waves, num_harmonics) - FULLY LEARNABLE!
        
        print(f"   Total learnable amplitudes: {vocab_size} × {num_waves} × {num_harmonics} = {vocab_size * num_waves * num_harmonics:,}")
        
        # === CROSS-WAVE COUPLING ===
        # Allows waves to influence each other (captures frequency interactions)
        self.wave_coupling = nn.Parameter(
            torch.eye(num_waves) + torch.randn(num_waves, num_waves) * 0.05
        )
        
        # === HARMONIC COUPLING ===
        # Allows harmonics to influence each other (captures harmonic relationships)
        self.harmonic_coupling = nn.Parameter(
            torch.eye(num_harmonics) + torch.randn(num_harmonics, num_harmonics) * 0.05
        )
        
        total_params = (vocab_size * num_waves) + (vocab_size * num_waves) + (vocab_size * num_waves * num_harmonics)
        print(f"🎯 Total wave excitation parameters: {total_params:,} (all learnable!)")
        
    def forward(self, token_ids):
        """
        Wave excitation: Token → Full Wave Spectrum.
        
        EVERY token gets ALL frequencies, ALL harmonics.
        Position encoding comes from temporal phase evolution.
        
        Returns: WaveState tuple (freqs, phases, amps)
        """
        B, T = token_ids.shape
        device = token_ids.device
        
        # === LOOKUP WAVE PARAMETERS ===
        # Each token has its own FULL spectrum (all frequencies, all harmonics)
        token_freqs = self.base_freqs[token_ids]      # (B, T, num_waves)
        token_phases = self.phases[token_ids]          # (B, T, num_waves)
        token_amps = self.amplitudes[token_ids]        # (B, T, num_waves, num_harmonics)
        
        # === CROSS-WAVE COUPLING ===
        # Waves can influence each other (learned interactions)
        coupled_freqs = torch.matmul(token_freqs, self.wave_coupling)  # (B, T, num_waves)
        
        # === HARMONIC COUPLING ===
        # Harmonics can influence each other (learned harmonic relationships)
        # Reshape for matmul: (B, T, num_waves, num_harmonics) @ (num_harmonics, num_harmonics)
        coupled_amps = torch.matmul(token_amps, self.harmonic_coupling)  # (B, T, num_waves, num_harmonics)
        
        # === TEMPORAL PHASE EVOLUTION ===
        # This is the ONLY position encoding: φ(t) = ω*t + φ₀
        # Position in sequence = Time
        positions = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1)
        
        # Phase evolves according to wave equation
        evolved_phases = coupled_freqs * positions + token_phases  # (B, T, num_waves)
        
        # Ensure frequencies stay positive
        final_freqs = F.softplus(coupled_freqs) + 1e-6
        
        # Ensure amplitudes stay positive
        final_amps = F.softplus(coupled_amps)
        
        return WaveState(final_freqs, evolved_phases, final_amps)


class WaveState:
    """Container for wave parameters - the fundamental representation."""
    def __init__(self, freqs, phases, amps):
        self.freqs = freqs    # (B, T, num_waves)
        self.phases = phases  # (B, T, num_waves)
        self.amps = amps      # (B, T, num_waves, num_harmonics)
    
    def to_phasors(self):
        """Convert to complex phasors for interference computation."""
        # Sum amplitudes across harmonics for total amplitude per wave
        total_amp = self.amps.sum(dim=-1)  # (B, T, num_waves)
        phasors = total_amp * torch.exp(1j * self.phases.to(torch.complex64))
        return phasors  # (B, T, num_waves) complex


class PureWaveLayer(nn.Module):
    """
    Pure Wave Transformer Layer.
    
    Wave → Wave computation:
    1. Wave Interference Attention
    2. Wave Residual Connection
    3. Wave MLP (Resonance Filtering)
    4. Wave Residual Connection
    """
    
    def __init__(self, num_waves, num_harmonics, num_heads, dropout=0.1):
        super().__init__()
        
        self.wave_attention = PureWaveInterference(
            num_waves=num_waves,
            num_harmonics=num_harmonics,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.wave_mlp = PureWaveMLP(
            num_waves=num_waves,
            num_harmonics=num_harmonics,
            dropout=dropout
        )
        
        # Wave normalization (RMS-style, preserves wave physics)
        self.norm1 = WaveRMSNorm(num_waves, num_harmonics)
        self.norm2 = WaveRMSNorm(num_waves, num_harmonics)
        
    def forward(self, wave_state: WaveState) -> WaveState:
        """Wave → Wave transformation."""
        
        # Normalize
        norm_state = self.norm1(wave_state)
        
        # Wave interference attention
        attn_state = self.wave_attention(norm_state)
        
        # Residual in wave space
        res_freqs = wave_state.freqs + attn_state.freqs
        res_phases = wave_state.phases + attn_state.phases
        res_amps = wave_state.amps + attn_state.amps
        res_state = WaveState(res_freqs, res_phases, res_amps)
        
        # Normalize again
        norm_state2 = self.norm2(res_state)
        
        # Wave MLP
        mlp_state = self.wave_mlp(norm_state2)
        
        # Final residual
        out_freqs = res_state.freqs + mlp_state.freqs
        out_phases = res_state.phases + mlp_state.phases
        out_amps = res_state.amps + mlp_state.amps
        
        return WaveState(out_freqs, out_phases, out_amps)


class PureWaveInterference(nn.Module):
    """
    Pure Wave Superposition Interference - Projection via Wave Addition!
    
    KEY INSIGHT: Instead of matrix projections, we PROJECT waves by
    ADDING/SUBTRACTING other learnable waves. This is pure wave physics!
    
    Projection = Wave Superposition:
        W_projected = W_input + W_projection  (learnable wave!)
    
    The projection wave W_projection has its own:
        - Frequencies (learnable)
        - Phases (learnable)  
        - Amplitudes (learnable)
    
    This creates INFINITE expressivity because:
        - Superposition creates beating patterns
        - Phase differences create interference
        - Frequency mixing creates harmonics
        - Everything emerges from wave physics!
    
    "Attention" emerges from interference of superposed waves.
    It's not learned attention weights - it's PHYSICS!
    """
    
    def __init__(self, num_waves, num_harmonics, num_heads, dropout=0.1):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        self.num_channels = num_heads  # Multiple interference channels
        
        # === PROJECTION WAVES ===
        # These are LEARNABLE WAVES that get added to input waves
        # Each channel has its own projection wave for emitter and receiver
        
        # Emitter projection waves (what we add before emitting)
        # Shape: (num_channels, num_waves) - one wave per channel
        self.emit_proj_freqs = nn.Parameter(
            self._init_projection_freqs(self.num_channels, num_waves)
        )
        self.emit_proj_phases = nn.Parameter(
            torch.rand(self.num_channels, num_waves) * 2 * math.pi
        )
        self.emit_proj_amps = nn.Parameter(
            torch.ones(self.num_channels, num_waves, num_harmonics) * 0.5
        )
        
        # Receiver projection waves (what we add before receiving)
        self.recv_proj_freqs = nn.Parameter(
            self._init_projection_freqs(self.num_channels, num_waves)
        )
        self.recv_proj_phases = nn.Parameter(
            torch.rand(self.num_channels, num_waves) * 2 * math.pi
        )
        self.recv_proj_amps = nn.Parameter(
            torch.ones(self.num_channels, num_waves, num_harmonics) * 0.5
        )
        
        # Field content projection wave (what propagates through the field)
        self.field_proj_freqs = nn.Parameter(
            self._init_projection_freqs(1, num_waves).squeeze(0)
        )
        self.field_proj_phases = nn.Parameter(
            torch.rand(num_waves) * 2 * math.pi
        )
        self.field_proj_amps = nn.Parameter(
            torch.ones(num_waves, num_harmonics) * 0.5
        )
        
        # === SUPERPOSITION WEIGHTS ===
        # How much of the projection wave to add (learnable mixing)
        self.emit_mix = nn.Parameter(torch.ones(self.num_channels) * 0.5)
        self.recv_mix = nn.Parameter(torch.ones(self.num_channels) * 0.5)
        self.field_mix = nn.Parameter(torch.tensor(0.5))
        
        # === OUTPUT PROJECTION WAVE ===
        # Final wave to add when absorbing from field
        self.output_proj_freqs = nn.Parameter(torch.zeros(num_waves))
        self.output_proj_phases = nn.Parameter(torch.zeros(num_waves))
        self.output_proj_amps = nn.Parameter(torch.zeros(num_waves, num_harmonics))
        
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8
        
        # Physics parameters
        self.interference_strength = nn.Parameter(torch.ones(1))
        
    def _init_projection_freqs(self, num_channels, num_waves):
        """
        Initialize projection wave frequencies.
        
        Each channel's projection wave is tuned to different frequency bands.
        This creates diverse interference patterns across channels.
        """
        freqs = torch.zeros(num_channels, num_waves)
        base_spectrum = torch.logspace(math.log10(0.005), math.log10(2.0), num_waves)
        
        for c in range(num_channels):
            # Each channel shifts the base spectrum slightly
            shift = (c - num_channels / 2) * 0.1
            freqs[c] = base_spectrum * (1.0 + shift)
            
            # Add some randomness
            freqs[c] = freqs[c] * (1.0 + torch.randn(num_waves) * 0.1)
        
        return freqs
        
    def forward(self, wave_state: WaveState) -> WaveState:
        """
        Wave interference via WAVE SUPERPOSITION projections!
        
        Pipeline:
        1. EMIT: Add projection wave to input → emitter wave
        2. RECEIVE: Add projection wave to input → receiver wave  
        3. INTERFERE: Emitter and receiver waves interfere
        4. SUPERPOSE: Weighted sum based on interference strength
        5. OUTPUT: Add output projection wave
        
        NO MATRICES! Just wave addition and interference physics!
        """
        B, T, W = wave_state.freqs.shape
        C = self.num_channels
        NH = self.num_harmonics
        device = wave_state.freqs.device
        
        # === STEP 1: CREATE EMITTER WAVES via Superposition ===
        # emitter = input_wave + emit_proj_wave (per channel)
        # Shape: (B, C, T, W)
        
        emit_freqs = wave_state.freqs.unsqueeze(1) + self.emit_mix.view(1, C, 1, 1) * self.emit_proj_freqs.unsqueeze(0).unsqueeze(2)
        emit_phases = wave_state.phases.unsqueeze(1) + self.emit_mix.view(1, C, 1, 1) * self.emit_proj_phases.unsqueeze(0).unsqueeze(2)
        emit_amps = wave_state.amps.unsqueeze(1) + self.emit_mix.view(1, C, 1, 1, 1) * self.emit_proj_amps.unsqueeze(0).unsqueeze(2)
        
        # Ensure positive values
        emit_freqs = F.softplus(emit_freqs) + self.eps
        emit_amps = F.softplus(emit_amps)
        
        # === STEP 2: CREATE RECEIVER WAVES via Superposition ===
        # receiver = input_wave + recv_proj_wave (per channel)
        
        recv_freqs = wave_state.freqs.unsqueeze(1) + self.recv_mix.view(1, C, 1, 1) * self.recv_proj_freqs.unsqueeze(0).unsqueeze(2)
        recv_phases = wave_state.phases.unsqueeze(1) + self.recv_mix.view(1, C, 1, 1) * self.recv_proj_phases.unsqueeze(0).unsqueeze(2)
        recv_amps = wave_state.amps.unsqueeze(1) + self.recv_mix.view(1, C, 1, 1, 1) * self.recv_proj_amps.unsqueeze(0).unsqueeze(2)
        
        recv_freqs = F.softplus(recv_freqs) + self.eps
        recv_amps = F.softplus(recv_amps)
        
        # === STEP 3: CREATE FIELD CONTENT via Superposition ===
        # field = input_wave + field_proj_wave
        
        field_freqs = wave_state.freqs + self.field_mix * self.field_proj_freqs
        field_phases = wave_state.phases + self.field_mix * self.field_proj_phases
        field_amps = wave_state.amps + self.field_mix * self.field_proj_amps
        
        field_freqs = F.softplus(field_freqs) + self.eps
        field_amps = F.softplus(field_amps)
        
        # === STEP 4: TEMPORAL PHASE EVOLUTION ===
        positions = torch.arange(T, device=device, dtype=torch.float32)
        
        # Emitter phase at each position: θ = ω*t + φ
        emit_theta = emit_freqs * positions.view(1, 1, T, 1) + emit_phases  # (B, C, T, W)
        
        # Receiver phase at each position
        recv_theta = recv_freqs * positions.view(1, 1, T, 1) + recv_phases  # (B, C, T, W)
        
        # === STEP 5: WAVE INTERFERENCE ===
        # Create phasors: A * e^(iθ)
        emit_amp_total = emit_amps.sum(dim=-1)  # Sum harmonics: (B, C, T, W)
        recv_amp_total = recv_amps.sum(dim=-1)
        
        emit_phasor = emit_amp_total * torch.exp(1j * emit_theta.to(torch.complex64))
        recv_phasor = recv_amp_total * torch.exp(1j * recv_theta.to(torch.complex64))
        
        # Interference: Re(emit · recv*) for each pair of positions
        # This is the PHYSICS: I = A₁*A₂*cos(θ₁ - θ₂)
        interference = torch.matmul(
            emit_phasor,  # (B, C, T, W)
            recv_phasor.conj().transpose(-2, -1)  # (B, C, W, T)
        ).real  # (B, C, T, T)
        
        # Scale by number of waves
        interference = interference / (W ** 0.5)
        
        # === STEP 6: FULL INTENSITY FORMULA ===
        # I = A_emit² + A_recv² + 2*A_emit*A_recv*cos(Δφ)
        emit_energy = (emit_amp_total ** 2).sum(dim=-1, keepdim=True)  # (B, C, T, 1)
        recv_energy = (recv_amp_total ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)  # (B, C, 1, T)
        
        intensity = emit_energy + recv_energy + 2 * interference * self.interference_strength
        
        # === STEP 7: PHYSICS-BASED NORMALIZATION ===
        emit_total = emit_amp_total.sum(dim=-1, keepdim=True)
        recv_total = recv_amp_total.sum(dim=-1, keepdim=True).transpose(-2, -1)
        max_intensity = (emit_total + recv_total) ** 2 + self.eps
        
        coupling = intensity / max_intensity  # Transmission coefficient [0, 1]
        
        # === STEP 8: CAUSAL MASK (Light Cone) ===
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        coupling = coupling.masked_fill(causal_mask, 0.0)
        
        # Normalize coupling weights
        coupling = coupling / (coupling.sum(dim=-1, keepdim=True) + self.eps)
        coupling = self.dropout(coupling)
        
        # === STEP 9: WAVE SUPERPOSITION (Weighted Sum) ===
        # Apply coupling to field content
        out_freqs = torch.matmul(coupling, field_freqs.unsqueeze(1).expand(-1, C, -1, -1))
        out_phases = torch.matmul(coupling, field_phases.unsqueeze(1).expand(-1, C, -1, -1))
        
        field_amps_flat = field_amps.view(B, T, -1).unsqueeze(1).expand(-1, C, -1, -1)
        out_amps_flat = torch.matmul(coupling, field_amps_flat)
        
        # Average across channels
        out_freqs = out_freqs.mean(dim=1)  # (B, T, W)
        out_phases = out_phases.mean(dim=1)
        out_amps_flat = out_amps_flat.mean(dim=1)
        out_amps = out_amps_flat.view(B, T, W, NH)
        
        # === STEP 10: OUTPUT PROJECTION via Wave Addition ===
        final_freqs = out_freqs + self.output_proj_freqs
        final_phases = out_phases + self.output_proj_phases
        final_amps = out_amps + self.output_proj_amps
        
        return WaveState(final_freqs, final_phases, final_amps)


class PureWaveMLP(nn.Module):
    """
    Wave-Native MLP (Resonance Filtering).
    
    Operates directly on wave parameters:
    - Frequency transformation (resonance coupling)
    - Phase transformation (phase relationships)
    - Amplitude transformation (energy redistribution)
    """
    
    def __init__(self, num_waves, num_harmonics, dropout=0.1):
        super().__init__()
        
        # Frequency MLP
        self.freq_mlp = nn.Sequential(
            nn.Linear(num_waves, 4 * num_waves),
            nn.GELU(),
            nn.Linear(4 * num_waves, num_waves),
            nn.Dropout(dropout)
        )
        
        # Phase MLP
        self.phase_mlp = nn.Sequential(
            nn.Linear(num_waves, 4 * num_waves),
            nn.GELU(),
            nn.Linear(4 * num_waves, num_waves),
            nn.Dropout(dropout)
        )
        
        # Amplitude MLP
        amp_dim = num_waves * num_harmonics
        self.amp_mlp = nn.Sequential(
            nn.Linear(amp_dim, 4 * amp_dim),
            nn.GELU(),
            nn.Linear(4 * amp_dim, amp_dim),
            nn.Dropout(dropout)
        )
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # FIXED: Initialize MLP weights smaller for stability
        with torch.no_grad():
            for module in [self.freq_mlp, self.phase_mlp, self.amp_mlp]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data *= 0.1
                        if layer.bias is not None:
                            layer.bias.data.zero_()
        
    def forward(self, wave_state: WaveState) -> WaveState:
        B, T, W = wave_state.freqs.shape
        
        new_freqs = self.freq_mlp(wave_state.freqs)
        new_phases = self.phase_mlp(wave_state.phases)
        
        amps_flat = wave_state.amps.view(B, T, -1)
        new_amps = self.amp_mlp(amps_flat).view(B, T, W, -1)
        
        return WaveState(new_freqs, new_phases, new_amps)


class WaveRMSNorm(nn.Module):
    """RMS Normalization for wave parameters."""
    
    def __init__(self, num_waves, num_harmonics):
        super().__init__()
        self.freq_scale = nn.Parameter(torch.ones(num_waves))
        self.phase_scale = nn.Parameter(torch.ones(num_waves))
        self.amp_scale = nn.Parameter(torch.ones(num_waves, num_harmonics))
        self.eps = 1e-8
        
    def forward(self, wave_state: WaveState) -> WaveState:
        # RMS normalize each parameter type
        freq_rms = wave_state.freqs.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        norm_freqs = wave_state.freqs / freq_rms * self.freq_scale
        
        phase_rms = wave_state.phases.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        norm_phases = wave_state.phases / phase_rms * self.phase_scale
        
        amp_rms = wave_state.amps.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=self.eps)
        norm_amps = wave_state.amps / amp_rms * self.amp_scale
        
        return WaveState(norm_freqs, norm_phases, norm_amps)


class WaveCollapse(nn.Module):
    """
    Enhanced Wave → Logits (Measurement/Collapse) with Attention.
    
    The final "measurement" that collapses the wave function
    into a probability distribution over vocabulary.
    
    Enhanced with:
    - Multi-head attention over wave parameters
    - Frequency-specific processing
    - Harmonic content analysis
    """
    
    def __init__(self, vocab_size, num_waves, num_harmonics):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # === WAVE PARAMETER PROCESSING ===
        # Separate processing for different wave parameter types
        self.freq_processor = nn.Sequential(
            nn.Linear(num_waves, num_waves * 2),
            nn.GELU(),
            nn.Linear(num_waves * 2, num_waves),
            nn.Dropout(0.1)
        )
        
        self.phase_processor = nn.Sequential(
            nn.Linear(num_waves, num_waves * 2),
            nn.GELU(),
            nn.Linear(num_waves * 2, num_waves),
            nn.Dropout(0.1)
        )
        
        amp_dim = num_waves * num_harmonics
        self.amp_processor = nn.Sequential(
            nn.Linear(amp_dim, amp_dim * 2),
            nn.GELU(),
            nn.Linear(amp_dim * 2, amp_dim),
            nn.Dropout(0.1)
        )
        
        # === WAVE ATTENTION ===
        # Self-attention over wave parameters to capture cross-wave interactions
        wave_dim = num_waves * 3  # freqs + phases + processed_amps
        self.wave_attention = nn.MultiheadAttention(
            embed_dim=wave_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # === HARMONIC ANALYSIS ===
        # Analyze harmonic content for richer representation
        self.harmonic_analyzer = nn.Sequential(
            nn.Linear(num_harmonics, num_harmonics * 2),
            nn.GELU(),
            nn.Linear(num_harmonics * 2, num_harmonics),
            nn.Dropout(0.1)
        )
        
        # === FINAL PROJECTION ===
        final_dim = wave_dim + num_waves * num_harmonics  # wave_features + harmonic_features
        self.wave_norm = nn.LayerNorm(final_dim)
        
        # Multi-layer projection for more expressivity
        self.collapse_proj = nn.Sequential(
            nn.Linear(final_dim, final_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(final_dim * 2, vocab_size)
        )
        
        # Initialize with smaller weights for stability
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stability"""
        for module in [self.freq_processor, self.phase_processor, self.amp_processor, 
                      self.harmonic_analyzer, self.collapse_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    layer.weight.data *= 0.1
                    if layer.bias is not None:
                        layer.bias.data.zero_()
        
    def forward(self, wave_state: WaveState) -> torch.Tensor:
        B, T, W = wave_state.freqs.shape
        H = wave_state.amps.shape[-1]
        
        # === PROCESS WAVE PARAMETERS ===
        # Normalize and process frequencies
        freq_norm = torch.tanh(wave_state.freqs * 0.2)  # Increased sensitivity
        processed_freqs = self.freq_processor(freq_norm)
        
        # Process phases with circular normalization
        phase_norm = torch.stack([torch.sin(wave_state.phases), torch.cos(wave_state.phases)], dim=-1)
        phase_norm = phase_norm.mean(dim=-1)  # Combine sin/cos
        processed_phases = self.phase_processor(phase_norm)
        
        # Process amplitudes
        amps_flat = wave_state.amps.view(B, T, -1)
        amp_norm = torch.clamp(amps_flat, min=0.0, max=3.0)  # Increased range
        processed_amps_flat = self.amp_processor(amp_norm)
        processed_amps = processed_amps_flat.view(B, T, W, H)
        
        # === HARMONIC ANALYSIS ===
        # Analyze harmonic content for each wave
        harmonic_features = []
        for wave_idx in range(W):
            wave_harmonics = processed_amps[:, :, wave_idx, :]  # (B, T, H)
            analyzed_harmonics = self.harmonic_analyzer(wave_harmonics)
            harmonic_features.append(analyzed_harmonics)
        
        harmonic_features = torch.stack(harmonic_features, dim=2)  # (B, T, W, H)
        harmonic_features_flat = harmonic_features.view(B, T, -1)
        
        # === WAVE ATTENTION ===
        # Combine processed wave parameters
        # Summarize amplitudes: sum across harmonics to get per-wave amplitude
        amps_per_wave = processed_amps.sum(dim=-1)  # (B, T, W) - sum harmonics
        
        wave_features = torch.cat([
            processed_freqs,
            processed_phases,
            amps_per_wave  # Summarized amps per wave
        ], dim=-1)  # (B, T, wave_dim = 3*W)
        
        # Self-attention over wave parameters
        attended_waves, _ = self.wave_attention(wave_features, wave_features, wave_features)
        
        # === COMBINE ALL FEATURES ===
        final_features = torch.cat([
            attended_waves,
            harmonic_features_flat
        ], dim=-1)  # (B, T, final_dim)
        
        # Normalize and project to vocabulary
        final_features = self.wave_norm(final_features)
        logits = self.collapse_proj(final_features)  # (B, T, vocab_size)
        
        return logits


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

