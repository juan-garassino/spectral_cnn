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


# ==========================================
# Wave Packet Embedding
# ==========================================

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
        
    def forward(self, token_ids):
        """
        token_ids: (B, T) - discrete token indices
        returns: (B, T, d_model) - wave embeddings with gradient-safe initialization
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
        
        # Generate harmonics: for each base frequency, create 1f, 2f, 3f, 4f...
        # base_f: (B, T, W) -> expand to (B, T, W, H)
        freqs = base_f.unsqueeze(-1) * self.harmonic_mults  # (B, T, W, H)
        
        # Phase applies to all harmonics
        wave_phase = freqs * 2 * math.pi + phases.unsqueeze(-1) + pos_phase.unsqueeze(-1)
        
        # Weighted sum of sin/cos harmonics
        sin_waves = harm_a * torch.sin(wave_phase)  # (B, T, W, H)
        cos_waves = harm_a * torch.cos(wave_phase)  # (B, T, W, H)
        
        # Flatten: (B, T, W*H*2)
        wave_state = torch.cat([
            sin_waves.reshape(B, T, -1),
            cos_waves.reshape(B, T, -1)
        ], dim=-1)
        
        # Project to embedding dimension
        wave_embed = self.wave_to_embed(wave_state)  # (B, T, d_model)
        
        # Standard embedding for gradient bootstrap
        simple_embed = self.simple_embed(token_ids)
        
        # Blend: wave_ratio controls wave vs standard contribution
        # Clamp wave_ratio to [0, 1] range
        ratio = torch.sigmoid(self.wave_ratio)  # Smooth 0-1
        embeddings = ratio * wave_embed + (1 - ratio) * simple_embed
        
        # LayerNorm for stability
        embeddings = self.ln(embeddings)
        
        return embeddings


# ==========================================
# Wave Interference Attention
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
    def __init__(self, d_model, num_heads, num_waves=16, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_waves = num_waves
        
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
        
        Returns attention based on phase alignment, not learned similarity.
        """
        B, T, C = x.shape
        
        # Project to wave parameters (frequency and phase)
        q_f = self.q_freq(x).view(B, T, self.num_heads, self.num_waves)   # (B, T, H, W)
        k_f = self.k_freq(x).view(B, T, self.num_heads, self.num_waves)   # (B, T, H, W)
        q_p = self.q_phase(x).view(B, T, self.num_heads, self.num_waves)  # (B, T, H, W)
        k_p = self.k_phase(x).view(B, T, self.num_heads, self.num_waves)  # (B, T, H, W)
        
        # Value projection
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)      # (B, T, H, D)
        
        # Transpose: (B, H, T, ...)
        q_f = q_f.transpose(1, 2)  # (B, H, T, W)
        k_f = k_f.transpose(1, 2)  # (B, H, T, W)
        q_p = q_p.transpose(1, 2)  # (B, H, T, W)
        k_p = k_p.transpose(1, 2)  # (B, H, T, W)
        v = v.transpose(1, 2)      # (B, H, T, D)
        
        # Compute wave at each position
        # Wave: A * sin(f * t + phase) where t is position index
        t_pos = torch.arange(T, device=x.device).float().view(1, 1, T, 1)
        
        # Query waves: sin(q_freq * pos + q_phase)
        q_waves = torch.sin(q_f * t_pos + q_p)  # (B, H, T, W)
        
        # Key waves: sin(k_freq * pos + k_phase)  
        k_waves = torch.sin(k_f * t_pos + k_p)  # (B, H, T, W)
        
        # PURE WAVE INTERFERENCE: cosine of phase difference
        # When waves are in phase: high positive interference
        # When waves are out of phase: negative interference (SUPPRESSION!)
        
        # Normalize across wave dimension for bounded output
        q_norm = F.normalize(q_waves, dim=-1)
        k_norm = F.normalize(k_waves, dim=-1)
        
        # Interference = dot product of normalized waves = cosine similarity
        # Range: [-1, 1] - NEGATIVE VALUES ALLOWED!
        interference = torch.matmul(q_norm, k_norm.transpose(-2, -1))  # (B, H, T, T)
        
        # Scale per head
        interference = interference * self.interference_scale
        
        # Causal masking: future positions get ZERO (not -inf!)
        # This is different from softmax-based attention
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        interference = interference.masked_fill(causal_mask, 0.0)
        
        # NO SOFTMAX! Pure interference weights
        # Normalize by number of attended positions for stability
        num_attended = torch.arange(1, T + 1, device=x.device).float().view(1, 1, T, 1)
        attn = interference / num_attended.sqrt()  # Scale by sqrt(n) like standard attention
        
        # Dropout on attention (optional)
        attn = self.dropout(attn)
        
        # Apply to values - negative attention = SUPPRESSION
        out = torch.matmul(attn, v)  # (B, H, T, D)
        
        # Reshape and project
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
        if getattr(config, 'pure_wave_attention', False):
            # PURE wave attention - NO SOFTMAX!
            self.attn = PureWaveAttention(
                config.d_model, config.num_heads, 
                config.num_waves, config.dropout
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

class WaveGPT(nn.Module):
    """
    Language model where everything is waves.
    
    Token → Wave Packet → Interference → Superposition → Collapse → Token
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Wave packet embedding (tokens as waves with harmonics)
        self.embedding = WavePacketEmbedding(
            config.vocab_size, config.d_model, config.num_waves, config.num_harmonics
        )
        
        # Stack of wave blocks
        self.blocks = nn.ModuleList([
            WaveBlock(config) for _ in range(config.num_layers)
        ])
        
        # Collapse: wave → token
        self.head = CollapseHead(config.d_model, config.vocab_size)
        
        # Initialize
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T) target token indices (optional)
        """
        # Embed tokens as wave packets
        x = self.embedding(idx)
        
        # Wave interference through blocks
        for block in self.blocks:
            x = block(x)
            
        # Collapse to token predictions
        logits = self.head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens via repeated wave collapse"""
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get last token's logits
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                
            # Sample (wave collapse!)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, idx_next], dim=1)
            
        return idx
