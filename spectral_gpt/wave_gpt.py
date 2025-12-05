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
    block_size: int        # Context length
    dropout: float = 0.1


# ==========================================
# Wave Packet Embedding
# ==========================================

class WavePacketEmbedding(nn.Module):
    """
    Embed tokens as wave packets instead of discrete vectors.
    
    Each token becomes a superposition of waves:
    - Learnable base frequency per token
    - Learnable phase offset per token  
    - Learnable amplitude per wave component
    
    Output: Wave state that can interfere with other tokens
    """
    def __init__(self, vocab_size, d_model, num_waves=16):
        super().__init__()
        self.d_model = d_model
        self.num_waves = num_waves
        
        # Each token has learnable wave parameters
        # Frequencies: what "pitch" does this token resonate at?
        self.token_freqs = nn.Parameter(torch.randn(vocab_size, num_waves) * 0.5 + 1.0)
        
        # Phases: where in the wave cycle does this token start?
        self.token_phases = nn.Parameter(torch.rand(vocab_size, num_waves) * 2 * math.pi)
        
        # Amplitudes: how strong is each wave component?
        self.token_amps = nn.Parameter(torch.randn(vocab_size, num_waves) * 0.1)
        
        # Project wave state to d_model dimension
        self.wave_to_embed = nn.Linear(num_waves * 2, d_model)  # *2 for sin+cos
        
        # Learnable positional wave (position = phase modulation)
        self.pos_freq = nn.Parameter(torch.randn(1, 1, num_waves) * 0.1)
        
    def forward(self, token_ids):
        """
        token_ids: (B, T) - discrete token indices
        returns: (B, T, d_model) - continuous wave embeddings
        """
        B, T = token_ids.shape
        device = token_ids.device
        
        # Get wave parameters for each token
        freqs = self.token_freqs[token_ids]    # (B, T, num_waves)
        phases = self.token_phases[token_ids]  # (B, T, num_waves)
        amps = self.token_amps[token_ids]      # (B, T, num_waves)
        
        # Add positional encoding as phase modulation
        positions = torch.arange(T, device=device).float().view(1, T, 1)
        pos_phase = positions * self.pos_freq  # Position affects phase
        
        # Generate wave packet: superposition of sin and cos waves
        # This creates a rich continuous representation
        wave_phase = freqs * 2 * math.pi + phases + pos_phase
        
        sin_waves = amps * torch.sin(wave_phase)  # (B, T, num_waves)
        cos_waves = amps * torch.cos(wave_phase)  # (B, T, num_waves)
        
        # Concatenate sin and cos for full wave representation
        wave_state = torch.cat([sin_waves, cos_waves], dim=-1)  # (B, T, num_waves*2)
        
        # Project to embedding dimension
        embeddings = self.wave_to_embed(wave_state)  # (B, T, d_model)
        
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
        
        # Wave packet embedding (tokens as waves)
        self.embedding = WavePacketEmbedding(
            config.vocab_size, config.d_model, config.num_waves
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
