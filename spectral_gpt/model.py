"""
Unified Spectral GPT Model 🌊🧠

Combines two powerful spectral paradigms into a single architecture:
1. Spectral Parameterization ("Wave Weights"): Q/K/V/MLP weights are wave superpositions
2. Spectral Architecture ("FFT Mixing"): Attention replaced by global FFT mixing

Configurable via `layer_type` and `weight_type`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from typing import Optional, Literal

# Import UserWaveLinear for "Wave Weights" mode
import sys
sys.path.append('..')
from src.models.layers import UserWaveLinear


# ==========================================
# 1. SPECTRAL EMBEDDINGS
# ==========================================

class SpectralEmbedding(nn.Module):
    """Tokens as wave superpositions (continuous representation)."""
    def __init__(self, vocab_size, d_model, num_waves=64, num_harmonics=7, wave_mode="fourier_series", init_mode="standard"):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        
        # Wave generator: Token ID -> Wave Pattern
        self.wave_generator = UserWaveLinear(
            in_dim=1, out_dim=d_model, num_waves=num_waves, 
            num_harmonics=num_harmonics, wave_mode=wave_mode,
            adaptive_freqs=True, per_neuron_coeffs=False,
            init_mode=init_mode
        )
        
    def forward(self, x):
        # Normalize token IDs to [0, 1]
        normalized_ids = x.float() / self.vocab_size
        normalized_ids = normalized_ids.unsqueeze(-1)
        return self.wave_generator(normalized_ids) * self.scale

    def freeze_high_frequencies(self, threshold):
        self.wave_generator.freeze_high_frequencies(threshold)

    def progressive_unfreeze(self, epoch, total, strategy):
        self.wave_generator.progressive_unfreeze_schedule(epoch, total, strategy)
    
    def constrain_energy(self):
        self.wave_generator.constrain_energy()


class SpectralPositionalEncoding(nn.Module):
    """Learnable wave-based positional encoding."""
    def __init__(self, d_model, max_len=5000, num_waves=16, num_harmonics=5, init_mode="standard"):
        super().__init__()
        self.max_len = max_len
        self.wave_encoder = UserWaveLinear(
            in_dim=1, out_dim=d_model, num_waves=num_waves,
            num_harmonics=num_harmonics, wave_mode="fourier_series",
            adaptive_freqs=True, init_mode=init_mode
        )
        
    def forward(self, x):
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).float().unsqueeze(-1) / self.max_len
        pe = self.wave_encoder(positions)
        return x + pe.unsqueeze(0)

    def freeze_high_frequencies(self, threshold):
        self.wave_encoder.freeze_high_frequencies(threshold)

    def progressive_unfreeze(self, epoch, total, strategy):
        self.wave_encoder.progressive_unfreeze_schedule(epoch, total, strategy)

    def constrain_energy(self):
        self.wave_encoder.constrain_energy()


# ==========================================
# 2. MIXING LAYERS (The "Brain")
# ==========================================

class SpectralGatingMixing(nn.Module):
    """
    FFT-based Global Mixing (The "Transformer Killer" Layer).
    O(N log N) complexity. Replaces Attention.
    """
    def __init__(self, dim, seq_len, init_mode="standard"):
        super().__init__()
        self.dim = dim
        self.n_fft = seq_len * 2
        self.n_freqs = self.n_fft // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(self.n_freqs, dim, 2) * 0.02)
        
        if init_mode == "dft":
            # Initialize as Identity / All-Pass Filter
            # Real part = 1, Imag part = 0
            nn.init.ones_(self.complex_weight[:, :, 0])
            nn.init.zeros_(self.complex_weight[:, :, 1])
            # Add tiny noise to break symmetry
            self.complex_weight.data += torch.randn_like(self.complex_weight) * 0.001
        elif init_mode == "holographic":
            # 1/f scaling for frequencies
            with torch.no_grad():
                for f in range(self.n_freqs):
                    scale = 1.0 / (f + 1)
                    self.complex_weight.data[f] *= scale

    def forward(self, x, progress=1.0):
        B, N, C = x.shape
        
        # Pad to the expected FFT size
        pad_len = self.n_fft - N
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x[:, :self.n_fft, :]
        
        # FFT
        x_fft = torch.fft.rfft(x_padded.float(), dim=1, norm='ortho')
        
        # Filter
        weight = torch.view_as_complex(self.complex_weight)
        if self.training:
            # Curriculum: Low-pass filter
            keep_ratio = 0.1 + (0.9 * progress)
            cutoff = int(self.n_freqs * keep_ratio)
            mask = torch.zeros(self.n_freqs, device=x.device)
            mask[:cutoff] = 1.0
            weight = weight * mask.unsqueeze(1)
            
        out_fft = x_fft * weight.unsqueeze(0)
        
        # IFFT
        out = torch.fft.irfft(out_fft, n=self.n_fft, dim=1, norm='ortho')
        return out[:, :N, :].to(x.dtype)
    
    def constrain_energy(self):
        """Hamiltonian Descent: Normalize filter energy."""
        with torch.no_grad():
            # Normalize complex weights [freqs, dim, 2]
            # We want sum(|w|^2) to be constant or bounded
            norm = torch.norm(self.complex_weight, p=2, dim=2, keepdim=True) # Magnitude
            # Normalize across frequencies? Or per frequency?
            # Let's normalize per dimension across frequencies to conserve energy per channel
            channel_energy = torch.norm(norm, p=2, dim=0, keepdim=True)
            self.complex_weight.data.div_(channel_energy + 1e-8)


class ComplexSpectralAttention(nn.Module):
    """
    Complex-Valued Attention Mechanism 🌊
    
    Implements true wave interference using complex numbers:
    Q, K, V are complex-valued vectors.
    Attention Score = Re(Q * K^H)  (Real part of inner product)
    Output = Score * V
    """
    def __init__(self, dim, num_heads, num_waves=12, num_harmonics=5, dropout=0.1, init_mode="standard"):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Complex Projections (Output dimension is 2*dim for Real+Imag parts)
        # We use UserWaveLinear to generate the complex weights
        self.q_proj = UserWaveLinear(dim, dim * 2, num_waves, num_harmonics, adaptive_freqs=True, init_mode=init_mode)
        self.k_proj = UserWaveLinear(dim, dim * 2, num_waves, num_harmonics, adaptive_freqs=True, init_mode=init_mode)
        self.v_proj = UserWaveLinear(dim, dim * 2, num_waves, num_harmonics, adaptive_freqs=True, init_mode=init_mode)
        self.o_proj = UserWaveLinear(dim * 2, dim, num_waves, num_harmonics, adaptive_freqs=True, init_mode=init_mode)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, progress=1.0):
        B, N, C = x.shape
        
        # Project to Complex (Real, Imag)
        # Shape: [B, N, 2*C] -> [B, N, H, 2*D] -> [B, H, N, 2*D]
        q = self.q_proj(x).view(B, N, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        
        # Split into Real and Imag parts
        q_r, q_i = q.chunk(2, dim=-1)
        k_r, k_i = k.chunk(2, dim=-1)
        v_r, v_i = v.chunk(2, dim=-1)
        
        # Complex Inner Product: (a+bi)(c-di) = (ac+bd) + i(bc-ad)
        # We only care about the magnitude/interference for the score, or maybe just the real part?
        # Standard complex attention usually uses Re(Q K^H) for logits
        
        # Real part of Q * K^H = Q_r * K_r + Q_i * K_i
        attn_scores = (q_r @ k_r.transpose(-2, -1)) + (q_i @ k_i.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        # Causal Masking (Internal)
        mask = torch.tril(torch.ones(N, N, device=x.device)).view(1, 1, N, N)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Output = Prob * V (Complex multiplication)
        # Out_r = Prob * V_r
        # Out_i = Prob * V_i
        # Note: This is a simplification. True complex attention might rotate V too.
        # But for "interference", weighting the complex vector V by the real probability is standard.
        
        out_r = attn_probs @ v_r
        out_i = attn_probs @ v_i
        
        # Concatenate back to [B, H, N, 2*D]
        out = torch.cat([out_r, out_i], dim=-1)
        
        # Reshape to [B, N, 2*C]
        out = out.transpose(1, 2).contiguous().view(B, N, 2 * self.dim)
        
        # Final projection back to real [B, N, C]
        return self.o_proj(out)


class SpectralAttention(nn.Module):
    """
    Wave-based Multi-Head Attention.
    Standard Attention mechanism, but Q/K/V/Out are Wave Layers.
    """
    def __init__(self, dim, num_heads, num_waves=12, num_harmonics=5, dropout=0.1, init_mode="standard"):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Wave-based projections
        self.q_proj = UserWaveLinear(dim, dim, num_waves, num_harmonics, adaptive_freqs=True, init_mode=init_mode)
        self.k_proj = UserWaveLinear(dim, dim, num_waves, num_harmonics, adaptive_freqs=True, init_mode=init_mode)
        self.v_proj = UserWaveLinear(dim, dim, num_waves, num_harmonics, adaptive_freqs=True, init_mode=init_mode)
        self.o_proj = UserWaveLinear(dim, dim, num_waves, num_harmonics, adaptive_freqs=True, init_mode=init_mode)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, progress=1.0):
        B, T, C = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, -1).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, -1).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, -1).transpose(1, 2)
        
        att = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ V
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

    def freeze_high_frequencies(self, threshold):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            m.freeze_high_frequencies(threshold)

    def progressive_unfreeze(self, epoch, total, strategy):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            m.progressive_unfreeze_schedule(epoch, total, strategy)

    def constrain_energy(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            m.constrain_energy()


# ==========================================
# 3. UNIFIED BLOCK & MODEL
# ==========================================

class SpectralBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Mixing Layer Selection
        if config.layer_type == 'fft':
            self.mixer = SpectralGatingMixing(config.d_model, config.block_size, init_mode=getattr(config, 'init_mode', 'standard'))
        elif config.layer_type == 'attention':
            if config.weight_type == 'wave':
                self.mixer = SpectralAttention(config.d_model, config.num_heads, 
                                             config.num_waves, config.num_harmonics, config.dropout,
                                             init_mode=getattr(config, 'init_mode', 'standard'))
            else:
                self.mixer = nn.MultiheadAttention(config.d_model, config.num_heads, 
                                                 dropout=config.dropout, batch_first=True)
        
        # MLP Selection
        act_type = getattr(config, 'activation_type', 'gelu')
        
        if act_type == 'bilinear':
            # SwiGLU / Bilinear style (requires different dim structure usually, but we'll adapt)
            # Standard SwiGLU: (Swish(xW) * xV)W2
            # Here we'll just use a GLU-like structure within the sequential block if possible, 
            # or just replace the activation function.
            # Let's implement a custom module for the activation to keep it clean.
            activation = nn.SiLU() # Swish
        elif act_type == 'modulate':
            # Physics-based: x * cos(x)
            # Creates sidebands (sum/diff frequencies)
            class CosineModulation(nn.Module):
                def forward(self, x): return x * torch.cos(x)
            activation = CosineModulation()
        else:
            activation = nn.GELU()

        if config.weight_type == 'wave':
            self.mlp = nn.Sequential(
                UserWaveLinear(config.d_model, config.d_ff, config.num_waves, config.num_harmonics, 
                             adaptive_freqs=True, init_mode=getattr(config, 'init_mode', 'standard')),
                activation,
                nn.Dropout(config.dropout),
                UserWaveLinear(config.d_ff, config.d_model, config.num_waves, config.num_harmonics, 
                             adaptive_freqs=True, init_mode=getattr(config, 'init_mode', 'standard')),
                nn.Dropout(config.dropout)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                activation,
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(config.dropout)
            )

    def forward(self, x, progress=1.0):
        # Mixer
        if self.config.layer_type == 'fft':
            x = x + self.mixer(self.norm1(x), progress)
        elif self.config.layer_type == 'attention':
            if self.config.weight_type == 'wave':
                x = x + self.mixer(self.norm1(x), progress)
            else:
                # Standard Attention needs mask
                attn_out, _ = self.mixer(self.norm1(x), self.norm1(x), self.norm1(x), 
                                       attn_mask=nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device),
                                       is_causal=True)
                x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

    def freeze_high_frequencies(self, threshold):
        if hasattr(self.mixer, 'freeze_high_frequencies'):
            self.mixer.freeze_high_frequencies(threshold)
        for m in self.mlp:
            if hasattr(m, 'freeze_high_frequencies'):
                m.freeze_high_frequencies(threshold)

    def progressive_unfreeze(self, epoch, total, strategy):
        if hasattr(self.mixer, 'progressive_unfreeze'):
            self.mixer.progressive_unfreeze(epoch, total, strategy)
        for m in self.mlp:
            if hasattr(m, 'progressive_unfreeze_schedule'):
                m.progressive_unfreeze_schedule(epoch, total, strategy)

    def constrain_energy(self):
        if hasattr(self.mixer, 'constrain_energy'):
            self.mixer.constrain_energy()
        for m in self.mlp:
            if hasattr(m, 'constrain_energy'):
                m.constrain_energy()


class SpectralGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        if config.weight_type == 'wave':
            self.token_emb = SpectralEmbedding(config.vocab_size, config.d_model, 
                                             config.num_waves*2, config.num_harmonics+2,
                                             init_mode=getattr(config, 'init_mode', 'standard'))
            self.pos_emb = SpectralPositionalEncoding(config.d_model, config.block_size,
                                                    config.num_waves, config.num_harmonics,
                                                    init_mode=getattr(config, 'init_mode', 'standard'))
        else:
            self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
            self.pos_emb = nn.Embedding(config.block_size, config.d_model)
            
        self.dropout = nn.Dropout(config.dropout)
        
        # Blocks
        self.blocks = nn.ModuleList([SpectralBlock(config, layer_index=i) for i in range(config.num_layers)])
        
        # Head
        self.norm_f = nn.LayerNorm(config.d_model)
        if config.weight_type == 'wave':
            self.head = UserWaveLinear(config.d_model, config.vocab_size, config.num_waves, 
                                     config.num_harmonics, adaptive_freqs=True,
                                     init_mode=getattr(config, 'init_mode', 'standard'))
        else:
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx, targets=None, progress=1.0):
        B, T = idx.shape
        
        # Embeddings
        if self.config.weight_type == 'wave':
            tok = self.token_emb(idx)
            pos = self.pos_emb(torch.zeros_like(tok)) # Pos encoding handles position internally
        else:
            tok = self.token_emb(idx)
            pos = self.pos_emb(torch.arange(T, device=idx.device))
            
        x = self.dropout(tok + pos)
        
        # Blocks
        for block in self.blocks:
            x = block(x, progress)
            
        x = self.norm_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    def freeze_high_frequencies(self, threshold=0.2):
        if self.config.weight_type != 'wave': return
        self.token_emb.freeze_high_frequencies(threshold)
        self.pos_emb.freeze_high_frequencies(threshold)
        for block in self.blocks:
            block.freeze_high_frequencies(threshold)
        self.head.freeze_high_frequencies(threshold)

    def progressive_unfreeze(self, epoch, total, strategy='linear'):
        if self.config.weight_type != 'wave': return
        self.token_emb.progressive_unfreeze(epoch, total, strategy)
        self.pos_emb.progressive_unfreeze(epoch, total, strategy)
        for block in self.blocks:
            block.progressive_unfreeze(epoch, total, strategy)
        self.head.progressive_unfreeze_schedule(epoch, total, strategy)

    def constrain_energy(self):
        """Apply Hamiltonian energy conservation constraints."""
        if self.config.weight_type == 'wave':
            self.token_emb.constrain_energy()
            self.pos_emb.constrain_energy()
            self.head.constrain_energy()
        
        for block in self.blocks:
            block.constrain_energy()

    def stabilize_waves(self):
        """Stabilize all wave parameters to prevent explosion while preserving interference."""
        if self.config.weight_type == 'wave':
            if hasattr(self.token_emb, 'wave_generator'):
                self.token_emb.wave_generator.stabilize_waves()
            if hasattr(self.pos_emb, 'wave_encoder'):
                self.pos_emb.wave_encoder.stabilize_waves()
            if hasattr(self.head, 'stabilize_waves'):
                self.head.stabilize_waves()
        
        for block in self.blocks:
            # Stabilize mixer if it's wave-based
            if hasattr(block.mixer, 'q_proj'):  # SpectralAttention
                block.mixer.q_proj.stabilize_waves()
                block.mixer.k_proj.stabilize_waves()
                block.mixer.v_proj.stabilize_waves()
                block.mixer.o_proj.stabilize_waves()
            
            # Stabilize MLP layers
            for m in block.mlp:
                if hasattr(m, 'stabilize_waves'):
                    m.stabilize_waves()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond, progress=1.0)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
