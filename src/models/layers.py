import torch
import torch.nn as nn
import numpy as np

# 1. USER WAVE
class UserWaveLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_waves=12):
        super().__init__()
        self.num_waves = num_waves
        rank = 1
        self.u = nn.Parameter(torch.randn(num_waves, out_dim, rank) * 1.0)
        self.v = nn.Parameter(torch.randn(num_waves, in_dim, rank) * 1.0)
        init_freqs = torch.tensor([1.5**i for i in range(num_waves)]).float()
        self.freqs = nn.Parameter(init_freqs.view(num_waves, 1, 1))
        self.amplitudes = nn.Parameter(torch.randn(num_waves) * 0.1) # Boosted
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        theta_base = torch.bmm(self.u, self.v.transpose(1, 2))
        theta = theta_base * self.freqs
        W = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        for i in range(self.num_waves):
            wave = torch.cos(theta[i]) + 0.5 * torch.cos(2.0 * theta[i]) + 0.25 * torch.cos(4.0 * theta[i])
            W = W + self.amplitudes[i] * wave
        return x @ W.t() + self.bias

    def get_weight(self):
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            W = torch.zeros(self.u.shape[1], self.v.shape[1], device=self.u.device)
            for i in range(self.num_waves):
                wave = torch.cos(theta[i]) + 0.5*torch.cos(2*theta[i]) + 0.25*torch.cos(4*theta[i])
                W = W + self.amplitudes[i] * wave
        return W
    
    def get_waves(self):
        """Returns the individual wave components for visualization."""
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            waves = []
            for i in range(self.num_waves):
                wave = torch.cos(theta[i]) + 0.5*torch.cos(2*theta[i]) + 0.25*torch.cos(4*theta[i])
                waves.append(wave * self.amplitudes[i])
            return torch.stack(waves)

    def constrain(self): pass

# 2. POLY NET
class PolyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_waves=12):
        super().__init__()
        self.num_waves = num_waves
        rank = 1
        self.u = nn.Parameter(torch.randn(num_waves, out_dim, rank) * 1.5)
        self.v = nn.Parameter(torch.randn(num_waves, in_dim, rank) * 1.5)
        self.coeffs = nn.Parameter(torch.randn(num_waves) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        theta = torch.bmm(self.u, self.v.transpose(1, 2))
        t = torch.tanh(theta)
        W = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        for i in range(self.num_waves):
            W = W + (self.coeffs[i] * torch.pow(t[i], i+1))
        return x @ W.t() + self.bias
    def get_weight(self):
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2))
            t = torch.tanh(theta)
            W = torch.zeros_like(theta[0])
            for i in range(self.num_waves):
                W = W + (self.coeffs[i] * torch.pow(t[i], i+1))
        return W
    def constrain(self): pass

# 3. WAVELET NET
class WaveletLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_waves=12):
        super().__init__()
        self.num_waves = num_waves
        rank = 1
        self.u = nn.Parameter(torch.randn(num_waves, out_dim, rank) * 2.0)
        self.v = nn.Parameter(torch.randn(num_waves, in_dim, rank) * 2.0)
        self.amplitudes = nn.Parameter(torch.randn(num_waves) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        theta = torch.bmm(self.u, self.v.transpose(1, 2))
        W = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        for i in range(self.num_waves):
            t = theta[i]
            envelope = torch.exp(-torch.pow(t, 2))
            oscillation = torch.cos(5.0 * t)
            W = W + (self.amplitudes[i] * envelope * oscillation)
        return x @ W.t() + self.bias
    def get_weight(self):
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2))
            W = torch.zeros_like(theta[0])
            for i in range(self.num_waves):
                t = theta[i]
                W = W + (self.amplitudes[i] * torch.exp(-torch.pow(t, 2)) * torch.cos(5.0*t))
        return W
    def constrain(self): pass

# 4. FACTOR NET
class FactorLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_waves=12):
        super().__init__()
        limit = min(in_dim, out_dim)
        target = max(2, limit // 2)
        self.rank = target
        k = 1.0 / np.sqrt(in_dim)
        self.U = nn.Parameter(torch.randn(out_dim, self.rank) * k)
        self.V = nn.Parameter(torch.randn(in_dim, self.rank) * k)
        self.bias = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        W = self.U @ self.V.t()
        return x @ W.t() + self.bias
    def get_weight(self): return self.U @ self.V.t()
    def constrain(self): pass

# 5. SIREN NET
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, input):
        return torch.sin(self.w0 * input)
class SirenLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_waves=12):
        super().__init__()
        self.rows = out_dim
        self.cols = in_dim
        hidden_dim = 64
        self.L = 4
        input_dim = 2 + (2 * 2 * self.L)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, out_dim), torch.linspace(-1, 1, in_dim), indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)
        embeds = [coords]
        for i in range(self.L):
            freq = 2.0**i
            embeds.append(torch.sin(freq * torch.pi * coords))
            embeds.append(torch.cos(freq * torch.pi * coords))
        self.register_buffer('embedded_coords', torch.cat(embeds, dim=-1))
        self.bias = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        W_flat = self.net(self.embedded_coords)
        W = W_flat.view(self.rows, self.cols)
        return x @ W.t() + self.bias
    def get_weight(self):
        return self.net(self.embedded_coords).view(self.rows, self.cols)
    def constrain(self): pass

# 6. GATED WAVE NET (V14 Optimized)
class GatedWaveLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_waves=12):
        super().__init__()
        # V14 BUDGET MATH:
        # Target: ~9420 params per layer
        # Signal: 10 Waves x (784+12+2) = 7980
        # Gate: Rank 2 x (784+12) = 1592
        # Total: 9572. (Matches Standard's 9420 + overhead).

        self.signal_waves = 10
        rank = 1

        # Signal Init
        self.u = nn.Parameter(torch.randn(self.signal_waves, out_dim, rank) * 1.0)
        self.v = nn.Parameter(torch.randn(self.signal_waves, in_dim, rank) * 1.0)
        init_freqs = torch.tensor([1.5**i for i in range(self.signal_waves)]).float()
        self.freqs = nn.Parameter(init_freqs.view(self.signal_waves, 1, 1))

        # BOOSTED INIT: 0.1 allows signal to flow through the gate immediately
        self.amplitudes = nn.Parameter(torch.randn(self.signal_waves) * 0.1)

        # Gate Init (Rank 2)
        self.gate_rank = 2
        self.u_gate = nn.Parameter(torch.randn(out_dim, self.gate_rank) * 0.1)
        self.v_gate = nn.Parameter(torch.randn(in_dim, self.gate_rank) * 0.1)

        # Bias +2.0 keeps gate open at start
        self.gate_bias = nn.Parameter(torch.ones(1) * 2.0)

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
        signal = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        for i in range(self.signal_waves):
            w = torch.cos(theta[i]) + 0.5*torch.cos(2*theta[i]) + 0.25*torch.cos(4*theta[i])
            signal = signal + self.amplitudes[i] * w

        gate = torch.sigmoid((self.u_gate @ self.v_gate.t()) + self.gate_bias)

        W = signal * gate
        return x @ W.t() + self.bias

    def get_weight(self):
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            signal = torch.zeros(self.u.shape[1], self.v.shape[1], device=self.u.device)
            for i in range(self.signal_waves):
                w = torch.cos(theta[i]) + 0.5*torch.cos(2*theta[i]) + 0.25*torch.cos(4*theta[i])
                signal = signal + self.amplitudes[i] * w
            gate = torch.sigmoid((self.u_gate @ self.v_gate.t()) + self.gate_bias)
            W = signal * gate
        return W
    
    def get_waves(self):
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            waves = []
            for i in range(self.signal_waves):
                w = torch.cos(theta[i]) + 0.5*torch.cos(2*theta[i]) + 0.25*torch.cos(4*theta[i])
                waves.append(w * self.amplitudes[i])
            return torch.stack(waves)

    def constrain(self): pass
