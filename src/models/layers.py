import torch
import torch.nn as nn
import numpy as np

# 1. USER WAVE
class UserWaveLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_waves=12, num_harmonics=3, 
                 adaptive_freqs=False, per_neuron_coeffs=False):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            num_waves: Number of waves to superpose
            num_harmonics: Number of Fourier components per wave (default: 3)
            adaptive_freqs: If True, harmonic frequencies are learnable (default: False, uses [1, 2, 4, 8, ...])
            per_neuron_coeffs: If True, each output neuron has its own coefficients (default: False, shared)
        """
        super().__init__()
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        self.adaptive_freqs = adaptive_freqs
        self.per_neuron_coeffs = per_neuron_coeffs
        
        rank = 1
        self.u = nn.Parameter(torch.randn(num_waves, out_dim, rank) * 1.0)
        self.v = nn.Parameter(torch.randn(num_waves, in_dim, rank) * 1.0)
        
        # Base frequency per wave
        init_freqs = torch.tensor([1.5**i for i in range(num_waves)]).float()
        self.freqs = nn.Parameter(init_freqs.view(num_waves, 1, 1))
        
        # Harmonic multipliers (1×, 2×, 4×, 8×, ...)
        if adaptive_freqs:
            # Learnable harmonic frequencies
            init_harm_freqs = torch.tensor([2.0**i for i in range(num_harmonics)]).float()
            self.harmonic_freqs = nn.Parameter(init_harm_freqs)
        else:
            # Fixed harmonic frequencies [1, 2, 4, 8, ...]
            self.register_buffer('harmonic_freqs', torch.tensor([2.0**i for i in range(num_harmonics)]).float())
        
        # Fourier coefficients
        if per_neuron_coeffs:
            # Each output neuron has its own coefficients [num_waves, out_dim, num_harmonics]
            init_coeffs = torch.randn(num_waves, out_dim, num_harmonics) * 0.1
            # Bias towards decreasing amplitude with frequency
            for h in range(num_harmonics):
                init_coeffs[:, :, h] += (0.5 ** h)
            self.fourier_coeffs = nn.Parameter(init_coeffs)
        else:
            # Shared coefficients across neurons [num_waves, num_harmonics]
            init_coeffs = torch.randn(num_waves, num_harmonics) * 0.1
            for h in range(num_harmonics):
                init_coeffs[:, h] += (0.5 ** h)
            self.fourier_coeffs = nn.Parameter(init_coeffs)
        
        # Overall amplitude per wave
        self.amplitudes = nn.Parameter(torch.randn(num_waves) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        theta_base = torch.bmm(self.u, self.v.transpose(1, 2))
        theta = theta_base * self.freqs
        W = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        
        for i in range(self.num_waves):
            wave = torch.zeros_like(theta[i])
            for h in range(self.num_harmonics):
                harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                if self.per_neuron_coeffs:
                    # Different coefficients per output neuron
                    wave = wave + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                else:
                    # Shared coefficients
                    wave = wave + self.fourier_coeffs[i, h] * harmonic
            W = W + self.amplitudes[i] * wave
        return x @ W.t() + self.bias

    def get_weight(self):
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            W = torch.zeros(self.u.shape[1], self.v.shape[1], device=self.u.device)
            for i in range(self.num_waves):
                wave = torch.zeros_like(theta[i])
                for h in range(self.num_harmonics):
                    harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                    if self.per_neuron_coeffs:
                        wave = wave + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                    else:
                        wave = wave + self.fourier_coeffs[i, h] * harmonic
                W = W + self.amplitudes[i] * wave
        return W
    
    def get_waves(self):
        """Returns the individual wave components for visualization."""
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            waves = []
            for i in range(self.num_waves):
                wave = torch.zeros_like(theta[i])
                for h in range(self.num_harmonics):
                    harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                    if self.per_neuron_coeffs:
                        wave = wave + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                    else:
                        wave = wave + self.fourier_coeffs[i, h] * harmonic
                waves.append(wave * self.amplitudes[i])
            return torch.stack(waves)
    
    def get_wave_components(self):
        """Returns the individual Fourier components for each wave."""
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            components = []
            for i in range(self.num_waves):
                comp_dict = {}
                for h in range(self.num_harmonics):
                    harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                    if self.per_neuron_coeffs:
                        comp = self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic * self.amplitudes[i]
                    else:
                        comp = self.fourier_coeffs[i, h] * harmonic * self.amplitudes[i]
                    comp_dict[f'comp{h+1}'] = comp
                
                comp_dict['theta'] = theta[i]
                comp_dict['harmonic_freqs'] = self.harmonic_freqs.clone()
                if self.per_neuron_coeffs:
                    comp_dict['coeffs'] = self.fourier_coeffs[i, 0, :].clone()  # Show neuron 0's coeffs
                else:
                    comp_dict['coeffs'] = self.fourier_coeffs[i, :].clone()
                components.append(comp_dict)
            return components
    
    def get_l1_loss(self):
        """Returns L1 penalty on Fourier coefficients for sparsity."""
        return torch.abs(self.fourier_coeffs).sum()

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
            nn.Linear(input_dim, hidden_dim), Sine(w0=30.0),
            nn.Linear(hidden_dim, hidden_dim), Sine(w0=30.0),
            nn.Linear(hidden_dim, 1)
        )
        
        # SIREN Initialization
        first = True
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                if first:
                    nn.init.uniform_(m.weight, -1 / num_input, 1 / num_input)
                    first = False
                else:
                    nn.init.uniform_(m.weight, -np.sqrt(6 / num_input) / 30.0, np.sqrt(6 / num_input) / 30.0)
                
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, out_dim), torch.linspace(-1, 1, in_dim), indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)
        # Keep Fourier features as they help high frequency details even in SIRENs sometimes, 
        # but pure SIREN usually takes raw coords. 
        # Let's keep the embedding to minimize structural changes but use Sine activations.
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
        self.signal_waves = 10
        self.num_harmonics = 3
        rank = 1

        # Signal Init
        self.u = nn.Parameter(torch.randn(self.signal_waves, out_dim, rank) * 1.0)
        self.v = nn.Parameter(torch.randn(self.signal_waves, in_dim, rank) * 1.0)
        init_freqs = torch.tensor([1.5**i for i in range(self.signal_waves)]).float()
        self.freqs = nn.Parameter(init_freqs.view(self.signal_waves, 1, 1))

        # LEARNABLE Fourier coefficients
        init_coeffs = torch.tensor([[1.0, 0.5, 0.25] for _ in range(self.signal_waves)]).float()
        self.fourier_coeffs = nn.Parameter(init_coeffs * (torch.randn(self.signal_waves, self.num_harmonics) * 0.1 + 1.0))
        
        self.amplitudes = nn.Parameter(torch.randn(self.signal_waves) * 0.1)

        # Gate Init (Rank 2)
        self.gate_rank = 2
        self.u_gate = nn.Parameter(torch.randn(out_dim, self.gate_rank) * 0.1)
        self.v_gate = nn.Parameter(torch.randn(in_dim, self.gate_rank) * 0.1)
        self.gate_bias = nn.Parameter(torch.ones(1) * 2.0)

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
        signal = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        for i in range(self.signal_waves):
            w = (self.fourier_coeffs[i, 0] * torch.cos(theta[i]) + 
                self.fourier_coeffs[i, 1] * torch.cos(2*theta[i]) + 
                self.fourier_coeffs[i, 2] * torch.cos(4*theta[i]))
            signal = signal + self.amplitudes[i] * w

        gate = torch.sigmoid((self.u_gate @ self.v_gate.t()) + self.gate_bias)
        W = signal * gate
        return x @ W.t() + self.bias

    def get_weight(self):
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            signal = torch.zeros(self.u.shape[1], self.v.shape[1], device=self.u.device)
            for i in range(self.signal_waves):
                w = (self.fourier_coeffs[i, 0] * torch.cos(theta[i]) + 
                    self.fourier_coeffs[i, 1] * torch.cos(2*theta[i]) + 
                    self.fourier_coeffs[i, 2] * torch.cos(4*theta[i]))
                signal = signal + self.amplitudes[i] * w
            gate = torch.sigmoid((self.u_gate @ self.v_gate.t()) + self.gate_bias)
            W = signal * gate
        return W
    
    def get_waves(self):
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            waves = []
            for i in range(self.signal_waves):
                w = (self.fourier_coeffs[i, 0] * torch.cos(theta[i]) + 
                    self.fourier_coeffs[i, 1] * torch.cos(2*theta[i]) + 
                    self.fourier_coeffs[i, 2] * torch.cos(4*theta[i]))
                waves.append(w * self.amplitudes[i])
            return torch.stack(waves)
    
    def get_wave_components(self):
        """Returns the individual Fourier components for each wave."""
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            components = []
            for i in range(self.signal_waves):
                comp1 = self.fourier_coeffs[i, 0] * torch.cos(theta[i]) * self.amplitudes[i]
                comp2 = self.fourier_coeffs[i, 1] * torch.cos(2*theta[i]) * self.amplitudes[i]
                comp3 = self.fourier_coeffs[i, 2] * torch.cos(4*theta[i]) * self.amplitudes[i]
                components.append({
                    'comp1': comp1, 
                    'comp2': comp2, 
                    'comp3': comp3, 
                    'theta': theta[i],
                    'coeffs': self.fourier_coeffs[i].clone()
                })
            return components

    def constrain(self): pass

