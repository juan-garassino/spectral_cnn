import torch
import torch.nn as nn
import numpy as np

# 1. USER WAVE
class UserWaveLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_waves=12, num_harmonics=3, 
                 adaptive_freqs=False, per_neuron_coeffs=False, wave_mode="outer_product"):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            num_waves: Number of waves to superpose
            num_harmonics: Number of Fourier components per wave (default: 3)
            adaptive_freqs: If True, harmonic frequencies are learnable (default: False, uses [1, 2, 4, 8, ...])
            per_neuron_coeffs: If True, each output neuron has its own coefficients (default: False, shared)
            wave_mode: "outer_product" (2D patterns) or "fourier_series" (1D smooth sinusoids)
        """
        super().__init__()
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        self.adaptive_freqs = adaptive_freqs
        self.per_neuron_coeffs = per_neuron_coeffs
        self.wave_mode = wave_mode
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if wave_mode == "outer_product":
            # Original 2D outer product approach
            rank = 1
            self.u = nn.Parameter(torch.randn(num_waves, out_dim, rank) * 1.0)
            self.v = nn.Parameter(torch.randn(num_waves, in_dim, rank) * 1.0)
            
            # Base frequency per wave
            init_freqs = torch.tensor([1.5**i for i in range(num_waves)]).float()
            self.freqs = nn.Parameter(init_freqs.view(num_waves, 1, 1))
        
        elif wave_mode == "fourier_series":
            # 1D Fourier series approach - smooth sinusoidal waves
            # Base frequencies for each wave
            init_freqs = torch.tensor([float(i+1) for i in range(num_waves)]).float()
            self.freqs = nn.Parameter(init_freqs)
            
            # Learnable phase shifts for each wave
            self.phases = nn.Parameter(torch.randn(num_waves) * 0.1)
        
        # Harmonic multipliers (1×, 2×, 4×, 8×, ...)
        if adaptive_freqs:
            init_harm_freqs = torch.tensor([2.0**i for i in range(num_harmonics)]).float()
            self.harmonic_freqs = nn.Parameter(init_harm_freqs)
        else:
            self.register_buffer('harmonic_freqs', torch.tensor([2.0**i for i in range(num_harmonics)]).float())
        
        # Fourier coefficients
        if per_neuron_coeffs:
            init_coeffs = torch.randn(num_waves, out_dim, num_harmonics) * 0.1
            for h in range(num_harmonics):
                init_coeffs[:, :, h] += (0.5 ** h)
            self.fourier_coeffs = nn.Parameter(init_coeffs)
        else:
            init_coeffs = torch.randn(num_waves, num_harmonics) * 0.1
            for h in range(num_harmonics):
                init_coeffs[:, h] += (0.5 ** h)
            self.fourier_coeffs = nn.Parameter(init_coeffs)
        
        # Overall amplitude per wave (in fourier_series mode, this is per output neuron per wave)
        if wave_mode == "fourier_series":
            self.amplitudes = nn.Parameter(torch.randn(out_dim, num_waves) * 0.1)
        else:
            self.amplitudes = nn.Parameter(torch.randn(num_waves) * 0.1)
        
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        if self.wave_mode == "outer_product":
            return self._forward_outer_product(x)
        else:
            return self._forward_fourier_series(x)
    
    def _forward_outer_product(self, x):
        """Original 2D outer product approach"""
        theta_base = torch.bmm(self.u, self.v.transpose(1, 2))
        theta = theta_base * self.freqs
        W = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        
        for i in range(self.num_waves):
            wave = torch.zeros_like(theta[i])
            for h in range(self.num_harmonics):
                harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                if self.per_neuron_coeffs:
                    wave = wave + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                else:
                    wave = wave + self.fourier_coeffs[i, h] * harmonic
            W = W + self.amplitudes[i] * wave
        return x @ W.t() + self.bias
    
    def _forward_fourier_series(self, x):
        """1D Fourier series approach - smooth sinusoidal basis functions"""
        # Create position indices for input dimension
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=x.device) / self.in_dim
        
        # Build weight matrix [out_dim, in_dim]
        W = torch.zeros(self.out_dim, self.in_dim, device=x.device)
        
        for wave_idx in range(self.num_waves):
            # Create 1D wave along input dimension
            wave_1d = torch.zeros(self.in_dim, device=x.device)
            
            for h in range(self.num_harmonics):
                # Smooth sinusoidal component
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                
                if self.per_neuron_coeffs:
                    # Each output neuron has different coefficients
                    for out_idx in range(self.out_dim):
                        harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                        W[out_idx, :] += (self.fourier_coeffs[wave_idx, out_idx, h] * 
                                         self.amplitudes[out_idx, wave_idx] * harmonic)
                else:
                    # Shared coefficients
                    harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                    wave_1d += self.fourier_coeffs[wave_idx, h] * harmonic
            
            if not self.per_neuron_coeffs:
                # Add wave to all output neurons with learned amplitudes
                for out_idx in range(self.out_dim):
                    W[out_idx, :] += self.amplitudes[out_idx, wave_idx] * wave_1d
        
        return x @ W.t() + self.bias

    def get_weight(self):
        with torch.no_grad():
            if self.wave_mode == "outer_product":
                return self._get_weight_outer_product()
            else:
                return self._get_weight_fourier_series()
    
    def _get_weight_outer_product(self):
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
    
    def _get_weight_fourier_series(self):
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=self.freqs.device) / self.in_dim
        W = torch.zeros(self.out_dim, self.in_dim, device=self.freqs.device)
        
        for wave_idx in range(self.num_waves):
            wave_1d = torch.zeros(self.in_dim, device=self.freqs.device)
            for h in range(self.num_harmonics):
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                if self.per_neuron_coeffs:
                    for out_idx in range(self.out_dim):
                        harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                        W[out_idx, :] += (self.fourier_coeffs[wave_idx, out_idx, h] * 
                                         self.amplitudes[out_idx, wave_idx] * harmonic)
                else:
                    harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                    wave_1d += self.fourier_coeffs[wave_idx, h] * harmonic
            
            if not self.per_neuron_coeffs:
                for out_idx in range(self.out_dim):
                    W[out_idx, :] += self.amplitudes[out_idx, wave_idx] * wave_1d
        return W
    
    def get_waves(self):
        """Returns the individual wave components for visualization."""
        with torch.no_grad():
            if self.wave_mode == "outer_product":
                return self._get_waves_outer_product()
            else:
                return self._get_waves_fourier_series()
    
    def _get_waves_outer_product(self):
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
    
    def _get_waves_fourier_series(self):
        """Get 1D Fourier series waves for visualization"""
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=self.freqs.device) / self.in_dim
        waves = []
        
        for wave_idx in range(self.num_waves):
            # Build complete wave for first output neuron
            wave_1d = torch.zeros(self.in_dim, device=self.freqs.device)
            for h in range(self.num_harmonics):
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                if self.per_neuron_coeffs:
                    wave_1d += self.fourier_coeffs[wave_idx, 0, h] * harmonic
                else:
                    wave_1d += self.fourier_coeffs[wave_idx, h] * harmonic
            
            # Broadcast to match expected shape [out_dim, in_dim]
            wave_2d = wave_1d.unsqueeze(0).expand(self.out_dim, -1)
            waves.append(wave_2d * (self.amplitudes[0, wave_idx] if self.wave_mode =="fourier_series" else self.amplitudes[wave_idx]))
        
        return torch.stack(waves)
    
    def get_wave_components(self):
        """Returns the individual Fourier components for each wave."""
        with torch.no_grad():
            if self.wave_mode == "outer_product":
                return self._get_components_outer_product()
            else:
                return self._get_components_fourier_series()
    
    def _get_components_outer_product(self):
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
                comp_dict['coeffs'] = self.fourier_coeffs[i, 0, :].clone()
            else:
                comp_dict['coeffs'] = self.fourier_coeffs[i, :].clone()
            components.append(comp_dict)
        return components
    
    def _get_components_fourier_series(self):
        """Get Fourier series components for visualization"""
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=self.freqs.device) / self.in_dim
        components = []
        
        for wave_idx in range(self.num_waves):
            comp_dict = {}
            for h in range(self.num_harmonics):
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                
                if self.per_neuron_coeffs:
                    # Use first output neuron's coefficients for visualization
                    comp = self.fourier_coeffs[wave_idx, 0, h] * harmonic * self.amplitudes[0, wave_idx]
                else:
                    comp = self.fourier_coeffs[wave_idx, h] * harmonic
                    # Scale by amplitude for first neuron
                    comp = comp * self.amplitudes[0, wave_idx]
                
                # Broadcast to match expected shape [out_dim, in_dim]
                comp_dict[f'comp{h+1}'] = comp.unsqueeze(0).expand(self.out_dim, -1)
            
            comp_dict['theta'] = None  # No theta in Fourier series mode
            comp_dict['harmonic_freqs'] = self.harmonic_freqs.clone()
            if self.per_neuron_coeffs:
                comp_dict['coeffs'] = self.fourier_coeffs[wave_idx, 0, :].clone()
            else:
                comp_dict['coeffs'] = self.fourier_coeffs[wave_idx, :].clone()
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
    def __init__(self, in_dim, out_dim, num_waves=12, num_harmonics=3, 
                 adaptive_freqs=False, per_neuron_coeffs=False, wave_mode="outer_product"):
        super().__init__()
        self.signal_waves = 10
        self.num_harmonics = num_harmonics
        self.adaptive_freqs = adaptive_freqs
        self.per_neuron_coeffs = per_neuron_coeffs
        self.wave_mode = wave_mode  # Store but currently only outer_product supported for GatedWave
        rank = 1

        # Signal Init
        self.u = nn.Parameter(torch.randn(self.signal_waves, out_dim, rank) * 1.0)
        self.v = nn.Parameter(torch.randn(self.signal_waves, in_dim, rank) * 1.0)
        init_freqs = torch.tensor([1.5**i for i in range(self.signal_waves)]).float()
        self.freqs = nn.Parameter(init_freqs.view(self.signal_waves, 1, 1))

        # Harmonic multipliers (1×, 2×, 4×, 8×, ...)
        if adaptive_freqs:
            init_harm_freqs = torch.tensor([2.0**i for i in range(num_harmonics)]).float()
            self.harmonic_freqs = nn.Parameter(init_harm_freqs)
        else:
            self.register_buffer('harmonic_freqs', torch.tensor([2.0**i for i in range(num_harmonics)]).float())

        # Fourier coefficients
        if per_neuron_coeffs:
            init_coeffs = torch.randn(self.signal_waves, out_dim, num_harmonics) * 0.1
            for h in range(num_harmonics):
                init_coeffs[:, :, h] += (0.5 ** h)
            self.fourier_coeffs = nn.Parameter(init_coeffs)
        else:
            init_coeffs = torch.randn(self.signal_waves, num_harmonics) * 0.1
            for h in range(num_harmonics):
                init_coeffs[:, h] += (0.5 ** h)
            self.fourier_coeffs = nn.Parameter(init_coeffs)
        
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
            w = torch.zeros_like(theta[i])
            for h in range(self.num_harmonics):
                harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                if self.per_neuron_coeffs:
                    w = w + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                else:
                    w = w + self.fourier_coeffs[i, h] * harmonic
            signal = signal + self.amplitudes[i] * w

        gate = torch.sigmoid((self.u_gate @ self.v_gate.t()) + self.gate_bias)
        W = signal * gate
        return x @ W.t() + self.bias

    def get_weight(self):
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            signal = torch.zeros(self.u.shape[1], self.v.shape[1], device=self.u.device)
            for i in range(self.signal_waves):
                w = torch.zeros_like(theta[i])
                for h in range(self.num_harmonics):
                    harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                    if self.per_neuron_coeffs:
                        w = w + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                    else:
                        w = w + self.fourier_coeffs[i, h] * harmonic
                signal = signal + self.amplitudes[i] * w
            gate = torch.sigmoid((self.u_gate @ self.v_gate.t()) + self.gate_bias)
            W = signal * gate
        return W
    
    def get_waves(self):
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            waves = []
            for i in range(self.signal_waves):
                w = torch.zeros_like(theta[i])
                for h in range(self.num_harmonics):
                    harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                    if self.per_neuron_coeffs:
                        w = w + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                    else:
                        w = w + self.fourier_coeffs[i, h] * harmonic
                waves.append(w * self.amplitudes[i])
            return torch.stack(waves)
    
    def get_wave_components(self):
        """Returns the individual Fourier components for each wave."""
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            components = []
            for i in range(self.signal_waves):
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
                    comp_dict['coeffs'] = self.fourier_coeffs[i, 0, :].clone()
                else:
                    comp_dict['coeffs'] = self.fourier_coeffs[i, :].clone()
                components.append(comp_dict)
            return components
    
    def get_l1_loss(self):
        return torch.abs(self.fourier_coeffs).sum()

    def constrain(self): pass

