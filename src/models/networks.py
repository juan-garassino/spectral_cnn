import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import UserWaveLinear, PolyLinear, WaveletLinear, FactorLinear, SirenLinear, GatedWaveLinear

class UniversalMLP(nn.Module):
    def __init__(self, layer_type, num_waves=12, num_harmonics=3, 
                 adaptive_freqs=False, per_neuron_coeffs=False, wave_mode="outer_product"):
        super().__init__()
        if layer_type == "UserWave": Layer = UserWaveLinear
        elif layer_type == "Poly":   Layer = PolyLinear
        elif layer_type == "Wavelet":Layer = WaveletLinear
        elif layer_type == "Factor": Layer = FactorLinear
        elif layer_type == "Siren":  Layer = SirenLinear
        elif layer_type == "GatedWave": Layer = GatedWaveLinear
        elif layer_type == "Standard":Layer = nn.Linear

        HIDDEN = 12
        if layer_type == "Standard":
            self.fc1 = nn.Linear(28*28, HIDDEN)
            self.fc2 = nn.Linear(HIDDEN, HIDDEN)
            self.fc3 = nn.Linear(HIDDEN, 10)
        elif layer_type in ["UserWave", "GatedWave"]:
            # Spectral layers with Fourier configuration
            self.fc1 = Layer(28*28, HIDDEN, num_waves=num_waves, num_harmonics=num_harmonics,
                           adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                           wave_mode=wave_mode)
            self.fc2 = Layer(HIDDEN, HIDDEN, num_waves=num_waves, num_harmonics=num_harmonics,
                           adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                           wave_mode=wave_mode)
            self.fc3 = Layer(HIDDEN, 10, num_waves=num_waves, num_harmonics=num_harmonics,
                           adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                           wave_mode=wave_mode)
        else:
            self.fc1 = Layer(28*28, HIDDEN, num_waves=num_waves)
            self.fc2 = Layer(HIDDEN, HIDDEN, num_waves=num_waves)
            self.fc3 = Layer(HIDDEN, 10, num_waves=num_waves)
        self.type = layer_type

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def constrain_all(self):
        if self.type in ["UserWave", "GatedWave"]:
            self.fc1.constrain(); self.fc2.constrain(); self.fc3.constrain()
    
    def get_l1_loss(self):
        """Returns total L1 penalty from all layers (for sparsity regularization)."""
        if self.type not in ["UserWave", "GatedWave"]:
            return 0.0
        return self.fc1.get_l1_loss() + self.fc2.get_l1_loss() + self.fc3.get_l1_loss()

    def get_first_layer_weight(self):
        if self.type == "Standard": return self.fc1.weight
        return self.fc1.get_weight()
