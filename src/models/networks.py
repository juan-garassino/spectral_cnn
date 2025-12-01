import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import UserWaveLinear, PolyLinear, WaveletLinear, FactorLinear, SirenLinear, GatedWaveLinear

class UniversalMLP(nn.Module):
    def __init__(self, layer_type, num_waves=12):
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

    def get_first_layer_weight(self):
        if self.type == "Standard": return self.fc1.weight
        return self.fc1.get_weight()
