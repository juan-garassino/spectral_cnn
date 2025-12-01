# @title WEIGHT PARAMETERIZATION BENCHMARK: THE 5 FITS (V14 - THE OPTIMIZED DUEL)


"""
WEIGHT PARAMETERIZATION BENCHMARK: THE 5 FITS (V14 - THE OPTIMIZED DUEL)
=============================================================================
A unified laboratory comparing 7 different mathematical paradigms on MNIST.

CHANGELOG V14 (The "Precision" Update):
-----------------------------------------------------------------------------
1. EXACT PARAMETER MATH:
   - Fixed the budget calculation to match Standard Dense Layer (9,420 params).
   - Configuration: 10 Waves + Rank 2 Gate.
   - Total Params: ~9,572 (vs Standard 9,706 Total).

2. INITIALIZATION OVERHAUL (The "Spotlight" Fix):
   - Amplitudes: Increased from 0.02 -> 0.1 (Matches Kaiming Variance).
   - Gate: Initialized to be semi-transparent but structured, preventing
     the "Multiplicative Chaos" of V13.

3. GOAL: Beat 93% accuracy with <10k parameters.
=============================================================================
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = f"./WeightParam_Benchmark_{timestamp}/"
os.makedirs(base_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 3
BATCH_SIZE = 64
NUM_WAVES = 12

# Decoupled LRs - Tuned for Wave convergence
LR_PHASE = 0.0050 # Faster frequency tuning
LR_AMP   = 0.0020 # Stronger signal growth
LR_BIAS  = 0.0010

# ==================================================
# PART 1: THE 7 UNIVERSAL LAYERS
# ==================================================

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
    def constrain(self): pass

# --- Model Wrapper ---
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

# ==================================================
# PART 2: TRAINING & VISUALIZATION
# ==================================================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=transform), batch_size=256, shuffle=False)

def get_optimizer(model, mode):
    if mode == "Standard": return torch.optim.Adam(model.parameters(), lr=LR_BIAS)
    phase, amp, bias = [], [], []
    for name, p in model.named_parameters():
        if 'bias' in name: bias.append(p)
        elif any(k in name for k in ['u', 'v', 'U', 'V', 'net']): phase.append(p)
        elif any(k in name for k in ['amp', 'coeff']): amp.append(p)
        else: phase.append(p)
    return torch.optim.Adam([{'params':phase,'lr':LR_PHASE},{'params':amp,'lr':LR_AMP},{'params':bias,'lr':LR_BIAS}])

def train_fit(mode_name, num_waves):
    print(f"\n--- Fitting: {mode_name} ---")
    model = UniversalMLP(mode_name, num_waves).to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    opt = get_optimizer(model, mode_name)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)

    start = time.time()
    history = {'train_acc': [], 'test_acc': []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        correct_train = 0
        total_train = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if isinstance(model, nn.DataParallel): model.module.constrain_all()
            else: model.constrain_all()

            correct_train += out.argmax(1).eq(y).sum().item()
            total_train += y.size(0)

        train_acc = 100.0 * correct_train / total_train
        history['train_acc'].append(train_acc)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += model(x.to(device)).argmax(1).eq(y.to(device)).sum().item()
        acc = 100.0 * correct / len(test_loader.dataset)
        history['test_acc'].append(acc)

        sched.step(acc)
        if epoch % 5 == 0 or epoch == NUM_EPOCHS-1:
            print(f"Ep {epoch+1:3d} | Train: {train_acc:.2f}% | Test: {acc:.2f}%")

    calc_model = model.module if isinstance(model, nn.DataParallel) else model
    return {
        'test_acc': acc,
        'train_acc': train_acc,
        'params': sum(p.numel() for p in calc_model.parameters() if p.requires_grad),
        'model': calc_model,
        'history': history
    }

# ==================================================
# PART 3: EXECUTION & VISUALIZATION
# ==================================================
modes = ["Standard", "UserWave", "Poly", "Wavelet", "Factor", "Siren", "GatedWave"]
results = {}

print(f"STARTING 5-FIT BENCHMARK V14 (The Optimized Duel)")
print(f"Target Baseline: ~9,700 params")

for m in modes: results[m] = train_fit(m, NUM_WAVES)

print("\n" + "="*125)
print(f"FINAL LEADERBOARD ({NUM_EPOCHS} Epochs) - V14 RESULTS")
print("="*125)
print(f"| {'Model':<15} | {'Params':<10} | {'Comp Ratio':<10} | {'Train %':<8} | {'Test %':<8} | {'Gen Gap':<8} | {'Eff. Score':<10} |")
print("-" * 125)

baseline_params = results["Standard"]["params"]
for m in modes:
    r = results[m]
    comp_ratio = baseline_params / r['params'] if r['params'] > 0 else 0
    gen_gap = r['train_acc'] - r['test_acc']
    eff_score = (r['test_acc'] / np.log10(r['params'])) if r['params'] > 0 else 0
    print(f"| {m:<15} | {r['params']:<10,} | {comp_ratio:<9.1f}x | {r['train_acc']:<8.2f} | {r['test_acc']:<8.2f} | {gen_gap:<8.2f} | {eff_score:<10.2f} |")
print("-" * 125)

# --- VISUALIZATION ---
print("\n[Generating Plots...]")
N = len(modes)

# Heatmaps
plt.figure(figsize=(3*N, 3))
for i, m in enumerate(modes):
    model = results[m]['model']
    W = model.get_first_layer_weight().detach().cpu().numpy()
    W_neuron0 = W[0].reshape(28, 28)

    plt.subplot(1, N, i+1)
    plt.imshow(W_neuron0, cmap='viridis')
    plt.title(f"{m}\n{results[m]['test_acc']:.1f}%")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"{base_dir}/weight_patterns.png")

# FFT
plt.figure(figsize=(3*N, 3))
for i, m in enumerate(modes):
    model = results[m]['model']
    W = model.get_first_layer_weight().detach().cpu().numpy()
    W_neuron0 = W[0].reshape(28, 28)
    f_transform = np.fft.fft2(W_neuron0)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)

    plt.subplot(1, N, i+1)
    plt.imshow(magnitude_spectrum, cmap='inferno')
    plt.title(f"{m} FFT")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"{base_dir}/frequency_analysis.png")

# Loss Curves
plt.figure(figsize=(10, 6))
for m in modes:
    hist = results[m]['history']
    plt.plot(hist['test_acc'], label=f"{m} (Test)", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title(f"Learning Curves ({NUM_EPOCHS} Epochs)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{base_dir}/training_dynamics.png")

print("All plots saved.")