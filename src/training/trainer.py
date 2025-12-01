import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
from src.models.networks import UniversalMLP

# Configuration
BATCH_SIZE = 64
LR_PHASE = 0.0050
LR_AMP   = 0.0020
LR_BIAS  = 0.0010

def get_optimizer(model, mode):
    if mode == "Standard": return torch.optim.Adam(model.parameters(), lr=LR_BIAS)
    phase, amp, bias = [], [], []
    for name, p in model.named_parameters():
        if 'bias' in name: bias.append(p)
        elif any(k in name for k in ['u', 'v', 'U', 'V', 'net']): phase.append(p)
        elif any(k in name for k in ['amp', 'coeff', 'fourier_coeffs']): amp.append(p)  # Fourier coefficients get amp LR
        else: phase.append(p)
    return torch.optim.Adam([{'params':phase,'lr':LR_PHASE},{'params':amp,'lr':LR_AMP},{'params':bias,'lr':LR_BIAS}])

def train_fit(mode_name, num_waves, num_epochs, device):
    print(f"\n--- Fitting: {mode_name} ---")
    
    # Data Loading (Ideally this should be passed in or cached, but keeping it simple for now)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=transform), batch_size=256, shuffle=False)

    model = UniversalMLP(mode_name, num_waves).to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    opt = get_optimizer(model, mode_name)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)

    start = time.time()
    history = {'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
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
        if epoch % 5 == 0 or epoch == num_epochs-1:
            print(f"Ep {epoch+1:3d} | Train: {train_acc:.2f}% | Test: {acc:.2f}%")

    total_time = time.time() - start
    
    # Inference Speed Test
    model.eval()
    start_infer = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            _ = model(x.to(device))
    infer_time = time.time() - start_infer
    samples_per_sec = len(test_loader.dataset) / infer_time

    calc_model = model.module if isinstance(model, nn.DataParallel) else model
    return {
        'test_acc': acc,
        'train_acc': train_acc,
        'params': sum(p.numel() for p in calc_model.parameters() if p.requires_grad),
        'model': calc_model,
        'history': history,
        'total_time': total_time,
        'inference_speed': samples_per_sec
    }
