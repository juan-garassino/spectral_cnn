import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_results(results, modes, base_dir, num_epochs):
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
    plt.close()

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
    plt.close()

    # Loss Curves
    plt.figure(figsize=(10, 6))
    for m in modes:
        hist = results[m]['history']
        plt.plot(hist['test_acc'], label=f"{m} (Test)", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.title(f"Learning Curves ({num_epochs} Epochs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_dir}/training_dynamics.png")
    plt.close()

def plot_layer_waves(model, mode_name, base_dir):
    """
    Visualizes the internal waves of the first layer if supported.
    """
    if not hasattr(model.fc1, 'get_waves'):
        print(f"Skipping wave plot for {mode_name} (not supported)")
        return

    print(f"Generating wave plot for {mode_name}...")
    waves = model.fc1.get_waves().detach().cpu().numpy() # [num_waves, out_dim, in_dim] or similar depending on implementation
    # Actually, get_waves returns [num_waves, out_dim, in_dim] ? 
    # Let's check the implementation in layers.py
    # UserWaveLinear: 
    # theta = bmm(u, v.T) * freqs -> [num_waves, out, in]
    # wave = cos(...) -> [num_waves, out, in]
    # returns stack(waves) -> [num_waves, out, in]
    
    # We want to visualize the waves for the first neuron (index 0 in out_dim)
    # waves shape: [num_waves, out_dim, in_dim]
    
    num_waves = waves.shape[0]
    in_dim = waves.shape[2]
    side = int(np.sqrt(in_dim)) 
    
    # Plot top 5 waves or all if less
    waves_to_plot = min(num_waves, 10)
    
    plt.figure(figsize=(waves_to_plot*2, 2))
    for i in range(waves_to_plot):
        wave_img = waves[i, 0, :].reshape(side, side)
        plt.subplot(1, waves_to_plot, i+1)
        plt.imshow(wave_img, cmap='coolwarm')
        plt.title(f"Wave {i}")
        plt.axis('off')
    
    plt.suptitle(f"Internal Waves for {mode_name} (Neuron 0)", y=1.05)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{mode_name}_waves.png")
    plt.close()
