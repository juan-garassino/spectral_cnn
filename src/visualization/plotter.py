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
    waves = model.fc1.get_waves().detach().cpu().numpy()
    
    num_waves = waves.shape[0]
    in_dim = waves.shape[2]
    side = int(np.sqrt(in_dim)) 
    
    waves_to_plot = min(num_waves, 8)
    
    # Create a 3-row visualization
    fig = plt.figure(figsize=(waves_to_plot*2.5, 9))
    
    for i in range(waves_to_plot):
        wave_2d = waves[i, 0, :].reshape(side, side)
        
        # Row 1: 2D Heatmap of the wave
        plt.subplot(3, waves_to_plot, i+1)
        vmax = max(np.abs(wave_2d).max(), 1e-8)  # Prevent zero range
        plt.imshow(wave_2d, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        plt.title(f"Wave {i}\n(2D)", fontsize=9)
        plt.axis('off')
        
        # Row 2: 1D Cross-section (Middle Row) - This shows the sinusoidal nature
        plt.subplot(3, waves_to_plot, i+1 + waves_to_plot)
        mid_row = wave_2d[side//2, :]
        plt.plot(mid_row, linewidth=1.5)
        plt.title(f"1D Slice", fontsize=9)
        plt.grid(True, alpha=0.3)
        max_amp = max(np.abs(waves[:waves_to_plot, 0, :]).max(), 1e-8)  # Prevent zero range
        plt.ylim(-max_amp, max_amp)
        
    # Row 3: Superposition (how waves combine)
    plt.subplot(3, 1, 3)
    final_weight = np.zeros(side)
    for i in range(waves_to_plot):
        wave_2d = waves[i, 0, :].reshape(side, side)
        mid_row = wave_2d[side//2, :]
        plt.plot(mid_row, alpha=0.4, linewidth=1, label=f"Wave {i}")
        final_weight += mid_row
    
    plt.plot(final_weight, 'k-', linewidth=2, label='Sum (Final Weight)')
    plt.title(f"Superposition: How {waves_to_plot} Waves Combine", fontsize=11, fontweight='bold')
    plt.xlabel("Position along weight vector")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"Wave Analysis for {mode_name} (Neuron 0)", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{mode_name}_waves.png", dpi=150)
    plt.close()


def plot_wave_decomposition(model, mode_name, base_dir, wave_idx=0):
    """
    Visualizes the Fourier decomposition of a single wave.
    Shows: Fourier Components → Combined Wave → 2D Heatmap
    """
    if not hasattr(model.fc1, 'get_wave_components'):
        print(f"Skipping Fourier decomposition for {mode_name} (not supported)")
        return

    print(f"Generating Fourier decomposition for {mode_name}, Wave {wave_idx}...")
    components = model.fc1.get_wave_components()
    
    if wave_idx >= len(components):
        wave_idx = 0
    
    comp_data = components[wave_idx]
    learned_coeffs = comp_data['coeffs'].cpu().numpy()  # Get learned coefficients
    harmonic_freqs = comp_data['harmonic_freqs'].cpu().numpy()  # Get frequency multipliers
    num_harmonics = len(learned_coeffs)
    
    # Get first component to determine shape
    comp1 = comp_data['comp1'].detach().cpu().numpy()
    side = int(np.sqrt(comp1.shape[1]))
    
    # Determine grid layout based on number of harmonics
    ncols = min(num_harmonics, 4)  # Max 4 columns
    nrows_per_section = int(np.ceil(num_harmonics / ncols))
    
    fig = plt.figure(figsize=(ncols * 3.5, 10))
    
    # Row 1: Individual Fourier Components (1D) - show up to 8
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    for h in range(min(num_harmonics, 8)):
        comp = comp_data[f'comp{h+1}'].detach().cpu().numpy()
        comp_2d = comp[0, :].reshape(side, side)
        mid_row = comp_2d[side//2, :]
        
        plt.subplot(3, min(num_harmonics, 8), h+1)
        label = f"{learned_coeffs[h]:.3f}·cos({int(harmonic_freqs[h])}θ)"
        plt.plot(mid_row, color=colors[h % len(colors)], linewidth=2, label=label)
        plt.title(f"Component: {label}", fontweight='bold', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.ylabel("Amplitude", fontsize=8)
        plt.legend(fontsize=7)
    
    # Row 2: Combined Wave
    plt.subplot(3, 2, (min(num_harmonics, 8) + 1) // min(num_harmonics, 8) * 2 - 1)
    combined_wave = sum(comp_data[f'comp{h+1}'].detach().cpu().numpy() for h in range(num_harmonics))
    combined_2d = combined_wave[0, :].reshape(side, side)
    mid_row_combined = combined_2d[side//2, :]
    
    # Plot individual components faintly
    for h in range(min(num_harmonics, 8)):
        comp = comp_data[f'comp{h+1}'].detach().cpu().numpy()
        comp_mid = comp[0, :].reshape(side, side)[side//2, :]
        plt.plot(comp_mid, alpha=0.3, linewidth=1, 
                label=f'{learned_coeffs[h]:.3f}·cos({int(harmonic_freqs[h])}θ)', 
                color=colors[h % len(colors)])
    
    plt.plot(mid_row_combined, 'k-', linewidth=2.5, label='Combined Wave')
    plt.title("Wave Superposition (1D)", fontweight='bold', fontsize=12)
    plt.xlabel("Position")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    
    # 2D Heatmap of combined wave
    plt.subplot(3, 2, (min(num_harmonics, 8) + 1) // min(num_harmonics, 8) * 2)
    plt.imshow(combined_2d, cmap='viridis')
    plt.title("Combined Wave (2D Heatmap)", fontweight='bold', fontsize=12)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Row 3: Frequency spectrum with LEARNED amplitudes  
    plt.subplot(3, 2, 5)
    plt.stem(harmonic_freqs, learned_coeffs, basefmt=" ")
    plt.title("Learned Frequency Spectrum", fontweight='bold')
    plt.xlabel("Frequency (×ω)")
    plt.ylabel("Learned Amplitude")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Add text showing coefficients
    plt.subplot(3, 2, 6)
    plt.axis('off')
    coeff_text = f"Learned Coefficients (Wave {wave_idx}):\n\n"
    for h in range(num_harmonics):
        coeff_text += f"c_{h+1} = {learned_coeffs[h]:.4f} (×{int(harmonic_freqs[h])}ω)\n"
    coeff_text += f"\nFormula:\nwave = " + " + ".join([f"c_{h+1}·cos({int(harmonic_freqs[h])}θ)" for h in range(num_harmonics)])
    
    plt.text(0.5, 0.5, coeff_text,
             ha='center', va='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"Fourier Decomposition: {mode_name} - Wave {wave_idx}\n({num_harmonics} Harmonics, Optimized Coefficients)", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{mode_name}_fourier_decomposition_wave{wave_idx}.png", dpi=150)
    plt.close()
