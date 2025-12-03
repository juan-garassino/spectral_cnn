import torch
import matplotlib.pyplot as plt
import os
from src.models.layers import UserWaveLinear

def visualize_frequency_balance():
    """
    Visualize how the new frequency-aware initialization distributes
    energy across low/mid/high frequency components.
    """
    print("🎵 Analyzing Frequency-Aware Initialization...")
    
    # Create a layer with default (old) vs frequency-aware (new) init
    layer = UserWaveLinear(784, 10, num_waves=12, num_harmonics=9, wave_mode="outer_product")
    
    # Extract initialization values
    coeffs = layer.fourier_coeffs.detach().cpu().numpy()
    amps = layer.amplitudes.detach().cpu().numpy()
    
    # Analyze by frequency band
    num_harmonics = coeffs.shape[-1]
    low_band = coeffs[..., :num_harmonics//3]
    mid_band = coeffs[..., num_harmonics//3:2*num_harmonics//3]
    high_band = coeffs[..., 2*num_harmonics//3:]
    
    print(f"\n📊 Initial Coefficient Statistics:")
    print(f"  Low Freq  (H 0-2): Mean={low_band.mean():.3f}, Std={low_band.std():.3f}")
    print(f"  Mid Freq  (H 3-5): Mean={mid_band.mean():.3f}, Std={mid_band.std():.3f}")
    print(f"  High Freq (H 6-8): Mean={high_band.mean():.3f}, Std={high_band.std():.3f}")
    
    print(f"\n📊 Wave Amplitude Statistics:")
    print(f"  Wave 0-3  (Low):  Mean={amps[..., :4].mean():.3f}")
    print(f"  Wave 4-7  (Mid):  Mean={amps[..., 4:8].mean():.3f}")
    print(f"  Wave 8-11 (High): Mean={amps[..., 8:].mean():.3f}")
    
    # Create visualizations
    os.makedirs("results/frequency_analysis", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Coefficient distribution by harmonic
    ax = axes[0]
    harmonic_means = [coeffs[..., h].mean() for h in range(num_harmonics)]
    harmonic_stds = [coeffs[..., h].std() for h in range(num_harmonics)]
    
    colors = ['blue']*3 + ['green']*3 + ['red']*3
    bars = ax.bar(range(num_harmonics), harmonic_means, yerr=harmonic_stds, 
                   color=colors, alpha=0.6, capsize=5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Harmonic Index (Frequency Multiplier: 1x, 2x, 4x, ...)')
    ax.set_ylabel('Mean Fourier Coefficient')
    ax.set_title('Frequency-Aware Initialization:\nBalanced Low/Mid/High')
    ax.legend([plt.Rectangle((0,0),1,1, fc='blue', alpha=0.6),
               plt.Rectangle((0,0),1,1, fc='green', alpha=0.6),
               plt.Rectangle((0,0),1,1, fc='red', alpha=0.6)],
              ['Low (0.8)', 'Mid (1.0)', 'High (1.5)'])
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Wave amplitude progression
    ax = axes[1]
    wave_means = [amps[..., w].mean() for w in range(12)]
    ax.plot(wave_means, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Wave Index (Increasing Base Frequency)')
    ax.set_ylabel('Mean Amplitude')
    ax.set_title('Progressive Amplitude Boost:\nEncourages High-Freq Waves')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Text explanation
    ax = axes[2]
    ax.axis('off')
    explanation = """
    🧠 Frequency-Aware Strategy
    
    ╔══════════════════════════╗
    ║  Combating Spectral Bias ║
    ╚══════════════════════════╝
    
    Problem:
    • Networks naturally prefer
      low frequencies (smooth)
    • High frequencies (details)
      get suppressed
    
    Solution:
    • Boost high-freq harmonics
      at initialization
    • Low:  0.8 (standard)
    • Mid:  1.0 (slight boost)
    • High: 1.5 (strong boost)
    
    Result:
    • Balanced spectrum from start
    • Network can use details AND
      smooth patterns
    """
    ax.text(0.1, 0.5, explanation, fontsize=9, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('results/frequency_analysis/frequency_aware_init.png', dpi=150)
    plt.close()
    
    print("\n✅ Visualization saved to: results/frequency_analysis/frequency_aware_init.png")
    print("\nThe network now starts with a balanced frequency spectrum!")

if __name__ == "__main__":
    visualize_frequency_balance()
