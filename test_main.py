import os
import torch
from src.training.trainer import train_fit
from src.visualization.plotter import plot_results, plot_layer_waves

# Configuration - Reduced for testing
NUM_EPOCHS = 1
NUM_WAVES = 2
MODES = ["UserWave", "GatedWave"] # Test the ones with get_waves

def main():
    base_dir = f"./Test_Run/"
    os.makedirs(base_dir, exist_ok=True)
    device = torch.device("cpu")

    results = {}

    print(f"STARTING TEST RUN")

    for m in MODES:
        results[m] = train_fit(m, NUM_WAVES, NUM_EPOCHS, device)

    # Visualization
    plot_results(results, MODES, base_dir, NUM_EPOCHS)
    
    # Wave Visualization
    print("\n[Generating Wave Plots...]")
    for m in MODES:
        plot_layer_waves(results[m]['model'], m, base_dir)

    print(f"All results and plots saved to {base_dir}")

if __name__ == "__main__":
    main()
