import torch
from src.models.networks import UniversalMLP
from src.training.trainer import train_fit
import os

def test_dashboard():
    """Quick test to generate the new dashboard"""
    print("🎨 Testing new dashboard visualization...")
    
    device = torch.device("cpu")
    os.makedirs("results/dashboard_test", exist_ok=True)
    
    # Train two quick models (1 epoch each)
    print("\n1. Training Standard MLP (1 epoch)...")
    results_std = train_fit("Standard", num_waves=12, num_epochs=1, device=device)
    
    print("\n2. Training UserWave MLP (1 epoch)...")
    results_wave = train_fit("UserWave", num_waves=12, num_epochs=1, device=device, 
                            wave_mode="outer_product")
    
    # Create results dict
    results = {
        "Standard": results_std,
        "UserWave": results_wave
    }
    
    # Generate dashboard
    from src.visualization import plotter
    plotter.plot_results(results, ["Standard", "UserWave"], 
                        "results/dashboard_test", num_epochs=1)
    
    print("\n✅ Dashboard test complete!")
    print("Check: results/dashboard_test/comprehensive_dashboard.png")

if __name__ == "__main__":
    test_dashboard()
