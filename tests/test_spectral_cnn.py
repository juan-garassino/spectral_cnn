import torch
from src.models.networks import SpectralCNN

def test_spectral_cnn():
    print("Initializing SpectralCNN...")
    model = SpectralCNN(num_waves=4, num_harmonics=3, wave_mode="outer_product")
    print("Model initialized.")
    
    x = torch.randn(1, 1, 28, 28)
    print(f"Input shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (1, 10)
    print("Forward pass successful!")

if __name__ == "__main__":
    test_spectral_cnn()
