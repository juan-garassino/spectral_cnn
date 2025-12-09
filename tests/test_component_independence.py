"""
Test Component Independence for Wave Physics Refactor

This module tests that each physics component can be toggled independently
without breaking model functionality (Requirement 7.5).

Components tested:
1. Wave Embeddings (use_wave_embeddings)
2. Interference Attention (use_interference_attention)
3. RGD Optimizer (WaveNativeOptimizer)
4. QFE Loss (WaveCoherenceLoss)

All combinations should produce valid outputs without errors.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spectral_gpt'))

from spectral_gpt.wave_gpt import WaveGPT, WaveGPTConfig
from spectral_gpt.wave_physics_core import (
    WaveNativeOptimizer, 
    WaveCoherenceLoss,
    create_physics_optimizer,
    create_physics_loss
)


# Test configuration - small model for fast testing
TEST_CONFIG = {
    'vocab_size': 100,
    'd_model': 64,
    'num_layers': 2,
    'num_heads': 4,
    'num_waves': 8,
    'num_harmonics': 2,
    'block_size': 32,
    'dropout': 0.1
}


def create_test_config(**overrides):
    """Create a WaveGPTConfig with test defaults and optional overrides."""
    config_dict = {**TEST_CONFIG, **overrides}
    return WaveGPTConfig(**config_dict)


def create_test_batch(batch_size=2, seq_len=16, vocab_size=100, device='cpu'):
    """Create a test batch of input and target tensors."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


class TestEmbeddingIndependence:
    """Test that wave embeddings can be toggled independently."""
    
    def test_wave_embeddings_enabled(self):
        """Model works with wave embeddings enabled."""
        config = create_test_config(use_wave_embeddings=True, model_type="wave")
        model = WaveGPT(config)
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        logits, loss = model(x, y)
        
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None
        assert not torch.isnan(loss)
    
    def test_wave_embeddings_disabled(self):
        """Model works with wave embeddings disabled (standard embeddings)."""
        config = create_test_config(use_wave_embeddings=False, model_type="wave")
        model = WaveGPT(config)
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        logits, loss = model(x, y)
        
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None
        assert not torch.isnan(loss)
    
    def test_standard_model_type(self):
        """Standard model type works correctly."""
        config = create_test_config(model_type="standard")
        model = WaveGPT(config)
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        logits, loss = model(x, y)
        
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None
        assert not torch.isnan(loss)


class TestAttentionIndependence:
    """Test that attention types can be toggled independently."""
    
    def test_default_wave_attention(self):
        """Default wave attention (WaveInterferenceAttention) works."""
        config = create_test_config(
            use_interference_attention=False,
            pure_wave_attention=False,
            model_type="wave"
        )
        model = WaveGPT(config)
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        logits, loss = model(x, y)
        
        assert logits.shape == (2, 16, config.vocab_size)
        assert not torch.isnan(loss)
    
    def test_interference_attention(self):
        """Physics-based InterferenceAttention works."""
        config = create_test_config(
            use_interference_attention=True,
            pure_wave_attention=False,
            model_type="wave"
        )
        model = WaveGPT(config)
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        logits, loss = model(x, y)
        
        assert logits.shape == (2, 16, config.vocab_size)
        assert not torch.isnan(loss)
    
    def test_pure_wave_attention(self):
        """PureWaveAttention (no softmax) works."""
        config = create_test_config(
            use_interference_attention=False,
            pure_wave_attention=True,
            pure_wave_kernel='elu_plus_one',
            model_type="wave"
        )
        model = WaveGPT(config)
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        logits, loss = model(x, y)
        
        assert logits.shape == (2, 16, config.vocab_size)
        assert not torch.isnan(loss)


class TestOptimizerIndependence:
    """Test that optimizer can be toggled independently."""
    
    def test_adamw_optimizer(self):
        """Model trains with standard AdamW optimizer."""
        config = create_test_config(model_type="wave")
        model = WaveGPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        # Training step
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        assert not torch.isnan(loss)
    
    def test_wave_native_optimizer(self):
        """Model trains with WaveNativeOptimizer (RGD)."""
        config = create_test_config(model_type="wave")
        model = WaveGPT(config)
        optimizer = WaveNativeOptimizer(
            model.parameters(),
            lr=1e-3,
            damping=0.1,
            coherence_weight=0.7
        )
        
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        # Training step
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        assert not torch.isnan(loss)
    
    def test_optimizer_via_compatibility_function(self):
        """Optimizer created via create_physics_optimizer works."""
        config = create_test_config(model_type="wave")
        model = WaveGPT(config)
        
        # Test with resonance enabled
        optimizer = create_physics_optimizer(model, lr=1e-3, use_resonance=True)
        assert isinstance(optimizer, WaveNativeOptimizer)
        
        # Test with resonance disabled
        optimizer = create_physics_optimizer(model, lr=1e-3, use_resonance=False)
        assert isinstance(optimizer, torch.optim.AdamW)


class TestLossIndependence:
    """Test that loss function can be toggled independently."""
    
    def test_cross_entropy_loss(self):
        """Model trains with standard CrossEntropyLoss."""
        config = create_test_config(model_type="wave")
        model = WaveGPT(config)
        
        x, y = create_test_batch(vocab_size=config.vocab_size)
        logits, ce_loss = model(x, y)
        
        # Standard CE loss is returned by model
        assert not torch.isnan(ce_loss)
    
    def test_wave_coherence_loss(self):
        """Model trains with WaveCoherenceLoss (QFE)."""
        config = create_test_config(model_type="wave")
        model = WaveGPT(config)
        loss_fn = WaveCoherenceLoss(
            lambda_phase=0.01,
            lambda_energy=0.01,
            lambda_harmonic=0.01
        )
        
        x, y = create_test_batch(vocab_size=config.vocab_size)
        logits, _ = model(x, y)
        
        # Compute QFE loss
        loss_dict = loss_fn(logits, y)
        
        assert 'total' in loss_dict
        assert 'ce' in loss_dict
        assert 'coherence' in loss_dict
        assert not torch.isnan(loss_dict['total'])
    
    def test_loss_via_compatibility_function(self):
        """Loss created via create_physics_loss works."""
        # Test with QFE enabled
        loss_fn = create_physics_loss(use_qfe=True)
        assert isinstance(loss_fn, WaveCoherenceLoss)
        
        # Test with QFE disabled
        loss_fn = create_physics_loss(use_qfe=False)
        # Should return a wrapper that still provides dict interface
        logits = torch.randn(2, 16, 100)
        targets = torch.randint(0, 100, (2, 16))
        loss_dict = loss_fn(logits, targets)
        assert 'total' in loss_dict


class TestAllCombinations:
    """Test all combinations of physics components."""
    
    @pytest.mark.parametrize("use_wave_embeddings", [True, False])
    @pytest.mark.parametrize("use_interference_attention", [True, False])
    @pytest.mark.parametrize("use_rgd", [True, False])
    @pytest.mark.parametrize("use_qfe", [True, False])
    def test_component_combination(
        self, 
        use_wave_embeddings, 
        use_interference_attention, 
        use_rgd, 
        use_qfe
    ):
        """
        Test that all combinations of physics components work together.
        
        This is the core test for Requirement 7.5: Component Independence.
        Each component should be independently toggleable without breaking
        model functionality.
        """
        # Create config with specified component settings
        config = create_test_config(
            use_wave_embeddings=use_wave_embeddings,
            use_interference_attention=use_interference_attention,
            pure_wave_attention=False,  # Don't combine with interference attention
            model_type="wave"
        )
        
        # Create model
        model = WaveGPT(config)
        
        # Create optimizer based on use_rgd flag
        if use_rgd:
            optimizer = WaveNativeOptimizer(
                model.parameters(),
                lr=1e-3,
                damping=0.1,
                coherence_weight=0.7
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Create loss function based on use_qfe flag
        if use_qfe:
            loss_fn = WaveCoherenceLoss(
                lambda_phase=0.01,
                lambda_energy=0.01,
                lambda_harmonic=0.01
            )
        else:
            loss_fn = None
        
        # Create test batch
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        # Forward pass
        logits, ce_loss = model(x, y)
        
        # Compute loss
        if loss_fn is not None:
            loss_dict = loss_fn(logits, y)
            loss = loss_dict['total']
        else:
            loss = ce_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Verify outputs
        assert logits.shape == (2, 16, config.vocab_size), \
            f"Unexpected logits shape: {logits.shape}"
        assert not torch.isnan(loss), \
            f"NaN loss with config: wave_embed={use_wave_embeddings}, " \
            f"interference_attn={use_interference_attention}, " \
            f"rgd={use_rgd}, qfe={use_qfe}"


class TestGenerationWithComponents:
    """Test that generation works with different component configurations."""
    
    @pytest.mark.parametrize("use_wave_embeddings", [True, False])
    @pytest.mark.parametrize("use_interference_attention", [True, False])
    def test_generation(self, use_wave_embeddings, use_interference_attention):
        """Generation works with different component configurations."""
        config = create_test_config(
            use_wave_embeddings=use_wave_embeddings,
            use_interference_attention=use_interference_attention,
            model_type="wave"
        )
        model = WaveGPT(config)
        model.eval()
        
        # Start with a prompt
        prompt = torch.randint(0, config.vocab_size, (1, 4))
        
        # Generate
        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10)
        
        assert output.shape == (1, 12)  # 4 prompt + 8 generated
        assert (output >= 0).all() and (output < config.vocab_size).all()


class TestAnnealingWithComponents:
    """Test that embedding annealing works with different configurations."""
    
    def test_annealing_with_wave_embeddings(self):
        """Annealing works when wave embeddings are enabled."""
        config = create_test_config(use_wave_embeddings=True, model_type="wave")
        model = WaveGPT(config)
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        # Test different annealing ratios
        for ratio in [0.0, 0.5, 1.0]:
            logits, loss = model(x, y, standard_embed_ratio=ratio)
            assert not torch.isnan(loss), f"NaN loss at ratio {ratio}"
    
    def test_annealing_with_standard_embeddings(self):
        """Annealing is ignored when standard embeddings are used."""
        config = create_test_config(use_wave_embeddings=False, model_type="wave")
        model = WaveGPT(config)
        model.eval()  # Disable dropout for deterministic comparison
        x, y = create_test_batch(vocab_size=config.vocab_size)
        
        # Annealing ratio should be ignored for standard embeddings
        with torch.no_grad():
            logits1, loss1 = model(x, y, standard_embed_ratio=0.0)
            logits2, loss2 = model(x, y, standard_embed_ratio=1.0)
        
        # Outputs should be identical since standard embeddings don't use annealing
        assert torch.allclose(logits1, logits2), \
            "Standard embeddings should ignore annealing ratio"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
