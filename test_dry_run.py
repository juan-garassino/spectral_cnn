#!/usr/bin/env python3
"""
Quick test script for dry-run mode
"""
import os
import sys
import torch

# Add spectral_gpt to path
sys.path.insert(0, 'spectral_gpt')
sys.path.insert(0, 'spectral_gpt/prototyping')

from wave_experiments import create_mock_dataset, ABLATION_EXPERIMENTS, MODEL_CONFIGS
from monitoring import generate_experiment_id, create_experiment_directory
from monitoring import CheckpointManager, MetricsLogger, VisualizationManager, ConfigTracker

def test_dry_run_monitoring():
    """Test that all monitoring components work in dry-run mode"""
    print("🧪 Testing dry-run monitoring components...")
    
    # 1. Test mock dataset creation
    print("\n1. Testing mock dataset creation...")
    data = create_mock_dataset(vocab_size=50257, num_tokens=10000)
    assert data.shape == (10000,), f"Expected shape (10000,), got {data.shape}"
    assert data.dtype == torch.long, f"Expected dtype torch.long, got {data.dtype}"
    assert data.min() >= 0 and data.max() < 50257, "Token IDs out of range"
    print("   ✓ Mock dataset created successfully")
    
    # 2. Test experiment ID generation
    print("\n2. Testing experiment ID generation...")
    exp_id = generate_experiment_id("test_dry_run")
    assert "test_dry_run" in exp_id, f"Experiment name not in ID: {exp_id}"
    print(f"   ✓ Generated experiment ID: {exp_id}")
    
    # 3. Test directory creation
    print("\n3. Testing directory creation...")
    dirs = create_experiment_directory("experiments", exp_id)
    assert os.path.exists(dirs['root']), "Root directory not created"
    assert os.path.exists(dirs['checkpoints']), "Checkpoints directory not created"
    assert os.path.exists(dirs['logs']), "Logs directory not created"
    assert os.path.exists(dirs['visualizations']), "Visualizations directory not created"
    print(f"   ✓ Created directories: {dirs['root']}")
    
    # 4. Test CheckpointManager with dry-run intervals
    print("\n4. Testing CheckpointManager...")
    checkpoint_manager = CheckpointManager(
        experiment_dir=dirs['root'],
        save_interval=5,  # Dry-run interval
        keep_last_n=3
    )
    assert checkpoint_manager.should_checkpoint(5), "Should checkpoint at step 5"
    assert not checkpoint_manager.should_checkpoint(3), "Should not checkpoint at step 3"
    print("   ✓ CheckpointManager initialized with dry-run intervals")
    
    # 5. Test MetricsLogger with dry-run intervals
    print("\n5. Testing MetricsLogger...")
    metrics_logger = MetricsLogger(
        log_dir=dirs['logs'],
        log_interval=1  # Dry-run interval
    )
    assert metrics_logger.should_log(1), "Should log at step 1"
    assert metrics_logger.should_log(5), "Should log at step 5"
    
    # Log some test metrics
    metrics_logger.log_metrics(1, {'loss': 10.5, 'learning_rate': 0.0006})
    metrics_logger.log_metrics(2, {'loss': 9.8, 'learning_rate': 0.0006})
    
    # Verify metrics were logged
    logged_metrics = metrics_logger.load_metrics()
    assert len(logged_metrics) == 2, f"Expected 2 logged entries, got {len(logged_metrics)}"
    assert logged_metrics[0]['step'] == 1, "First entry should be step 1"
    assert logged_metrics[1]['step'] == 2, "Second entry should be step 2"
    print("   ✓ MetricsLogger working correctly")
    
    # 6. Test VisualizationManager with dry-run intervals
    print("\n6. Testing VisualizationManager...")
    viz_manager = VisualizationManager(
        viz_dir=dirs['visualizations'],
        viz_interval=5  # Dry-run interval
    )
    assert viz_manager.should_visualize(5), "Should visualize at step 5"
    assert not viz_manager.should_visualize(3), "Should not visualize at step 3"
    
    # Test training plots generation
    loss_history = [10.5, 9.8, 9.2, 8.7, 8.3]
    metrics = {
        'learning_rate': [0.0006, 0.0006, 0.0006, 0.0006, 0.0006],
        'perplexity': [36000, 18000, 10000, 6000, 4000]
    }
    viz_manager.generate_training_plots(5, loss_history, metrics)
    
    # Check if visualization file was created
    viz_files = os.listdir(dirs['visualizations'])
    assert len(viz_files) > 0, "No visualization files created"
    print(f"   ✓ Generated {len(viz_files)} visualization file(s)")
    
    # 7. Test ConfigTracker
    print("\n7. Testing ConfigTracker...")
    config_tracker = ConfigTracker(experiment_dir=dirs['root'])
    
    test_config = {
        'experiment_name': 'test_dry_run',
        'model_config': {'d_model': 384, 'num_layers': 8},
        'training_config': {'lr': 0.0006, 'steps': 10}
    }
    
    dataset_info = {
        'train_tokens': 9000,
        'val_tokens': 1000,
        'total_tokens': 10000
    }
    
    # Create a dummy model for testing
    import torch.nn as nn
    dummy_model = nn.Linear(10, 10)
    
    config_tracker.save_config(test_config, dummy_model, dataset_info)
    
    # Verify config file was created
    config_file = os.path.join(dirs['root'], 'config.json')
    assert os.path.exists(config_file), "Config file not created"
    print("   ✓ Configuration saved successfully")
    
    # 8. Test results saving
    print("\n8. Testing results saving...")
    final_metrics = {
        'val_loss': 8.3,
        'perplexity': 4000,
        'total_time_seconds': 120,
        'tokens_per_second': 75
    }
    
    config_tracker.save_results(
        final_metrics=final_metrics,
        best_checkpoint=None,
        generation_samples=None
    )
    
    # Verify results file was created
    results_file = os.path.join(dirs['root'], 'results.json')
    assert os.path.exists(results_file), "Results file not created"
    print("   ✓ Results saved successfully")
    
    print("\n✅ All dry-run monitoring tests passed!")
    print(f"\n📁 Test experiment directory: {dirs['root']}")
    
    # Cleanup
    import shutil
    print("\n🧹 Cleaning up test directory...")
    shutil.rmtree(dirs['root'])
    print("   ✓ Cleanup complete")

if __name__ == "__main__":
    test_dry_run_monitoring()
