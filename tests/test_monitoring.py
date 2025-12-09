"""
Tests for experiment monitoring infrastructure.

Tests cover:
- Experiment ID generation
- Directory structure creation
- CheckpointManager functionality
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add spectral_gpt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spectral_gpt'))

from monitoring import (
    generate_experiment_id,
    create_experiment_directory,
    get_git_hash,
    CheckpointManager,
    MetricsLogger
)


def test_experiment_id_generation():
    """Test that experiment IDs are generated correctly"""
    exp_id = generate_experiment_id("test_experiment")
    
    # Should contain experiment name, timestamp, and git hash
    assert "test_experiment" in exp_id
    assert "_" in exp_id
    
    # Should be unique (different timestamps)
    import time
    time.sleep(1.1)  # Sleep longer to ensure different timestamp
    exp_id2 = generate_experiment_id("test_experiment")
    assert exp_id != exp_id2
    
    print(f"✓ Experiment ID generation: {exp_id}")


def test_git_hash():
    """Test git hash retrieval"""
    git_hash = get_git_hash()
    
    # Should return either a hash or "unknown"
    assert isinstance(git_hash, str)
    assert len(git_hash) > 0
    
    print(f"✓ Git hash: {git_hash}")


def test_directory_creation():
    """Test experiment directory structure creation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_id = "test_exp_001"
        dirs = create_experiment_directory(tmpdir, exp_id)
        
        # Check all directories exist
        assert os.path.exists(dirs['root'])
        assert os.path.exists(dirs['checkpoints'])
        assert os.path.exists(dirs['logs'])
        assert os.path.exists(dirs['visualizations'])
        
        # Check structure
        assert dirs['root'] == os.path.join(tmpdir, exp_id)
        assert dirs['checkpoints'] == os.path.join(tmpdir, exp_id, 'checkpoints')
        
        print(f"✓ Directory creation: {dirs['root']}")


def test_checkpoint_manager_basic():
    """Test basic CheckpointManager functionality"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple model
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create checkpoint manager
        manager = CheckpointManager(
            experiment_dir=tmpdir,
            save_interval=100,
            keep_last_n=3
        )
        
        # Test should_checkpoint
        assert not manager.should_checkpoint(0)
        assert not manager.should_checkpoint(50)
        assert manager.should_checkpoint(100)
        assert manager.should_checkpoint(200)
        
        # Save a checkpoint
        loss_history = [1.5, 1.3, 1.1]
        config = {'lr': 0.001, 'model': 'test'}
        
        checkpoint_path = manager.save_checkpoint(
            step=100,
            model=model,
            optimizer=optimizer,
            loss_history=loss_history,
            config=config
        )
        
        assert os.path.exists(checkpoint_path)
        assert 'checkpoint_step_100.pt' in checkpoint_path
        
        # Check symlink exists
        assert manager.latest_symlink.exists()
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")


def test_checkpoint_load():
    """Test checkpoint loading"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save checkpoint
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        manager = CheckpointManager(experiment_dir=tmpdir, save_interval=100)
        
        loss_history = [1.5, 1.3, 1.1]
        config = {'lr': 0.001}
        
        manager.save_checkpoint(100, model, optimizer, loss_history, config)
        
        # Load checkpoint
        checkpoint = manager.load_latest_checkpoint()
        
        assert checkpoint is not None
        assert checkpoint['step'] == 100
        assert checkpoint['loss_history'] == loss_history
        assert checkpoint['config'] == config
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'timestamp' in checkpoint
        assert 'git_hash' in checkpoint
        
        print(f"✓ Checkpoint loaded: step={checkpoint['step']}")


def test_checkpoint_retention():
    """Test checkpoint retention policy"""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        manager = CheckpointManager(
            experiment_dir=tmpdir,
            save_interval=100,
            keep_last_n=3
        )
        
        # Save 5 checkpoints
        for step in [100, 200, 300, 400, 500]:
            manager.save_checkpoint(
                step=step,
                model=model,
                optimizer=optimizer,
                loss_history=[],
                config={}
            )
        
        # Should only keep last 3
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3
        
        # Should be the most recent ones
        steps = [cp['step'] for cp in checkpoints]
        assert steps == [300, 400, 500]
        
        print(f"✓ Checkpoint retention: kept {len(checkpoints)} of 5")


def test_checkpoint_resume():
    """Test resuming from checkpoint"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model and train for a few steps
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        manager = CheckpointManager(experiment_dir=tmpdir, save_interval=100)
        
        # Simulate training
        loss_history = []
        for step in range(1, 201):
            # Dummy forward pass
            x = torch.randn(4, 10)
            y = torch.randn(4, 5)
            loss = nn.functional.mse_loss(model(x), y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            
            if manager.should_checkpoint(step):
                manager.save_checkpoint(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    loss_history=loss_history,
                    config={'lr': 0.001}
                )
        
        # Save original model state
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Create new model and load checkpoint
        new_model = nn.Linear(10, 5)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        
        checkpoint = manager.load_latest_checkpoint()
        assert checkpoint is not None
        
        new_model.load_state_dict(checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Verify model state matches
        for key in original_state:
            assert torch.allclose(original_state[key], new_model.state_dict()[key])
        
        print(f"✓ Checkpoint resume: restored from step {checkpoint['step']}")


def test_checkpoint_list():
    """Test listing checkpoints"""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        manager = CheckpointManager(experiment_dir=tmpdir, save_interval=100)
        
        # Save multiple checkpoints
        for step in [100, 200, 300]:
            manager.save_checkpoint(step, model, optimizer, [], {})
        
        # List checkpoints
        checkpoints = manager.list_checkpoints()
        
        assert len(checkpoints) == 3
        assert all('step' in cp for cp in checkpoints)
        assert all('path' in cp for cp in checkpoints)
        assert all('size_mb' in cp for cp in checkpoints)
        assert all('modified' in cp for cp in checkpoints)
        
        # Should be sorted by step
        steps = [cp['step'] for cp in checkpoints]
        assert steps == sorted(steps)
        
        print(f"✓ Checkpoint list: {len(checkpoints)} checkpoints")


def test_metrics_logger_basic():
    """Test basic MetricsLogger functionality"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(log_dir=tmpdir, log_interval=10)
        
        # Test should_log
        assert logger.should_log(0)
        assert logger.should_log(10)
        assert logger.should_log(20)
        assert not logger.should_log(5)
        assert not logger.should_log(15)
        
        # Log some metrics
        metrics = {
            'loss': 1.5,
            'learning_rate': 0.001,
            'wave_ratio': 0.8
        }
        logger.log_metrics(step=10, metrics=metrics)
        
        # Check files exist
        assert logger.metrics_file.exists()
        assert logger.training_log.exists()
        
        print(f"✓ Metrics logged to {logger.metrics_file}")


def test_metrics_logger_load():
    """Test loading metrics from file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(log_dir=tmpdir, log_interval=10)
        
        # Log multiple steps
        for step in [10, 20, 30]:
            metrics = {
                'loss': 1.5 - step * 0.01,
                'learning_rate': 0.001,
                'perplexity': 4.5 - step * 0.02
            }
            logger.log_metrics(step=step, metrics=metrics)
        
        # Load metrics
        all_metrics = logger.load_metrics()
        
        assert len(all_metrics) == 3
        assert all_metrics[0]['step'] == 10
        assert all_metrics[1]['step'] == 20
        assert all_metrics[2]['step'] == 30
        
        # Check metrics content
        assert 'loss' in all_metrics[0]
        assert 'learning_rate' in all_metrics[0]
        assert 'perplexity' in all_metrics[0]
        assert 'timestamp' in all_metrics[0]
        
        print(f"✓ Loaded {len(all_metrics)} metric entries")


def test_metrics_logger_latest_step():
    """Test getting latest step"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(log_dir=tmpdir, log_interval=10)
        
        # Initially should be 0
        assert logger.get_latest_step() == 0
        
        # Log some steps
        for step in [10, 20, 30, 40]:
            logger.log_metrics(step=step, metrics={'loss': 1.0})
        
        # Should return last step
        assert logger.get_latest_step() == 40
        
        print(f"✓ Latest step: {logger.get_latest_step()}")


def test_metrics_logger_persistence():
    """Test that metrics persist across logger instances"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create logger and log metrics
        logger1 = MetricsLogger(log_dir=tmpdir, log_interval=10)
        logger1.log_metrics(step=10, metrics={'loss': 1.5})
        logger1.log_metrics(step=20, metrics={'loss': 1.3})
        
        # Create new logger instance pointing to same directory
        logger2 = MetricsLogger(log_dir=tmpdir, log_interval=10)
        
        # Should be able to load metrics from first logger
        metrics = logger2.load_metrics()
        assert len(metrics) == 2
        assert metrics[0]['step'] == 10
        assert metrics[1]['step'] == 20
        
        # Should get correct latest step
        assert logger2.get_latest_step() == 20
        
        print(f"✓ Metrics persisted across logger instances")


def test_metrics_logger_flush():
    """Test immediate buffer flush"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(log_dir=tmpdir, log_interval=10)
        
        # Log with flush=True (default)
        logger.log_metrics(step=10, metrics={'loss': 1.5}, flush=True)
        
        # Immediately read file - should be available
        with open(logger.metrics_file, 'r') as f:
            content = f.read()
            assert len(content) > 0
            assert '"step": 10' in content
        
        print(f"✓ Metrics flushed immediately")


def test_metrics_logger_summary():
    """Test metrics summary generation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(log_dir=tmpdir, log_interval=10)
        
        # Log metrics with varying values
        for step in range(10, 101, 10):
            metrics = {
                'loss': 2.0 - step * 0.01,
                'learning_rate': 0.001,
                'perplexity': 7.0 - step * 0.02
            }
            logger.log_metrics(step=step, metrics=metrics)
        
        # Get summary
        summary = logger.get_metrics_summary()
        
        assert summary['total_steps'] == 10
        assert summary['first_step'] == 10
        assert summary['last_step'] == 100
        
        # Check loss statistics
        assert 'loss_min' in summary
        assert 'loss_max' in summary
        assert 'loss_mean' in summary
        assert 'loss_final' in summary
        
        # Loss should be decreasing
        assert summary['loss_final'] < summary['loss_max']
        
        print(f"✓ Metrics summary: {summary['total_steps']} steps, loss {summary['loss_max']:.3f} → {summary['loss_final']:.3f}")


def test_metrics_logger_jsonl_format():
    """Test JSONL format validity"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(log_dir=tmpdir, log_interval=10)
        
        # Log multiple entries
        for step in [10, 20, 30]:
            logger.log_metrics(step=step, metrics={'loss': 1.0, 'lr': 0.001})
        
        # Read file and verify each line is valid JSON
        with open(logger.metrics_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        
        for line in lines:
            # Each line should be valid JSON
            data = json.loads(line)
            assert 'step' in data
            assert 'timestamp' in data
            assert 'loss' in data
            assert 'lr' in data
        
        print(f"✓ JSONL format valid: {len(lines)} lines")


def test_visualization_manager_basic():
    """Test basic VisualizationManager functionality"""
    from monitoring import VisualizationManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_manager = VisualizationManager(
            viz_dir=tmpdir,
            viz_interval=1000
        )
        
        # Test should_visualize
        assert not viz_manager.should_visualize(0)
        assert not viz_manager.should_visualize(500)
        assert viz_manager.should_visualize(1000)
        assert viz_manager.should_visualize(2000)
        
        print(f"✓ VisualizationManager created: {tmpdir}")


def test_visualization_training_plots():
    """Test training plots generation"""
    from monitoring import VisualizationManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_manager = VisualizationManager(viz_dir=tmpdir, viz_interval=1000)
        
        # Generate training plots
        loss_history = [2.0 - i * 0.01 for i in range(100)]
        metrics = {
            'learning_rate': [0.001 * (0.99 ** i) for i in range(100)],
            'perplexity': [7.0 - i * 0.02 for i in range(100)]
        }
        
        viz_manager.generate_training_plots(
            step=1000,
            loss_history=loss_history,
            metrics=metrics
        )
        
        # Check that plot file was created
        plot_files = list(Path(tmpdir).glob('training_dynamics_step_*.png'))
        assert len(plot_files) > 0
        
        print(f"✓ Training plots generated: {plot_files[0].name}")


def test_visualization_model_plots_no_embedding():
    """Test model plots with model that has no embedding"""
    from monitoring import VisualizationManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_manager = VisualizationManager(viz_dir=tmpdir, viz_interval=1000)
        
        # Create simple model without embedding
        model = nn.Linear(10, 5)
        
        # Should handle gracefully (print warning but not crash)
        viz_manager.generate_model_plots(step=1000, model=model)
        
        print(f"✓ Model plots handled model without embedding gracefully")


def test_visualization_comparison_plots():
    """Test comparison plots generation"""
    from monitoring import VisualizationManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_manager = VisualizationManager(viz_dir=tmpdir, viz_interval=1000)
        
        # Create mock experiment data
        experiments = [
            {
                'name': 'Experiment 1',
                'loss_history': [2.0 - i * 0.01 for i in range(100)],
                'metrics': {
                    'learning_rate': [0.001] * 100,
                    'perplexity': [7.0 - i * 0.02 for i in range(100)]
                }
            },
            {
                'name': 'Experiment 2',
                'loss_history': [2.5 - i * 0.015 for i in range(100)],
                'metrics': {
                    'learning_rate': [0.0005] * 100,
                    'perplexity': [8.0 - i * 0.025 for i in range(100)]
                }
            }
        ]
        
        viz_manager.generate_comparison_plots(experiments)
        
        # Check that comparison plot was created
        plot_files = list(Path(tmpdir).glob('experiment_comparison.png'))
        assert len(plot_files) > 0
        
        print(f"✓ Comparison plots generated: {plot_files[0].name}")


def test_config_tracker_basic():
    """Test basic ConfigTracker functionality"""
    from monitoring import ConfigTracker
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ConfigTracker(experiment_dir=tmpdir)
        
        # Save config
        config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 10
        }
        
        tracker.save_config(config)
        
        # Check config file exists
        assert tracker.config_file.exists()
        
        # Load config
        loaded_config = tracker.load_config()
        
        assert 'experiment_id' in loaded_config
        assert 'timestamp' in loaded_config
        assert 'git_hash' in loaded_config
        assert 'config' in loaded_config
        assert 'hardware' in loaded_config
        
        assert loaded_config['config'] == config
        
        print(f"✓ Config saved and loaded: {tracker.config_file}")


def test_config_tracker_with_model():
    """Test ConfigTracker with model info"""
    from monitoring import ConfigTracker
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ConfigTracker(experiment_dir=tmpdir)
        
        # Create model
        model = nn.Linear(10, 5)
        
        # Save config with model
        config = {'learning_rate': 0.001}
        tracker.save_config(config, model=model)
        
        # Load and verify
        loaded_config = tracker.load_config()
        
        assert 'model' in loaded_config
        assert 'type' in loaded_config['model']
        assert 'num_parameters' in loaded_config['model']
        assert 'num_trainable_parameters' in loaded_config['model']
        
        print(f"✓ Config with model info: {loaded_config['model']['num_parameters']} params")


def test_config_tracker_results():
    """Test saving results"""
    from monitoring import ConfigTracker
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ConfigTracker(experiment_dir=tmpdir)
        
        # Save results
        final_metrics = {
            'val_loss': 0.5,
            'perplexity': 1.65,
            'best_val_loss': 0.45,
            'best_step': 5000
        }
        
        generation_samples = [
            "Sample text 1",
            "Sample text 2"
        ]
        
        tracker.save_results(
            final_metrics=final_metrics,
            best_checkpoint='checkpoint_step_5000.pt',
            generation_samples=generation_samples
        )
        
        # Check results file exists
        assert tracker.results_file.exists()
        
        # Load and verify
        with open(tracker.results_file, 'r') as f:
            results = json.load(f)
        
        assert 'experiment_id' in results
        assert 'timestamp' in results
        assert 'final_metrics' in results
        assert 'best_checkpoint' in results
        assert 'generation_samples' in results
        
        assert results['final_metrics'] == final_metrics
        assert results['generation_samples'] == generation_samples
        
        print(f"✓ Results saved: {tracker.results_file}")


def test_config_tracker_hardware_info():
    """Test hardware info collection"""
    from monitoring import ConfigTracker
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ConfigTracker(experiment_dir=tmpdir)
        
        hardware_info = tracker._get_hardware_info()
        
        # Should have basic info
        assert 'platform' in hardware_info
        assert 'python_version' in hardware_info
        assert 'cpu_count' in hardware_info
        assert 'gpu_available' in hardware_info
        
        # If GPU available, should have GPU info
        if hardware_info['gpu_available']:
            assert 'gpu_count' in hardware_info
            assert 'gpu_model' in hardware_info
            assert 'cuda_version' in hardware_info
        
        print(f"✓ Hardware info: GPU={hardware_info['gpu_available']}, CPU={hardware_info['cpu_count']}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Experiment Monitoring Infrastructure")
    print("=" * 60)
    
    tests = [
        test_experiment_id_generation,
        test_git_hash,
        test_directory_creation,
        test_checkpoint_manager_basic,
        test_checkpoint_load,
        test_checkpoint_retention,
        test_checkpoint_resume,
        test_checkpoint_list,
        test_metrics_logger_basic,
        test_metrics_logger_load,
        test_metrics_logger_latest_step,
        test_metrics_logger_persistence,
        test_metrics_logger_flush,
        test_metrics_logger_summary,
        test_metrics_logger_jsonl_format,
        test_visualization_manager_basic,
        test_visualization_training_plots,
        test_visualization_model_plots_no_embedding,
        test_visualization_comparison_plots,
        test_config_tracker_basic,
        test_config_tracker_with_model,
        test_config_tracker_results,
        test_config_tracker_hardware_info
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{test.__name__}:")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
