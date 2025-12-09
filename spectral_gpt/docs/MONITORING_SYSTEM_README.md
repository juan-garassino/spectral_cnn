# Spectral GPT Monitoring System

A comprehensive experiment monitoring infrastructure for robust training with checkpointing, metrics logging, visualization, and configuration tracking.

## Overview

The monitoring system provides four core components that work together to ensure reliable, reproducible, and analyzable experiments:

1. **CheckpointManager** - Save and resume training runs without data loss
2. **MetricsLogger** - Incrementally log training metrics with immediate persistence
3. **VisualizationManager** - Generate and save plots during training
4. **ConfigTracker** - Track experiment configuration and results

## Quick Start

### Basic Usage

```python
from spectral_gpt.monitoring import (
    CheckpointManager,
    MetricsLogger,
    VisualizationManager,
    ConfigTracker,
    generate_experiment_id,
    create_experiment_directory
)

# Create experiment directory structure
experiment_id = generate_experiment_id("wave_rgd_qfe")
dirs = create_experiment_directory("experiments", experiment_id)

# Initialize monitoring components
checkpoint_mgr = CheckpointManager(
    experiment_dir=dirs['root'],
    save_interval=1000,
    keep_last_n=3
)

metrics_logger = MetricsLogger(
    log_dir=dirs['logs'],
    log_interval=10
)

viz_manager = VisualizationManager(
    viz_dir=dirs['visualizations'],
    viz_interval=1000
)

config_tracker = ConfigTracker(experiment_dir=dirs['root'])

# Save initial configuration
config_tracker.save_config(
    config={'lr': 0.001, 'batch_size': 32, ...},
    model=model,
    dataset_info={'name': 'TinyShakespeare', 'num_tokens': 1000000}
)

# Training loop
for step in range(num_steps):
    # ... training code ...
    
    # Log metrics
    if metrics_logger.should_log(step):
        metrics_logger.log_metrics(step, {
            'loss': loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'perplexity': perplexity
        })
    
    # Save checkpoint
    if checkpoint_mgr.should_checkpoint(step):
        checkpoint_mgr.save_checkpoint(
            step, model, optimizer, loss_history, config
        )
    
    # Generate visualizations
    if viz_manager.should_visualize(step):
        viz_manager.generate_training_plots(step, loss_history, metrics)
        viz_manager.generate_model_plots(step, model)

# Save final results
config_tracker.save_results(
    final_metrics={'val_loss': 4.5, 'perplexity': 90.0},
    best_checkpoint='checkpoints/checkpoint_step_5000.pt',
    generation_samples=['The theory of...', 'In the beginning...']
)
```

## Components

### 1. CheckpointManager

Manages model checkpointing with automatic cleanup and atomic writes.

#### Features

- **Atomic writes** - Uses temporary files to prevent corruption
- **Automatic cleanup** - Keeps only N most recent checkpoints
- **Easy resumption** - Symlink to latest checkpoint
- **Complete state** - Saves model, optimizer, loss history, and config

#### Configuration Options

```python
CheckpointManager(
    experiment_dir: str,      # Root directory for experiment
    save_interval: int = 1000,  # Steps between checkpoints
    keep_last_n: int = 3       # Number of checkpoints to retain
)
```

#### Usage Examples

**Save a checkpoint:**

```python
if checkpoint_mgr.should_checkpoint(step):
    checkpoint_path = checkpoint_mgr.save_checkpoint(
        step=step,
        model=model,
        optimizer=optimizer,
        loss_history=loss_history,
        config=config
    )
    print(f"Saved checkpoint: {checkpoint_path}")
```

**Resume from latest checkpoint:**

```python
checkpoint = checkpoint_mgr.load_latest_checkpoint()
if checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1
    loss_history = checkpoint['loss_history']
    print(f"Resumed from step {checkpoint['step']}")
else:
    start_step = 0
    loss_history = []
    print("Starting from scratch")
```

**Load specific checkpoint:**

```python
checkpoint = checkpoint_mgr.load_checkpoint(step=5000)
if checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
```

**List all checkpoints:**

```python
checkpoints = checkpoint_mgr.list_checkpoints()
for ckpt in checkpoints:
    print(f"Step {ckpt['step']}: {ckpt['size_mb']:.2f} MB, {ckpt['modified']}")
```

### 2. MetricsLogger

Logs training metrics with immediate persistence using JSONL format.

#### Features

- **JSONL format** - One JSON object per line for easy streaming
- **Immediate flush** - Ensures data persistence even on interruption
- **Dual logging** - Both machine-readable (JSONL) and human-readable formats
- **Flexible metrics** - Log any dictionary of metrics

#### Configuration Options

```python
MetricsLogger(
    log_dir: str,           # Directory for log files
    log_interval: int = 10  # Steps between log entries
)
```

#### Usage Examples

**Log metrics:**

```python
if metrics_logger.should_log(step):
    metrics_logger.log_metrics(step, {
        'loss': loss.item(),
        'learning_rate': optimizer.param_groups[0]['lr'],
        'wave_ratio': 0.8,
        'perplexity': 90.0,
        'tokens_per_sec': 1500.0
    })
```

**Load all metrics:**

```python
all_metrics = metrics_logger.load_metrics()
for entry in all_metrics:
    print(f"Step {entry['step']}: Loss={entry['loss']:.4f}")
```

**Get latest step:**

```python
latest_step = metrics_logger.get_latest_step()
print(f"Last logged step: {latest_step}")
```

**Get summary statistics:**

```python
summary = metrics_logger.get_metrics_summary()
print(f"Total steps: {summary['total_steps']}")
print(f"Final loss: {summary['loss_final']:.4f}")
print(f"Min loss: {summary['loss_min']:.4f}")
print(f"Mean loss: {summary['loss_mean']:.4f}")
```

### 3. VisualizationManager

Generates and saves visualizations during training.

#### Features

- **Training dynamics** - Loss curves, learning rate, perplexity
- **Model internals** - Wave properties (frequencies, phases, harmonics)
- **Timestamped files** - Track evolution over training
- **Dark theme** - Consistent, professional styling
- **Error handling** - Gracefully handles missing model attributes

#### Configuration Options

```python
VisualizationManager(
    viz_dir: str,            # Directory for visualization files
    viz_interval: int = 1000  # Steps between visualizations
)
```

#### Usage Examples

**Generate training plots:**

```python
if viz_manager.should_visualize(step):
    viz_manager.generate_training_plots(
        step=step,
        loss_history=loss_history,
        metrics={
            'learning_rate': lr_history,
            'perplexity': perplexity_history,
            'wave_ratio': wave_ratio_history
        }
    )
```

**Generate model plots:**

```python
if viz_manager.should_visualize(step):
    viz_manager.generate_model_plots(step=step, model=model)
```

**Generate comparison plots:**

```python
experiments = [
    {
        'name': 'Standard Transformer',
        'loss_history': [7.0, 6.5, 6.0, ...],
        'metrics': {'learning_rate': [0.001, 0.0009, ...]}
    },
    {
        'name': 'Wave GPT',
        'loss_history': [6.8, 6.2, 5.8, ...],
        'metrics': {'learning_rate': [0.001, 0.0009, ...]}
    }
]
viz_manager.generate_comparison_plots(experiments)
```

### 4. ConfigTracker

Tracks experiment configuration and results for reproducibility.

#### Features

- **Complete configuration** - Hyperparameters, model architecture, dataset info
- **Reproducibility** - Git hash, timestamp, hardware info
- **JSON format** - Easy parsing and comparison
- **Final results** - Best metrics, checkpoints, generation samples

#### Configuration Options

```python
ConfigTracker(
    experiment_dir: str  # Root directory for experiment
)
```

#### Usage Examples

**Save configuration:**

```python
config_tracker.save_config(
    config={
        'model': {
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'num_waves': 16,
            'num_harmonics': 4
        },
        'training': {
            'optimizer': 'RGD',
            'lr': 0.001,
            'batch_size': 32,
            'steps': 10000,
            'use_rgd': True,
            'use_qfe': True
        }
    },
    model=model,
    dataset_info={
        'name': 'TinyShakespeare',
        'num_tokens': 1000000,
        'train_split': 0.9,
        'val_split': 0.1
    }
)
```

**Load configuration:**

```python
config = config_tracker.load_config()
print(f"Experiment ID: {config['experiment_id']}")
print(f"Git hash: {config['git_hash']}")
print(f"GPU: {config['hardware']['gpu_model']}")
```

**Save results:**

```python
config_tracker.save_results(
    final_metrics={
        'val_loss': 4.5,
        'perplexity': 90.0,
        'best_val_loss': 4.3,
        'best_step': 8500,
        'total_time_seconds': 3600,
        'tokens_per_second': 1500
    },
    best_checkpoint='checkpoints/checkpoint_step_8500.pt',
    generation_samples=[
        'The theory of quantum mechanics...',
        'In the beginning, there was...'
    ]
)
```

## Resuming from Checkpoints

### Interactive Resumption

```python
# Check for existing checkpoint
checkpoint = checkpoint_mgr.load_latest_checkpoint()

if checkpoint:
    print(f"Found checkpoint at step {checkpoint['step']}")
    response = input("Resume from checkpoint? (y/n): ")
    
    if response.lower() == 'y':
        # Load model and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Resume from next step
        start_step = checkpoint['step'] + 1
        loss_history = checkpoint['loss_history']
        
        print(f"Resumed training from step {start_step}")
    else:
        start_step = 0
        loss_history = []
        print("Starting fresh training")
else:
    start_step = 0
    loss_history = []
    print("No checkpoint found, starting from scratch")
```

### Automatic Resumption

```python
# Always resume from latest checkpoint if available
checkpoint = checkpoint_mgr.load_latest_checkpoint()

if checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1
    loss_history = checkpoint['loss_history']
    print(f"✓ Resumed from step {checkpoint['step']}")
else:
    start_step = 0
    loss_history = []
    print("✓ Starting new training run")

# Continue training
for step in range(start_step, num_steps):
    # ... training code ...
```

### Verifying Loss Continuity

```python
checkpoint = checkpoint_mgr.load_latest_checkpoint()

if checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1
    loss_history = checkpoint['loss_history']
    
    # Verify loss continuity
    expected_loss = loss_history[-1]
    
    # Run one forward pass to check
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        current_loss = criterion(outputs, targets).item()
    model.train()
    
    loss_diff = abs(current_loss - expected_loss)
    if loss_diff > 0.1:
        print(f"⚠️  Warning: Loss discontinuity detected!")
        print(f"   Expected: {expected_loss:.4f}, Got: {current_loss:.4f}")
    else:
        print(f"✓ Loss continuity verified: {current_loss:.4f}")
```

## Directory Structure

The monitoring system creates the following directory structure:

```
experiments/
└── {experiment_id}/
    ├── checkpoints/
    │   ├── checkpoint_step_1000.pt
    │   ├── checkpoint_step_2000.pt
    │   ├── checkpoint_step_3000.pt
    │   └── checkpoint_latest.pt -> checkpoint_step_3000.pt
    ├── logs/
    │   ├── metrics.jsonl
    │   └── training.log
    ├── visualizations/
    │   ├── training_dynamics_step_1000.png
    │   ├── frequencies_step_1000.png
    │   ├── phases_step_1000.png
    │   ├── harmonics_step_1000.png
    │   ├── wave_packets_step_1000.png
    │   └── ...
    ├── config.json
    └── results.json
```

## Troubleshooting

### Checkpoint Issues

**Problem: Checkpoint file is corrupted**

```
Warning: Failed to load checkpoint from checkpoint_latest.pt: ...
```

**Solution:** The CheckpointManager automatically tries previous checkpoints:

```python
checkpoint = checkpoint_mgr.load_latest_checkpoint()
# Automatically tries checkpoint_step_3000.pt, checkpoint_step_2000.pt, etc.
```

**Problem: Disk full during checkpoint save**

**Solution:** The system skips the checkpoint and continues training:

```python
# Checkpoint save will log a warning but not crash
# Training continues without interruption
```

**Problem: Permission denied when saving checkpoint**

**Solution:** Check directory permissions before starting:

```python
import os
experiment_dir = "experiments/my_exp"
if not os.access(experiment_dir, os.W_OK):
    print(f"Error: No write permission for {experiment_dir}")
    exit(1)
```

### Logging Issues

**Problem: Metrics not appearing in log file**

**Solution:** Ensure flush=True (default) or manually flush:

```python
metrics_logger.log_metrics(step, metrics, flush=True)
```

**Problem: Invalid JSON in metrics.jsonl**

**Solution:** The loader skips invalid lines automatically:

```python
metrics = metrics_logger.load_metrics()
# Invalid lines are skipped with a warning
```

**Problem: Log file growing too large**

**Solution:** Reduce log_interval or archive old logs:

```python
# Reduce logging frequency
metrics_logger = MetricsLogger(log_dir=dirs['logs'], log_interval=100)

# Or archive old logs periodically
import shutil
shutil.move('logs/metrics.jsonl', 'logs/metrics_archive_20241209.jsonl')
```

### Visualization Issues

**Problem: Matplotlib rendering failure**

```
Warning: Failed to generate training plots: ...
```

**Solution:** The system catches exceptions and continues training. Check:

1. Matplotlib is installed: `pip install matplotlib`
2. Display backend is available (for headless servers, use 'Agg'):

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

**Problem: Model plots not generated**

```
Warning: Model does not have 'embedding' attribute, skipping model plots
```

**Solution:** Model plots require wave-specific attributes. This is expected for standard transformers. Only training dynamics plots will be generated.

**Problem: Out of memory during visualization**

**Solution:** Reduce visualization frequency or generate plots in background:

```python
# Increase interval
viz_manager = VisualizationManager(viz_dir=dirs['visualizations'], viz_interval=5000)

# Or generate asynchronously (advanced)
import threading
def generate_async():
    viz_manager.generate_model_plots(step, model.cpu())

thread = threading.Thread(target=generate_async)
thread.start()
# Don't wait for completion
```

### Configuration Issues

**Problem: Git hash shows "unknown"**

**Solution:** Ensure you're in a git repository:

```bash
git init
git add .
git commit -m "Initial commit"
```

**Problem: Hardware info missing GPU details**

**Solution:** Ensure CUDA is available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Best Practices

### 1. Choose Appropriate Intervals

```python
# For quick experiments (< 1000 steps)
checkpoint_interval = 100
log_interval = 10
viz_interval = 100

# For medium experiments (1000-10000 steps)
checkpoint_interval = 500
log_interval = 10
viz_interval = 500

# For long experiments (> 10000 steps)
checkpoint_interval = 1000
log_interval = 50
viz_interval = 1000
```

### 2. Monitor Disk Space

```python
import shutil

def check_disk_space(path, min_gb=10):
    """Check if sufficient disk space is available"""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb < min_gb:
        print(f"⚠️  Warning: Only {free_gb:.1f} GB free on {path}")
        return False
    return True

# Before starting experiment
if not check_disk_space("experiments", min_gb=10):
    print("Insufficient disk space!")
    exit(1)
```

### 3. Use Descriptive Experiment Names

```python
# Good: Descriptive names
experiment_id = generate_experiment_id("wave_rgd_qfe_lr0.001_bs32")

# Bad: Generic names
experiment_id = generate_experiment_id("exp1")
```

### 4. Save Configuration Early

```python
# Save config immediately after creating experiment directory
config_tracker.save_config(config, model, dataset_info)

# This ensures config is saved even if training crashes early
```

### 5. Verify Checkpoints Periodically

```python
# Every N checkpoints, verify the latest one loads correctly
if step % (checkpoint_interval * 10) == 0:
    test_checkpoint = checkpoint_mgr.load_latest_checkpoint()
    if test_checkpoint is None:
        print("⚠️  Warning: Failed to load latest checkpoint!")
```

## Advanced Usage

### Custom Metrics

```python
# Log custom metrics specific to your model
metrics_logger.log_metrics(step, {
    'loss': loss.item(),
    'wave_ratio': model.get_wave_ratio(),
    'coherence_loss': coherence_loss.item(),
    'gradient_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),
    'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2
})
```

### Conditional Checkpointing

```python
# Save checkpoint only when validation loss improves
if val_loss < best_val_loss:
    best_val_loss = val_loss
    checkpoint_mgr.save_checkpoint(
        step, model, optimizer, loss_history, config
    )
    print(f"✓ Saved best checkpoint at step {step}")
```

### Multi-GPU Support

```python
# CheckpointManager handles DataParallel automatically
model = nn.DataParallel(model)

# Saves model.module.state_dict() automatically
checkpoint_mgr.save_checkpoint(step, model, optimizer, loss_history, config)

# Load works the same way
checkpoint = checkpoint_mgr.load_latest_checkpoint()
model.load_state_dict(checkpoint['model_state_dict'])
```

### Experiment Comparison

```python
from spectral_gpt.monitoring import ResultsParser, ResultsAggregator

# Parse results from multiple experiments
parser = ResultsParser("spectral_gpt/experiment_results/results.txt")
experiments = parser.parse()

# Aggregate and compare
aggregator = ResultsAggregator()
aggregator.load_from_parser(parser)

# Generate comparison table
print(aggregator.generate_comparison_table())

# Get summary statistics
summary = aggregator.generate_summary_statistics()
print(f"Mean validation loss: {summary['val_loss']['mean']:.4f}")
print(f"Best validation loss: {summary['val_loss']['min']:.4f}")
```

## Integration with Existing Code

### Minimal Integration

Add monitoring to existing training loop with minimal changes:

```python
# Add at the top
from spectral_gpt.monitoring import (
    CheckpointManager, MetricsLogger, VisualizationManager, ConfigTracker,
    generate_experiment_id, create_experiment_directory
)

# Before training loop
experiment_id = generate_experiment_id("my_experiment")
dirs = create_experiment_directory("experiments", experiment_id)

checkpoint_mgr = CheckpointManager(dirs['root'])
metrics_logger = MetricsLogger(dirs['logs'])
viz_manager = VisualizationManager(dirs['visualizations'])
config_tracker = ConfigTracker(dirs['root'])

config_tracker.save_config(config, model, dataset_info)

# In training loop - add these 3 lines
if metrics_logger.should_log(step):
    metrics_logger.log_metrics(step, {'loss': loss.item()})

if checkpoint_mgr.should_checkpoint(step):
    checkpoint_mgr.save_checkpoint(step, model, optimizer, loss_history, config)

if viz_manager.should_visualize(step):
    viz_manager.generate_training_plots(step, loss_history, {})

# After training
config_tracker.save_results({'val_loss': final_val_loss})
```

## See Also

- [Paper Generation Documentation](PAPER_GENERATION_README.md) - Generate academic papers from experiments
- [Configuration Guide](../docs/configuration_guide.md) - Detailed configuration options
- [Experiment Results](experiment_results/) - Example experiment outputs
