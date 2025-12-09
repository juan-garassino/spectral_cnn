# Monitoring Quick Start Guide

## Overview

The Spectral GPT training infrastructure now includes comprehensive monitoring capabilities that automatically track experiments, save checkpoints, log metrics, and generate visualizations.

## Quick Start

### Running Experiments with Monitoring

Monitoring is **enabled by default** for all training runs. Simply run your experiments as usual:

```bash
# Run ablation experiments
python spectral_gpt/wave_experiments.py --experiment all --steps 10000

# Run benchmarks
python spectral_gpt/wave_benchmark.py
```

### What Gets Tracked

Each experiment automatically creates:

1. **Checkpoints** - Model and optimizer state saved every 1000 steps
2. **Metrics Logs** - Training metrics logged every 10 steps
3. **Visualizations** - Plots generated every 1000 steps
4. **Configuration** - Full experiment config saved at start
5. **Results** - Final metrics and samples saved at end

### Finding Your Results

All experiment data is saved in the `experiments/` directory:

```
experiments/
└── {experiment_name}_{timestamp}_{git_hash}/
    ├── checkpoints/
    ├── logs/
    ├── visualizations/
    ├── config.json
    └── results.json
```

Example:
```
experiments/wave_baseline_20241209_143022_a3f2b1c4/
```

## Resuming from Checkpoints

If training is interrupted, you can resume from the last checkpoint:

1. The system automatically detects existing checkpoints
2. Currently set to **not auto-resume** by default (for safety)
3. To enable auto-resume, modify the `resume` flag in `train_experiment()`:

```python
# In wave_experiments.py, line ~150
resume = True  # Change from False to True
```

## Monitoring Components

### CheckpointManager
- **Save interval**: 1000 steps (experiments), 500 steps (benchmarks)
- **Retention**: Keeps last 3 checkpoints
- **Location**: `experiments/{exp_id}/checkpoints/`

### MetricsLogger
- **Log interval**: 10 steps
- **Format**: JSONL (one JSON object per line)
- **Location**: `experiments/{exp_id}/logs/metrics.jsonl`
- **Metrics tracked**:
  - loss
  - learning_rate
  - tokens_per_sec
  - wave_ratio (for wave models)
  - coherence_loss (when using QFE)
  - perplexity

### VisualizationManager
- **Viz interval**: 1000 steps (experiments), 500 steps (benchmarks)
- **Location**: `experiments/{exp_id}/visualizations/`
- **Plots generated**:
  - Training dynamics (loss curves, learning rate, perplexity)
  - Frequency distributions
  - Phase distributions
  - Harmonic amplitudes
  - Wave packets

### ConfigTracker
- **Saves at**: Experiment start and end
- **Location**: `experiments/{exp_id}/config.json` and `results.json`
- **Includes**:
  - All hyperparameters
  - Model architecture details
  - Dataset information
  - Git commit hash
  - Hardware info
  - Final metrics

## Analyzing Results

### Loading Metrics

```python
from monitoring import MetricsLogger

logger = MetricsLogger('experiments/{exp_id}/logs')
metrics = logger.load_metrics()

# Get summary statistics
summary = logger.get_metrics_summary()
print(summary)
```

### Loading Checkpoints

```python
from monitoring import CheckpointManager

manager = CheckpointManager('experiments/{exp_id}')

# Load latest checkpoint
checkpoint = manager.load_latest_checkpoint()

# Load specific checkpoint
checkpoint = manager.load_checkpoint(step=5000)

# List all checkpoints
checkpoints = manager.list_checkpoints()
```

### Loading Configuration

```python
from monitoring import ConfigTracker

tracker = ConfigTracker('experiments/{exp_id}')
config = tracker.load_config()
print(config)
```

## Disabling Monitoring

If you need to disable monitoring (e.g., for quick tests):

```python
# In your training script
result = train_experiment(
    model, train_data, val_data, exp_config, model_config, console, device,
    experiment_dir=None,
    enable_monitoring=False
)
```

## Troubleshooting

### Disk Space Issues

If you're running out of disk space:

1. Reduce checkpoint retention:
   ```python
   CheckpointManager(experiment_dir, save_interval=1000, keep_last_n=1)
   ```

2. Increase checkpoint interval:
   ```python
   CheckpointManager(experiment_dir, save_interval=2000, keep_last_n=3)
   ```

3. Clean up old experiments:
   ```bash
   rm -rf experiments/old_experiment_*
   ```

### Monitoring Failures

All monitoring operations fail gracefully with warnings. If you see warnings:

1. Check disk space: `df -h`
2. Check write permissions: `ls -la experiments/`
3. Check logs: `cat experiments/{exp_id}/logs/training.log`

### Missing Visualizations

If visualizations aren't being generated:

1. Check matplotlib is installed: `pip install matplotlib`
2. Check visualization interval matches your training steps
3. Look for warnings in console output

## Best Practices

1. **Use descriptive experiment names** - Makes it easier to find results later
2. **Monitor disk space** - Checkpoints can be large for big models
3. **Keep git repo clean** - Git hash is used for reproducibility
4. **Document experiments** - Add notes to config files after runs
5. **Archive old experiments** - Move completed experiments to archive directory

## Advanced Usage

### Custom Monitoring Intervals

```python
# More frequent monitoring for debugging
checkpoint_manager = CheckpointManager(exp_dir, save_interval=100, keep_last_n=5)
metrics_logger = MetricsLogger(log_dir, log_interval=1)
viz_manager = VisualizationManager(viz_dir, viz_interval=100)
```

### Comparing Experiments

```python
from monitoring import MetricsLogger
import matplotlib.pyplot as plt

# Load metrics from multiple experiments
exp1_metrics = MetricsLogger('experiments/exp1/logs').load_metrics()
exp2_metrics = MetricsLogger('experiments/exp2/logs').load_metrics()

# Plot comparison
plt.plot([m['loss'] for m in exp1_metrics], label='Exp 1')
plt.plot([m['loss'] for m in exp2_metrics], label='Exp 2')
plt.legend()
plt.show()
```

## Support

For issues or questions:
1. Check the main documentation: `MONITORING_README.md`
2. Review implementation details: `MONITORING_INTEGRATION_SUMMARY.md`
3. Check the design document: `.kiro/specs/spectral-gpt-paper-and-monitoring/design.md`
