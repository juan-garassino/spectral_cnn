# Spectral GPT Documentation Index

Complete guide to the Spectral GPT monitoring and paper generation systems.

## Quick Links

### Core Documentation

- **[Monitoring System README](MONITORING_SYSTEM_README.md)** - Complete guide to experiment monitoring, checkpointing, metrics logging, and visualization
- **[Paper Generation README](PAPER_GENERATION_README.md)** - Guide to generating intuitive guides and technical papers from experiments

### Additional Resources

- **[Monitoring Quick Start](MONITORING_QUICK_START.md)** - Quick start guide for monitoring system
- **[Code Extraction README](CODE_EXTRACTION_README.md)** - Extracting code snippets for documentation
- **[Paper Visualizations README](PAPER_VISUALIZATIONS_README.md)** - Generating figures for papers
- **[Generate Paper CLI README](GENERATE_PAPER_CLI_README.md)** - Command-line interface for paper generation

## Getting Started

### 1. Set Up Monitoring for Your Experiments

Start with the [Monitoring System README](MONITORING_SYSTEM_README.md) to learn how to:

- Save checkpoints during training
- Log metrics incrementally
- Generate visualizations
- Track experiment configuration
- Resume from interruptions

**Quick example:**

```python
from spectral_gpt.monitoring import (
    CheckpointManager, MetricsLogger, VisualizationManager, ConfigTracker,
    generate_experiment_id, create_experiment_directory
)

# Create experiment
experiment_id = generate_experiment_id("my_experiment")
dirs = create_experiment_directory("experiments", experiment_id)

# Initialize monitoring
checkpoint_mgr = CheckpointManager(dirs['root'])
metrics_logger = MetricsLogger(dirs['logs'])
viz_manager = VisualizationManager(dirs['visualizations'])
config_tracker = ConfigTracker(dirs['root'])

# Use in training loop
if checkpoint_mgr.should_checkpoint(step):
    checkpoint_mgr.save_checkpoint(step, model, optimizer, loss_history, config)

if metrics_logger.should_log(step):
    metrics_logger.log_metrics(step, {'loss': loss.item()})

if viz_manager.should_visualize(step):
    viz_manager.generate_training_plots(step, loss_history, metrics)
```

### 2. Generate Papers from Your Results

Once you have experiment results, use the [Paper Generation README](PAPER_GENERATION_README.md) to:

- Generate intuitive guides with visual explanations
- Create technical papers with mathematical rigor
- Extract code snippets automatically
- Render to PDF with pandoc

**Quick example:**

```python
from spectral_gpt.paper_generator import SpectralGPTPaperGenerator
from spectral_gpt.monitoring import ResultsParser

# Parse results
parser = ResultsParser("spectral_gpt/experiment_results/results.txt")
experiments = parser.parse()

# Generate papers
generator = SpectralGPTPaperGenerator(output_dir="experiments/paper")
guide_path = generator.generate_intuitive_guide(experiments)
paper_path = generator.generate_technical_paper(experiments)

# Render to PDF
generator.render_to_pdf(guide_path)
generator.render_to_pdf(paper_path)
```

**Or use the CLI:**

```bash
python spectral_gpt/generate_paper.py --type both --output experiments/paper
```

## Documentation Structure

### Monitoring System

The monitoring system provides four core components:

1. **CheckpointManager** - Save and resume training runs
   - Atomic writes to prevent corruption
   - Automatic cleanup of old checkpoints
   - Easy resumption with symlinks

2. **MetricsLogger** - Log training metrics
   - JSONL format for easy parsing
   - Immediate buffer flush for persistence
   - Dual logging (machine + human readable)

3. **VisualizationManager** - Generate plots during training
   - Training dynamics (loss, learning rate, etc.)
   - Model internals (wave properties)
   - Comparison plots across experiments

4. **ConfigTracker** - Track experiment metadata
   - Complete configuration capture
   - Git hash for reproducibility
   - Hardware information
   - Final results and samples

### Paper Generation System

The paper generation system creates two levels of documentation:

1. **Intuitive Guide** - Visual, conceptual explanations
   - Side-by-side architecture comparisons
   - Layer-by-layer analysis
   - Why different architectures converge to similar loss
   - Intuitive wave properties explanation

2. **Technical Paper** - Mathematical, rigorous documentation
   - Complete mathematical formulation
   - Detailed architecture specifications
   - Experimental methodology
   - Results with statistical analysis
   - Ablation studies
   - Code appendix

### Supporting Components

- **ResultsParser** - Parse experiment results from text files
- **ResultsAggregator** - Compare and analyze multiple experiments
- **CodeExtractor** - Extract code snippets for documentation
- **PaperVisualizations** - Generate figures for papers

## Common Workflows

### Workflow 1: Run Experiment with Full Monitoring

```python
# 1. Set up experiment
experiment_id = generate_experiment_id("wave_rgd_qfe")
dirs = create_experiment_directory("experiments", experiment_id)

# 2. Initialize monitoring
checkpoint_mgr = CheckpointManager(dirs['root'], save_interval=1000)
metrics_logger = MetricsLogger(dirs['logs'], log_interval=10)
viz_manager = VisualizationManager(dirs['visualizations'], viz_interval=1000)
config_tracker = ConfigTracker(dirs['root'])

# 3. Save configuration
config_tracker.save_config(config, model, dataset_info)

# 4. Training loop with monitoring
for step in range(num_steps):
    # ... training code ...
    
    if metrics_logger.should_log(step):
        metrics_logger.log_metrics(step, metrics)
    
    if checkpoint_mgr.should_checkpoint(step):
        checkpoint_mgr.save_checkpoint(step, model, optimizer, loss_history, config)
    
    if viz_manager.should_visualize(step):
        viz_manager.generate_training_plots(step, loss_history, metrics)
        viz_manager.generate_model_plots(step, model)

# 5. Save final results
config_tracker.save_results(final_metrics, best_checkpoint, generation_samples)
```

### Workflow 2: Resume Interrupted Training

```python
# 1. Load latest checkpoint
checkpoint = checkpoint_mgr.load_latest_checkpoint()

if checkpoint:
    # 2. Restore state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1
    loss_history = checkpoint['loss_history']
    
    print(f"✓ Resumed from step {checkpoint['step']}")
else:
    # 3. Start fresh
    start_step = 0
    loss_history = []
    print("✓ Starting new training")

# 4. Continue training
for step in range(start_step, num_steps):
    # ... training code ...
```

### Workflow 3: Generate Papers from Multiple Experiments

```bash
# 1. Run experiments
python spectral_gpt/wave_experiments.py --config baseline
python spectral_gpt/wave_experiments.py --config rgd_only
python spectral_gpt/wave_experiments.py --config qfe_only
python spectral_gpt/wave_experiments.py --config full_physics

# 2. Generate papers
python spectral_gpt/generate_paper.py --type both --output experiments/paper

# 3. Review generated papers
open experiments/paper/spectral_gpt_intuitive_guide.pdf
open experiments/paper/spectral_gpt_technical_paper.pdf
```

### Workflow 4: Compare Experiments

```python
# 1. Parse results
parser = ResultsParser("spectral_gpt/experiment_results/results.txt")
experiments = parser.parse()

# 2. Aggregate results
aggregator = ResultsAggregator()
aggregator.load_from_parser(parser)

# 3. Generate comparison table
print(aggregator.generate_comparison_table())

# 4. Get summary statistics
summary = aggregator.generate_summary_statistics()
print(f"Mean val loss: {summary['val_loss']['mean']:.4f}")
print(f"Best val loss: {summary['val_loss']['min']:.4f}")

# 5. Compare specific experiments
comparison = aggregator.compare_experiments("Standard Transformer", "Wave GPT")
print(f"Loss improvement: {comparison['val_loss_diff']:.4f}")
```

## Troubleshooting

### Common Issues

1. **Checkpoint corruption** - System automatically tries previous checkpoints
2. **Disk full** - Checkpoints are skipped with warning, training continues
3. **Matplotlib errors** - Visualizations are skipped, training continues
4. **Pandoc not found** - Papers generated as markdown only
5. **Missing experiment data** - Papers generated with available data, warnings for missing sections

See individual documentation files for detailed troubleshooting guides.

## Requirements

### Core Requirements

```bash
pip install torch numpy matplotlib
```

### Paper Generation Requirements

```bash
pip install pygments  # For code syntax highlighting

# Install pandoc for PDF rendering
# macOS: brew install pandoc
# Ubuntu: sudo apt-get install pandoc texlive-latex-base
```

## Examples

### Example 1: Minimal Monitoring

```python
from spectral_gpt.monitoring import CheckpointManager

checkpoint_mgr = CheckpointManager("experiments/my_exp")

for step in range(10000):
    # ... training ...
    
    if checkpoint_mgr.should_checkpoint(step):
        checkpoint_mgr.save_checkpoint(step, model, optimizer, [], {})
```

### Example 2: Full Monitoring

See [Monitoring System README](MONITORING_SYSTEM_README.md) for complete example.

### Example 3: Generate Papers

See [Paper Generation README](PAPER_GENERATION_README.md) for complete example.

## Contributing

To add new documentation:

1. Create markdown file in `spectral_gpt/` directory
2. Add link to this index
3. Follow existing documentation structure
4. Include code examples and troubleshooting

## License

See main project LICENSE file.

## Support

For issues or questions:

1. Check the relevant documentation file
2. Review troubleshooting sections
3. Check existing issues on GitHub
4. Open a new issue with detailed description

## Version History

- **v1.0** - Initial documentation release
  - Monitoring System README
  - Paper Generation README
  - Documentation Index

---

**Last Updated:** December 9, 2024
