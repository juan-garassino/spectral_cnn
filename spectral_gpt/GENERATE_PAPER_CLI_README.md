# Paper Generation CLI

Command-line interface for generating Spectral GPT documentation from experiment results.

## Overview

The `generate_paper.py` script provides a convenient CLI for generating two types of documentation:

1. **Intuitive Guide**: Visual, conceptual explanations for understanding the architecture
2. **Technical Paper**: Mathematical, rigorous documentation for reproducibility

## Installation

No additional installation required beyond the main project dependencies. The CLI uses the existing `paper_generator.py` module.

## Basic Usage

### Generate Both Documents (Default)

```bash
python spectral_gpt/generate_paper.py
```

This will:
- Discover all experiments in `experiments/` directory
- Generate both intuitive guide and technical paper
- Output markdown files to `experiments/paper/`

### Generate Intuitive Guide Only

```bash
python spectral_gpt/generate_paper.py --type intuitive
```

### Generate Technical Paper Only

```bash
python spectral_gpt/generate_paper.py --type technical
```

## Advanced Options

### Specify Experiments

By default, the CLI discovers all experiments in the `experiments/` directory. You can specify specific experiments:

```bash
python spectral_gpt/generate_paper.py --experiments exp1 exp2 exp3
```

### Custom Output Directory

```bash
python spectral_gpt/generate_paper.py --output my_papers/
```

### Generate PDF (Requires Pandoc)

```bash
python spectral_gpt/generate_paper.py --format pdf
```

Or generate both markdown and PDF:

```bash
python spectral_gpt/generate_paper.py --format both
```

**Note**: PDF generation requires [pandoc](https://pandoc.org/installing.html) to be installed. If pandoc is not available, the CLI will generate markdown only and display a warning.

### Change Paper Template

For technical papers, you can specify different templates:

```bash
python spectral_gpt/generate_paper.py --type technical --template neurips
```

Available templates:
- `arxiv` (default)
- `neurips`
- `icml`

### Verbose Output

Enable detailed progress information:

```bash
python spectral_gpt/generate_paper.py --verbose
```

## Complete Examples

### Example 1: Generate Everything with Verbose Output

```bash
python spectral_gpt/generate_paper.py \
  --type both \
  --format both \
  --output papers/ \
  --verbose
```

### Example 2: Technical Paper for Specific Experiments

```bash
python spectral_gpt/generate_paper.py \
  --type technical \
  --experiments experiments/exp_001 experiments/exp_002 \
  --template arxiv \
  --format pdf \
  --output submission/
```

### Example 3: Quick Intuitive Guide

```bash
python spectral_gpt/generate_paper.py \
  --type intuitive \
  --output quick_guide/
```

## Command-Line Options

| Option | Choices | Default | Description |
|--------|---------|---------|-------------|
| `--type` | `intuitive`, `technical`, `both` | `both` | Type of documentation to generate |
| `--experiments` | List of paths | Auto-discover | Specific experiment directories to include |
| `--experiments-dir` | Path | `experiments` | Base directory containing experiments |
| `--output` | Path | `experiments/paper` | Output directory for generated papers |
| `--format` | `markdown`, `pdf`, `both` | `markdown` | Output format |
| `--template` | `arxiv`, `neurips`, `icml` | `arxiv` | Paper template (technical papers only) |
| `--verbose` | Flag | Off | Enable verbose output |

## Output Structure

After running the CLI, your output directory will contain:

```
output_directory/
├── spectral_gpt_intuitive_guide.md    # Intuitive guide (if generated)
├── spectral_gpt_intuitive_guide.pdf   # PDF version (if --format pdf/both)
├── spectral_gpt_technical_paper.md    # Technical paper (if generated)
├── spectral_gpt_technical_paper.pdf   # PDF version (if --format pdf/both)
└── figures/                            # Referenced figures
    ├── fig1_architecture.png
    ├── fig2_results.png
    └── ...
```

## Experiment Discovery

The CLI automatically discovers experiments by looking for directories containing:
- `config.json` - Experiment configuration
- `results.json` - Final results
- `logs/metrics.jsonl` - Training metrics

If no experiments are found, the CLI will generate documentation without experiment-specific data.

## Error Handling

The CLI handles common errors gracefully:

- **No experiments found**: Generates documentation without experiment data
- **Invalid experiment paths**: Skips invalid paths and continues with valid ones
- **Pandoc not available**: Generates markdown only and displays installation instructions
- **Missing data files**: Uses available data and continues

## Integration with Workflow

### After Training

```bash
# Train your model
python spectral_gpt/wave_experiments.py

# Generate documentation
python spectral_gpt/generate_paper.py --verbose
```

### For Paper Submission

```bash
# Generate technical paper with PDF
python spectral_gpt/generate_paper.py \
  --type technical \
  --format pdf \
  --template arxiv \
  --output submission/
```

### For Presentations

```bash
# Generate intuitive guide for presentations
python spectral_gpt/generate_paper.py \
  --type intuitive \
  --output presentation/
```

## Troubleshooting

### PDF Generation Fails

**Problem**: `Warning: pandoc not available`

**Solution**: Install pandoc:
- macOS: `brew install pandoc`
- Ubuntu: `apt-get install pandoc`
- Windows: Download from https://pandoc.org/installing.html

### No Experiments Found

**Problem**: `Warning: No experiments found in experiments/`

**Solution**: 
1. Check that experiments directory exists
2. Verify experiments have `config.json`, `results.json`, or `logs/` directory
3. Use `--experiments-dir` to specify a different location

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'paper_generator'`

**Solution**: Run the script from the project root directory:
```bash
cd /path/to/spectral_cnn
python spectral_gpt/generate_paper.py
```

## Development

### Running Tests

```bash
# Test the CLI
python -m pytest tests/test_generate_paper_cli.py -v

# Test the underlying paper generator
python -m pytest tests/test_paper_generator.py -v
```

### Adding New Templates

To add a new paper template:

1. Modify `paper_generator.py` to handle the new template
2. Add the template choice to the CLI argparse options
3. Update this README with the new template

## See Also

- [Paper Generator Module](paper_generator.py) - Underlying implementation
- [Code Extractor](code_extractor.py) - Code snippet extraction
- [Monitoring System](MONITORING_README.md) - Experiment tracking
