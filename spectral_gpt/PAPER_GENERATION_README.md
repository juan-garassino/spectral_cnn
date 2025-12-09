# Paper Generation Infrastructure

This document describes the paper generation infrastructure for Spectral GPT, which automatically generates comprehensive academic documentation from experiment results.

## Overview

The paper generation system creates two types of documents:

1. **Intuitive Guide**: Visual, conceptual explanations for understanding
2. **Technical Paper**: Mathematical, rigorous documentation for reproducibility

## Features

- ✅ **Automatic markdown generation** from experiment data
- ✅ **Figure referencing system** with automatic numbering
- ✅ **Two-level documentation** (intuitive + technical)
- ✅ **Layer-by-layer architecture comparisons**
- ✅ **Fitting analysis** explaining convergence
- ✅ **PDF rendering** via pandoc (optional)
- ✅ **Template-based generation** for different venues

## Quick Start

### Basic Usage

```python
from spectral_gpt.paper_generator import SpectralGPTPaperGenerator

# Initialize generator
generator = SpectralGPTPaperGenerator(
    output_dir="experiments/paper",
    experiments_base_dir="experiments"
)

# Generate intuitive guide
guide_path = generator.generate_intuitive_guide(
    experiments=["exp_001", "exp_002"]
)

# Generate technical paper
paper_path = generator.generate_technical_paper(
    experiments=["exp_001", "exp_002"],
    template="arxiv"
)

# Render to PDF (requires pandoc)
pdf_path = generator.render_to_pdf(paper_path)
```

### Running the Demo

```bash
python demo/demo_paper_generator.py
```

This will generate sample documents in `experiments/paper/`.

## Generated Documents

### Intuitive Guide

The intuitive guide includes:

- **Visual Introduction**: Tokens as particles vs waves
- **Layer-by-Layer Comparison**: Side-by-side architecture diagrams
- **Convergence Explanation**: Why different architectures achieve similar loss
- **Wave Properties**: Intuitive understanding of frequency, phase, harmonics
- **Architecture Differences**: Parameter counts, complexity, when to use each

**Target Audience**: Researchers, practitioners, students who want conceptual understanding

**Length**: ~10-15 pages

**Tone**: Conversational, explanatory, visual

### Technical Paper

The technical paper includes:

- **Abstract**: Concise summary of contributions
- **Introduction**: Problem statement, motivation, contributions
- **Related Work**: Fourier operators, implicit representations, physics-informed networks
- **Mathematical Formulation**: Wave packet embeddings, interference attention, RGD, QFE
- **Architecture Details**: Complete specifications, parameter counts, complexity analysis
- **Experimental Methodology**: Dataset, training config, ablation design
- **Results**: Main results, convergence analysis, ablation studies
- **Analysis**: Frequency analysis, optimization trajectories, spectral bias
- **Discussion**: Limitations, when to use, future directions
- **Conclusion**: Summary of contributions

**Target Audience**: Researchers who want to reproduce, extend, or rigorously evaluate

**Length**: ~20-30 pages

**Tone**: Formal, academic, mathematical

## Architecture

### Class Hierarchy

```
PaperGenerator (base class)
    ├── generate_intuitive_guide()
    ├── generate_technical_paper()
    ├── generate_layer_comparison()
    ├── generate_fitting_analysis()
    ├── generate_abstract()
    ├── generate_methods_section()
    ├── generate_results_section()
    ├── generate_ablation_analysis()
    └── render_to_pdf()

SpectralGPTPaperGenerator (concrete implementation)
    └── Implements all abstract methods for Spectral GPT
```

### Key Components

1. **Figure Management**: Automatic numbering and referencing
2. **Markdown Utilities**: File writing, formatting
3. **Experiment Loading**: Config, results, metrics from JSONL
4. **Pandoc Integration**: PDF rendering with LaTeX

## Customization

### Adding New Sections

To add a new section to the technical paper:

```python
def _generate_custom_section(self) -> str:
    md = "## Custom Section\n\n"
    md += "Your content here...\n\n"
    return md

# Add to generate_technical_paper():
md += self._generate_custom_section()
```

### Customizing Templates

The generator supports different templates (arxiv, neurips, icml):

```python
paper_path = generator.generate_technical_paper(
    experiments=["exp_001"],
    template="neurips"  # or "icml", "arxiv"
)
```

### Adding Figures

```python
# Add figure with automatic numbering
fig_ref = generator._add_figure_reference(
    "my_figure.png",
    "This is my figure caption"
)
md += fig_ref
```

## PDF Generation

### Requirements

PDF generation requires [pandoc](https://pandoc.org/installing.html):

```bash
# macOS
brew install pandoc

# Ubuntu/Debian
sudo apt-get install pandoc

# Windows
choco install pandoc
```

### Rendering

```python
# Check if pandoc is available
if generator._check_pandoc_available():
    pdf_path = generator.render_to_pdf(markdown_file)
else:
    print("Pandoc not available - install from https://pandoc.org")
```

### PDF Options

The generator uses these pandoc options:

- `--pdf-engine=pdflatex`: LaTeX engine
- `--toc`: Table of contents
- `--number-sections`: Numbered sections
- `-V geometry:margin=1in`: 1-inch margins
- `-V fontsize=11pt`: 11-point font

## Examples

### Example 1: Generate from Existing Experiments

```python
# Load experiments from monitoring infrastructure
experiments = [
    "experiments/full_monitoring_demo_20251209_141844_708b506",
    "experiments/another_experiment_20251209_150000_abc1234"
]

generator = SpectralGPTPaperGenerator("experiments/paper")
paper_path = generator.generate_technical_paper(experiments)
```

### Example 2: Generate Layer Comparison

```python
wave_info = {
    'num_parameters': 67_473_706,
    'num_waves': 8,
    'num_harmonics': 3
}

standard_info = {
    'num_parameters': 52_892_160
}

comparison = generator.generate_layer_comparison(wave_info, standard_info)
print(comparison)
```

### Example 3: Generate Fitting Analysis

```python
exp_data = [
    {
        'name': 'Standard Transformer',
        'loss_history': [8.3, 6.5, 5.7, 4.9, 4.44],
        'final_loss': 4.44
    },
    {
        'name': 'Spectral GPT',
        'loss_history': [7.9, 6.4, 5.6, 4.8, 4.48],
        'final_loss': 4.48
    }
]

fitting = generator.generate_fitting_analysis(exp_data)
print(fitting)
```

## Integration with Monitoring

The paper generator integrates seamlessly with the monitoring infrastructure:

```python
from spectral_gpt.monitoring import ConfigTracker, MetricsLogger
from spectral_gpt.paper_generator import SpectralGPTPaperGenerator

# After training with monitoring
experiment_dir = "experiments/my_experiment"

# Generate paper from monitored experiment
generator = SpectralGPTPaperGenerator("experiments/paper")
paper_path = generator.generate_technical_paper([experiment_dir])
```

The generator automatically loads:
- `config.json`: Experiment configuration
- `results.json`: Final metrics and results
- `logs/metrics.jsonl`: Training metrics over time

## Testing

Run tests:

```bash
python tests/test_paper_generator.py
```

Tests cover:
- Initialization
- Figure referencing
- Markdown file writing
- Intuitive guide generation
- Technical paper generation

## File Structure

```
spectral_gpt/
├── paper_generator.py          # Main implementation
└── PAPER_GENERATION_README.md  # This file

experiments/
└── paper/                      # Generated papers
    ├── spectral_gpt_intuitive_guide.md
    ├── spectral_gpt_technical_paper.md
    └── figures/                # Referenced figures

demo/
└── demo_paper_generator.py    # Demo script

tests/
└── test_paper_generator.py    # Unit tests
```

## Future Enhancements

Potential improvements:

1. **Interactive Visualizations**: Plotly/Bokeh for interactive exploration
2. **Animated Diagrams**: GIFs showing forward pass through architectures
3. **Code Extraction**: Automatic extraction of code snippets from source
4. **Multi-Language Support**: Generate papers in multiple languages
5. **Version Control**: Track changes to generated papers
6. **Collaborative Editing**: Integration with Overleaf or similar
7. **Citation Management**: Automatic bibliography generation
8. **Figure Generation**: Automatic creation of plots from experiment data

## Troubleshooting

### Issue: Pandoc not found

**Solution**: Install pandoc from https://pandoc.org/installing.html

### Issue: PDF rendering fails

**Solution**: Check pandoc version and LaTeX installation:

```bash
pandoc --version
pdflatex --version
```

### Issue: Figures not found

**Solution**: Ensure figure files are in the `figures/` subdirectory:

```python
# Copy figures to output directory
import shutil
shutil.copy("my_figure.png", "experiments/paper/figures/")
```

### Issue: Experiment data not loading

**Solution**: Verify experiment directory structure:

```
experiments/my_experiment/
├── config.json
├── results.json
└── logs/
    └── metrics.jsonl
```

## Contributing

To add new features:

1. Extend `PaperGenerator` base class
2. Implement new methods in `SpectralGPTPaperGenerator`
3. Add tests in `tests/test_paper_generator.py`
4. Update this README

## License

Same as parent project.

## Contact

For questions or issues, please open an issue on the project repository.
