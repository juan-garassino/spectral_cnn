# Spectral GPT Paper Generation System

Automatically generate comprehensive academic documentation from experiment results, including intuitive guides and technical papers.

## Overview

The paper generation system creates two levels of documentation:

1. **Intuitive Guide** - Visual, conceptual explanations for understanding the core ideas
2. **Technical Paper** - Mathematical, rigorous documentation for reproducibility

Both documents are generated from experiment results, code, and existing documentation, with automatic figure management and PDF rendering.

## Quick Start

### Basic Usage

```python
from spectral_gpt.paper_generator import SpectralGPTPaperGenerator
from spectral_gpt.monitoring import ResultsParser

# Parse experiment results
parser = ResultsParser("spectral_gpt/experiment_results/results.txt")
experiments = parser.parse()

# Initialize paper generator
generator = SpectralGPTPaperGenerator(output_dir="experiments/paper")

# Generate intuitive guide
guide_path = generator.generate_intuitive_guide(
    experiments=experiments
)
print(f"Generated intuitive guide: {guide_path}")

# Generate technical paper
paper_path = generator.generate_technical_paper(
    experiments=experiments,
    template="arxiv"
)
print(f"Generated technical paper: {paper_path}")

# Render to PDF (requires pandoc)
guide_pdf = generator.render_to_pdf(guide_path)
paper_pdf = generator.render_to_pdf(paper_path)
```

### Command-Line Interface

```bash
# Generate both intuitive guide and technical paper
python spectral_gpt/generate_paper.py --type both --output experiments/paper

# Generate only intuitive guide
python spectral_gpt/generate_paper.py --type intuitive

# Generate only technical paper
python spectral_gpt/generate_paper.py --type technical --template arxiv

# Generate with specific experiments
python spectral_gpt/generate_paper.py --experiments exp_001 exp_002 exp_003

# Generate markdown only (skip PDF)
python spectral_gpt/generate_paper.py --format markdown

# Generate PDF only (requires existing markdown)
python spectral_gpt/generate_paper.py --format pdf
```

## Components

### 1. PaperGenerator (Base Class)

The base class provides core functionality for paper generation.

#### Features

- **Markdown intermediate format** - Easy to edit and version control
- **Automatic figure numbering** - Consistent figure references
- **Pandoc integration** - Convert markdown to PDF
- **Template support** - Different formats for different venues

#### Usage

```python
from spectral_gpt.paper_generator import PaperGenerator

class MyPaperGenerator(PaperGenerator):
    def generate_intuitive_guide(self, experiments):
        # Custom implementation
        pass
    
    def generate_technical_paper(self, experiments, template="arxiv"):
        # Custom implementation
        pass

generator = MyPaperGenerator(output_dir="papers")
```

### 2. SpectralGPTPaperGenerator

Specialized generator for Spectral GPT documentation.

#### Features

- **Intuitive guide generation** - Visual explanations and comparisons
- **Technical paper generation** - Complete academic paper with all sections
- **Layer comparison** - Side-by-side architecture analysis
- **Fitting analysis** - Explains convergence to similar loss
- **Code extraction** - Automatic code snippet generation
- **Visualization integration** - Includes experiment plots

#### Configuration

```python
generator = SpectralGPTPaperGenerator(
    output_dir="experiments/paper"  # Output directory for papers and figures
)
```

### 3. ResultsParser

Parses experiment results from text files into structured format.

#### Features

- **Text parsing** - Extracts data from formatted text files
- **Structured output** - JSON format for easy processing
- **Experiment metadata** - Configuration, metrics, results
- **Ablation analysis** - Parses ablation study tables

#### Usage

```python
from spectral_gpt.monitoring import ResultsParser

# Parse results file
parser = ResultsParser("spectral_gpt/experiment_results/results.txt")
experiments = parser.parse()

# Access experiment data
for exp in experiments:
    print(f"Experiment: {exp['name']}")
    print(f"  Parameters: {exp['config']['parameters']}")
    print(f"  Val Loss: {exp['final_results']['val_loss']}")
    print(f"  Perplexity: {exp['final_results']['perplexity']}")

# Get specific experiment
standard_exp = parser.get_experiment_by_name("Standard Transformer")

# Save to JSON
parser.save_to_json("experiments/parsed_results.json")
```

### 4. ResultsAggregator

Aggregates and compares results from multiple experiments.

#### Features

- **Comparison tables** - Side-by-side experiment comparison
- **Summary statistics** - Mean, median, min, max, stdev
- **Ablation analysis** - Component contribution analysis
- **Visualization** - Comparison plots

#### Usage

```python
from spectral_gpt.monitoring import ResultsAggregator

# Create aggregator
aggregator = ResultsAggregator()

# Load from parser
aggregator.load_from_parser(parser)

# Or load from JSON
aggregator.load_from_json("experiments/parsed_results.json")

# Generate comparison table
table = aggregator.generate_comparison_table()
print(table)

# Get summary statistics
summary = aggregator.generate_summary_statistics()
print(f"Mean val loss: {summary['val_loss']['mean']:.4f}")
print(f"Best val loss: {summary['val_loss']['min']:.4f}")

# Compare two experiments
comparison = aggregator.compare_experiments("Standard Transformer", "Wave GPT")
print(f"Loss difference: {comparison['val_loss_diff']:.4f}")
```

### 5. CodeExtractor

Extracts code snippets from source files for documentation.

#### Features

- **Class/function extraction** - Extract specific code blocks
- **Syntax highlighting** - Pygments integration
- **Markdown formatting** - Ready for inclusion in papers
- **Pseudocode generation** - High-level algorithm descriptions

#### Usage

```python
from spectral_gpt.code_extractor import CodeExtractor

# Initialize extractor
extractor = CodeExtractor()

# Extract specific class
wave_embedding_code = extractor.extract_class(
    "spectral_gpt/wave_gpt.py",
    "WavePacketEmbedding"
)

# Extract specific function
rgd_code = extractor.extract_function(
    "spectral_gpt/physics_optim.py",
    "resonant_gradient_descent"
)

# Format for markdown
markdown_code = extractor.format_for_markdown(
    wave_embedding_code,
    language="python",
    caption="Wave Packet Embedding Implementation"
)

# Generate pseudocode
pseudocode = extractor.generate_pseudocode(wave_embedding_code)
```

## Generating Papers

### Intuitive Guide

The intuitive guide focuses on conceptual understanding with visual explanations.

#### Sections Included

1. **Visual Introduction**
   - Side-by-side comparison: Standard Transformer vs Wave-Native GPT
   - Animated diagrams showing wave interference vs vector operations
   - Intuitive explanation: "Tokens as particles vs tokens as waves"

2. **Layer-by-Layer Comparison**
   - Embedding layer: Lookup table vs wave packets
   - Attention layer: Dot product vs phase interference
   - Feed-forward layer: GELU vs wave activation
   - Visual representations of each layer's operation

3. **Why Different Architectures Achieve Similar Loss**
   - Loss landscape visualization
   - Multiple optimization paths
   - Inductive bias explanation
   - Convergence analysis

4. **Intuitive Understanding of Wave Properties**
   - Frequency: Global vs local patterns
   - Phase: Temporal relationships
   - Harmonics: Multi-scale features
   - Interference: Constructive and destructive patterns

5. **Real Architecture Differences**
   - Parameter count comparison
   - Computational complexity
   - Memory usage
   - Training dynamics

#### Example

```python
# Generate intuitive guide
guide_path = generator.generate_intuitive_guide(
    experiments=[
        standard_transformer_exp,
        wave_gpt_exp,
        rgd_only_exp
    ]
)

# Output: experiments/paper/spectral_gpt_intuitive_guide.md
```

### Technical Paper

The technical paper provides rigorous mathematical documentation.

#### Sections Included

1. **Abstract** - Concise summary of contributions and results

2. **Introduction**
   - Problem statement
   - Hypothesis
   - Contributions

3. **Related Work**
   - Fourier Neural Operators
   - Implicit Neural Representations
   - Complex-valued networks
   - Physics-informed neural networks

4. **Mathematical Formulation**
   - Wave packet embedding equations
   - Interference attention mechanism
   - RGD optimizer formulation
   - QFE loss function

5. **Architecture Details**
   - Complete model specification
   - Layer-by-layer parameter counts
   - Computational complexity (FLOPs)
   - Memory requirements

6. **Experimental Methodology**
   - Dataset details
   - Training procedure
   - Evaluation metrics
   - Ablation study design

7. **Results**
   - Main comparison table
   - Ablation study results
   - Statistical significance tests
   - Learning curves

8. **Analysis**
   - Frequency domain analysis
   - Optimization trajectories
   - Spectral bias
   - Phase coherence

9. **Discussion**
   - Limitations
   - When to use wave vs standard
   - Future directions

10. **Conclusion** - Summary and impact

11. **Appendix**
    - Complete hyperparameter tables
    - Additional ablation results
    - Implementation details
    - Code snippets

#### Example

```python
# Generate technical paper
paper_path = generator.generate_technical_paper(
    experiments=[
        standard_transformer_exp,
        wave_gpt_exp,
        rgd_only_exp,
        qfe_only_exp
    ],
    template="arxiv"  # or "neurips", "icml"
)

# Output: experiments/paper/spectral_gpt_technical_paper.md
```

### Templates

Different templates for different venues:

```python
# arXiv format (default)
paper_path = generator.generate_technical_paper(
    experiments=experiments,
    template="arxiv"
)

# NeurIPS format
paper_path = generator.generate_technical_paper(
    experiments=experiments,
    template="neurips"
)

# ICML format
paper_path = generator.generate_technical_paper(
    experiments=experiments,
    template="icml"
)
```

## Customizing Papers

### Adding New Sections

```python
class CustomPaperGenerator(SpectralGPTPaperGenerator):
    def generate_technical_paper(self, experiments, template="arxiv"):
        # Call parent implementation
        paper_path = super().generate_technical_paper(experiments, template)
        
        # Read existing content
        with open(paper_path, 'r') as f:
            content = f.read()
        
        # Add custom section before conclusion
        custom_section = """
## Custom Analysis

This section provides additional analysis specific to our use case.

### Custom Metric 1

Analysis of custom metric...

### Custom Metric 2

Analysis of another custom metric...
"""
        
        # Insert before conclusion
        content = content.replace("## Conclusion", custom_section + "\n## Conclusion")
        
        # Write back
        with open(paper_path, 'w') as f:
            f.write(content)
        
        return paper_path

# Use custom generator
generator = CustomPaperGenerator(output_dir="experiments/paper")
```

### Customizing Templates

Create a custom template file:

```python
# templates/custom_template.md
"""
---
title: "{title}"
author: "{authors}"
date: "{date}"
abstract: "{abstract}"
---

# Introduction

{introduction}

# Methods

{methods}

# Results

{results}

# Conclusion

{conclusion}
"""

# Use custom template
generator.add_template("custom", "templates/custom_template.md")
paper_path = generator.generate_technical_paper(
    experiments=experiments,
    template="custom"
)
```

### Adding Custom Figures

```python
# Generate custom figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
ax.set_title("Custom Analysis")
fig.savefig("experiments/paper/figures/custom_figure.png")

# Add to paper with automatic numbering
figure_md = generator._add_figure_reference(
    "custom_figure.png",
    "Custom analysis showing relationship between X and Y"
)

# Insert into paper content
# (This would be done within a custom generator method)
```

## Figure Management

### Automatic Figure Numbering

The system automatically numbers figures and maintains references:

```python
# First figure
fig1 = generator._add_figure_reference(
    "architecture_comparison.png",
    "Comparison of Standard Transformer and Wave-Native GPT architectures"
)
# Output: "Figure 1: Comparison of..."

# Second figure
fig2 = generator._add_figure_reference(
    "loss_curves.png",
    "Training loss curves for different configurations"
)
# Output: "Figure 2: Training loss curves..."

# Reference in text
text = f"As shown in {generator.figure_references['architecture_comparison.png']}, ..."
# Output: "As shown in Figure 1, ..."
```

### Figure Directory Structure

```
experiments/paper/
├── figures/
│   ├── architecture_comparison.png
│   ├── convergence_comparison.png
│   ├── frequency_heatmap.png
│   ├── frequency_spectrum_evolution.png
│   ├── harmonic_amplitudes.png
│   ├── interference_patterns.png
│   ├── loss_landscape_3d.png
│   ├── loss_landscape_contour.png
│   ├── parameter_breakdown.png
│   ├── phase_distribution.png
│   ├── standard_forward_pass.gif
│   └── wave_forward_pass.gif
├── spectral_gpt_intuitive_guide.md
├── spectral_gpt_intuitive_guide.pdf
├── spectral_gpt_technical_paper.md
└── spectral_gpt_technical_paper.pdf
```

## PDF Rendering

### Installing Pandoc

**macOS:**
```bash
brew install pandoc
brew install basictex  # For LaTeX support
```

**Ubuntu/Debian:**
```bash
sudo apt-get install pandoc
sudo apt-get install texlive-latex-base texlive-latex-extra
```

**Windows:**
Download from https://pandoc.org/installing.html

### Rendering Options

```python
# Basic rendering
pdf_path = generator.render_to_pdf("paper.md")

# Custom output path
pdf_path = generator.render_to_pdf("paper.md", output_pdf="custom_name.pdf")

# Check if pandoc is available
if generator._check_pandoc_available():
    pdf_path = generator.render_to_pdf("paper.md")
else:
    print("Pandoc not available, markdown only")
```

### Advanced Pandoc Options

```python
# Custom pandoc command
import subprocess

subprocess.run([
    'pandoc',
    'paper.md',
    '-o', 'paper.pdf',
    '--pdf-engine=xelatex',  # Use XeLaTeX instead of pdflatex
    '--toc',  # Table of contents
    '--toc-depth=3',  # TOC depth
    '--number-sections',  # Number sections
    '-V', 'geometry:margin=1in',  # Margins
    '-V', 'fontsize=11pt',  # Font size
    '-V', 'documentclass=article',  # Document class
    '--highlight-style=tango',  # Code highlighting style
    '--bibliography=references.bib',  # Bibliography
    '--csl=ieee.csl'  # Citation style
])
```

## Troubleshooting

### Paper Generation Issues

**Problem: No experiments found**

```
Error: No experiments provided for paper generation
```

**Solution:** Ensure experiments are parsed correctly:

```python
parser = ResultsParser("spectral_gpt/experiment_results/results.txt")
experiments = parser.parse()

if not experiments:
    print("No experiments found in results file")
else:
    generator.generate_intuitive_guide(experiments)
```

**Problem: Missing experiment data**

```
Warning: Experiment 'Wave GPT' missing final_results
```

**Solution:** Check that results file contains all required data:

```python
for exp in experiments:
    if not exp.get('final_results'):
        print(f"Warning: {exp['name']} missing final_results")
    if not exp.get('training_metrics'):
        print(f"Warning: {exp['name']} missing training_metrics")
```

### Figure Issues

**Problem: Figure not found**

```
Warning: Figure 'architecture_comparison.png' not found
```

**Solution:** Ensure figures are in the correct directory:

```python
import shutil

# Copy figures to paper directory
shutil.copy(
    "experiments/exp_001/visualizations/architecture.png",
    "experiments/paper/figures/architecture_comparison.png"
)
```

**Problem: Figure numbering incorrect**

**Solution:** Reset figure counter before generating new document:

```python
generator._reset_figure_counter()
guide_path = generator.generate_intuitive_guide(experiments)
```

### PDF Rendering Issues

**Problem: Pandoc not found**

```
Warning: pandoc not available. Skipping PDF generation.
```

**Solution:** Install pandoc (see "Installing Pandoc" section above)

**Problem: LaTeX errors during PDF generation**

```
Warning: pandoc failed with error:
! LaTeX Error: File 'unicode-math.sty' not found.
```

**Solution:** Install complete LaTeX distribution:

```bash
# macOS
brew install --cask mactex

# Ubuntu
sudo apt-get install texlive-full

# Or use minimal installation
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

**Problem: PDF generation timeout**

```
Warning: Failed to generate PDF: timeout
```

**Solution:** Increase timeout or generate markdown only:

```python
# Increase timeout
subprocess.run(cmd, timeout=300)  # 5 minutes

# Or skip PDF generation
guide_path = generator.generate_intuitive_guide(experiments)
# Don't call render_to_pdf()
```

### Code Extraction Issues

**Problem: Code extraction module not available**

```
Warning: code_examples module not available. Code extraction disabled.
```

**Solution:** Ensure code_examples.py is in the correct location:

```python
# Check if file exists
import os
if not os.path.exists("spectral_gpt/code_examples.py"):
    print("code_examples.py not found")

# Or disable code extraction
CODE_EXTRACTION_AVAILABLE = False
```

**Problem: Syntax highlighting not working**

**Solution:** Install pygments:

```bash
pip install pygments
```

## Best Practices

### 1. Generate Papers After Experiments Complete

```python
# Run experiments first
python spectral_gpt/wave_experiments.py

# Then generate papers
python spectral_gpt/generate_paper.py --type both
```

### 2. Version Control Papers

```bash
# Add papers to git
git add experiments/paper/*.md
git add experiments/paper/figures/*.png
git commit -m "Add generated papers for experiment batch 1"

# Don't commit PDFs (they're large and can be regenerated)
echo "*.pdf" >> .gitignore
```

### 3. Regenerate Papers When Experiments Change

```python
# After running new experiments
parser = ResultsParser("spectral_gpt/experiment_results/results.txt")
experiments = parser.parse()

# Regenerate papers with updated results
generator.generate_intuitive_guide(experiments)
generator.generate_technical_paper(experiments)
```

### 4. Review Generated Papers

```python
# Generate markdown first
guide_path = generator.generate_intuitive_guide(experiments)

# Review and edit markdown
# (Open in text editor and make changes)

# Then generate PDF
pdf_path = generator.render_to_pdf(guide_path)
```

### 5. Use Descriptive Figure Names

```python
# Good: Descriptive names
generator._add_figure_reference(
    "wave_gpt_frequency_spectrum_step_5000.png",
    "Frequency spectrum of Wave GPT at training step 5000"
)

# Bad: Generic names
generator._add_figure_reference(
    "fig1.png",
    "Some plot"
)
```

## Advanced Usage

### Batch Paper Generation

```python
# Generate papers for multiple experiment batches
experiment_batches = [
    "spectral_gpt/experiment_results/batch1_results.txt",
    "spectral_gpt/experiment_results/batch2_results.txt",
    "spectral_gpt/experiment_results/batch3_results.txt"
]

for i, results_file in enumerate(experiment_batches):
    parser = ResultsParser(results_file)
    experiments = parser.parse()
    
    output_dir = f"experiments/paper_batch_{i+1}"
    generator = SpectralGPTPaperGenerator(output_dir=output_dir)
    
    generator.generate_intuitive_guide(experiments)
    generator.generate_technical_paper(experiments)
```

### Incremental Paper Updates

```python
# Load existing paper
with open("experiments/paper/spectral_gpt_technical_paper.md", 'r') as f:
    existing_content = f.read()

# Parse new experiments
parser = ResultsParser("spectral_gpt/experiment_results/new_results.txt")
new_experiments = parser.parse()

# Update results section only
# (Custom implementation to replace specific sections)
updated_content = update_results_section(existing_content, new_experiments)

# Save updated paper
with open("experiments/paper/spectral_gpt_technical_paper.md", 'w') as f:
    f.write(updated_content)
```

### Integration with CI/CD

```yaml
# .github/workflows/generate_papers.yml
name: Generate Papers

on:
  push:
    paths:
      - 'spectral_gpt/experiment_results/**'

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          sudo apt-get install pandoc texlive-latex-base
      
      - name: Generate papers
        run: |
          python spectral_gpt/generate_paper.py --type both --format markdown
      
      - name: Commit papers
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add experiments/paper/*.md
          git commit -m "Auto-generate papers" || echo "No changes"
          git push
```

## See Also

- [Monitoring System Documentation](MONITORING_SYSTEM_README.md) - Experiment monitoring and checkpointing
- [Code Extraction README](CODE_EXTRACTION_README.md) - Extracting code snippets for documentation
- [Paper Visualizations README](PAPER_VISUALIZATIONS_README.md) - Generating figures for papers
- [Configuration Guide](../docs/configuration_guide.md) - Detailed configuration options
