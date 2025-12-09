# Code Extraction for Documentation

This module provides automated code extraction and formatting tools for generating academic documentation with real implementation examples.

## Overview

The code extraction system consists of two main components:

1. **CodeExtractor**: Low-level tool for extracting and formatting code snippets from Python files
2. **SpectralGPTCodeExamples**: High-level interface for extracting Spectral GPT specific components

## Features

- ✅ Extract classes, functions, and methods from Python source files
- ✅ Syntax highlighting with pygments (optional)
- ✅ Format code for markdown documentation
- ✅ Generate simplified pseudocode
- ✅ Extract API usage examples
- ✅ Generate complete code appendices for papers
- ✅ Integration with paper generator

## Installation

The code extraction module requires no additional dependencies beyond the base project requirements. For syntax highlighting support, install pygments:

```bash
pip install pygments
```

## Usage

### Basic Code Extraction

```python
from code_extractor import CodeExtractor

# Create extractor
extractor = CodeExtractor()

# Extract a class
snippet = extractor.extract_class(
    "spectral_gpt/wave_gpt.py",
    "WavePacketEmbedding",
    include_methods=["forward"]  # Optional: specific methods only
)

# Format for markdown
markdown = extractor.format_for_markdown(snippet, max_lines=50)
print(markdown)
```

### Spectral GPT Specific Examples

```python
from code_examples import SpectralGPTCodeExamples

# Create examples manager
examples = SpectralGPTCodeExamples()

# Extract key components
wave_embedding = examples.get_wave_packet_embedding()
rgd_optimizer = examples.get_rgd_optimizer()
qfe_loss = examples.get_qfe_loss()

# Get API usage examples
api_example = examples.get_api_usage_example()
training_example = examples.get_training_example()

# Generate complete code appendix
appendix = examples.generate_code_appendix(
    output_file="experiments/paper/code_appendix.md"
)
```

### Integration with Paper Generator

```python
from paper_generator import SpectralGPTPaperGenerator

# Create paper generator (code extraction enabled automatically)
generator = SpectralGPTPaperGenerator(
    output_dir="experiments/paper",
    experiments_base_dir="experiments"
)

# Generate code appendix
appendix = generator.generate_code_appendix(
    output_file="experiments/paper/appendix.md"
)

# The paper generator will automatically use real code extraction
# when generating technical papers
```

## Extracted Components

The system can extract the following key Spectral GPT components:

### 1. Wave Packet Embedding
- Class: `WavePacketEmbedding`
- File: `spectral_gpt/wave_gpt.py`
- Key methods: `__init__`, `forward`

### 2. Interference Attention
- Class: `WaveAttention`
- File: `spectral_gpt/wave_gpt.py`
- Key methods: `forward`

### 3. Resonant Gradient Descent (RGD)
- Class: `ResonantGradientDescent`
- File: `spectral_gpt/physics_optim.py`
- Key methods: `__init__`, `step`, `_compute_resonance_factor`

### 4. Quantum Field Entanglement (QFE) Loss
- Class: `QuantumFieldEntanglementLoss`
- File: `spectral_gpt/physics_optim.py`
- Key methods: `forward`

### 5. API Usage Examples
- High-level API usage
- Complete training example
- Model configuration
- Generation examples

## Output Format

The code extraction system generates markdown-formatted code blocks suitable for academic papers:

```markdown
# Appendix: Code Examples

## A.1 Wave Packet Embedding

The core innovation of Spectral GPT is representing tokens as wave packets...

\`\`\`python
# WavePacketEmbedding (class)
# Embed tokens as wave packets with HARMONICS.

class WavePacketEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, num_waves=16, num_harmonics=4):
        ...
\`\`\`
```

## Demo

Run the demo script to see all features in action:

```bash
python demo/demo_code_extraction.py
```

This will:
1. Extract key components from the codebase
2. Format them for markdown
3. Generate API usage examples
4. Create a complete code appendix
5. Demonstrate integration with paper generator

## Testing

Run the test suite:

```bash
pytest tests/test_code_extractor.py -v
```

Tests cover:
- Class extraction
- Method filtering
- Markdown formatting
- Spectral GPT specific components
- API example generation
- Code appendix generation

## Architecture

```
code_extractor.py
├── CodeSnippet (dataclass)
│   └── Metadata for extracted code
└── CodeExtractor (class)
    ├── extract_class()
    ├── extract_function()
    ├── format_for_markdown()
    ├── generate_pseudocode()
    └── highlight_code()

code_examples.py
└── SpectralGPTCodeExamples (class)
    ├── get_wave_packet_embedding()
    ├── get_interference_attention()
    ├── get_rgd_optimizer()
    ├── get_qfe_loss()
    ├── get_api_usage_example()
    ├── get_training_example()
    └── generate_code_appendix()

paper_generator.py (integration)
└── SpectralGPTPaperGenerator
    ├── code_examples (attribute)
    ├── generate_methods_section() [uses code extraction]
    └── generate_code_appendix()
```

## Benefits

1. **Reproducibility**: Real code from implementation, not simplified examples
2. **Maintainability**: Code examples stay in sync with implementation
3. **Automation**: No manual copying of code snippets
4. **Consistency**: Uniform formatting across all documentation
5. **Completeness**: Can extract entire classes or specific methods

## Limitations

- Currently supports Python only
- Requires valid Python syntax (no partial code)
- Pygments required for syntax highlighting (optional)
- AST-based extraction may not work with very complex metaprogramming

## Future Enhancements

- [ ] Support for other languages (C++, Julia)
- [ ] Interactive code examples with execution
- [ ] Automatic docstring extraction and formatting
- [ ] Code diff visualization for ablation studies
- [ ] Integration with Jupyter notebooks
- [ ] Automatic complexity analysis (Big-O notation)

## Requirements Validation

This implementation satisfies the following requirements from the spec:

- ✅ **Requirement 6.1**: Code snippets for wave packet embeddings, interference attention, and physics-informed optimization
- ✅ **Requirement 6.2**: Both high-level API usage and low-level implementation details
- ✅ **Requirement 6.3**: Pseudocode generation capability
- ✅ **Requirement 6.4**: Command-line examples and usage instructions

## Files

- `spectral_gpt/code_extractor.py` - Core extraction functionality
- `spectral_gpt/code_examples.py` - Spectral GPT specific examples
- `tests/test_code_extractor.py` - Test suite
- `demo/demo_code_extraction.py` - Demonstration script
- `experiments/paper/code_appendix.md` - Generated code appendix

## Contributing

When adding new components to Spectral GPT:

1. Ensure classes and functions have docstrings
2. Use clear, descriptive names
3. Add extraction methods to `SpectralGPTCodeExamples` if needed
4. Update tests to cover new components
5. Regenerate code appendix with `python spectral_gpt/code_examples.py`
