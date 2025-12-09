"""
Tests for code extraction functionality.
"""

import pytest
import sys
from pathlib import Path

# Add spectral_gpt to path
sys.path.insert(0, str(Path(__file__).parent.parent / "spectral_gpt"))

from code_extractor import CodeExtractor, CodeSnippet
from code_examples import SpectralGPTCodeExamples


class TestCodeExtractor:
    """Test the CodeExtractor class."""
    
    def test_extract_class(self):
        """Test extracting a class from a Python file."""
        extractor = CodeExtractor()
        snippet = extractor.extract_class(
            "spectral_gpt/wave_gpt.py",
            "WavePacketEmbedding"
        )
        
        assert snippet is not None
        assert snippet.name == "WavePacketEmbedding"
        assert snippet.snippet_type == "class"
        assert "def __init__" in snippet.code
        assert "def forward" in snippet.code
    
    def test_extract_class_with_methods(self):
        """Test extracting a class with specific methods."""
        extractor = CodeExtractor()
        snippet = extractor.extract_class(
            "spectral_gpt/wave_gpt.py",
            "WavePacketEmbedding",
            include_methods=["forward"]
        )
        
        assert snippet is not None
        assert "def forward" in snippet.code
        assert "def __init__" in snippet.code  # __init__ always included
    
    def test_format_for_markdown(self):
        """Test formatting code for markdown."""
        extractor = CodeExtractor()
        snippet = extractor.extract_class(
            "spectral_gpt/wave_gpt.py",
            "WavePacketEmbedding",
            include_methods=["forward"]
        )
        
        assert snippet is not None
        markdown = extractor.format_for_markdown(snippet)
        
        assert markdown.startswith("```python")
        assert markdown.endswith("```")
        assert "WavePacketEmbedding" in markdown


class TestSpectralGPTCodeExamples:
    """Test the SpectralGPTCodeExamples class."""
    
    def test_get_wave_packet_embedding(self):
        """Test extracting WavePacketEmbedding."""
        examples = SpectralGPTCodeExamples()
        snippet = examples.get_wave_packet_embedding()
        
        assert snippet is not None
        assert snippet.name == "WavePacketEmbedding"
        assert "num_waves" in snippet.code
        assert "num_harmonics" in snippet.code
    
    def test_get_rgd_optimizer(self):
        """Test extracting RGD optimizer."""
        examples = SpectralGPTCodeExamples()
        snippet = examples.get_rgd_optimizer()
        
        assert snippet is not None
        assert snippet.name == "ResonantGradientDescent"
        assert "resonance" in snippet.code.lower()
    
    def test_get_qfe_loss(self):
        """Test extracting QFE loss."""
        examples = SpectralGPTCodeExamples()
        snippet = examples.get_qfe_loss()
        
        assert snippet is not None
        assert snippet.name == "QuantumFieldEntanglementLoss"
        assert "coherence" in snippet.code.lower()
    
    def test_get_api_usage_example(self):
        """Test generating API usage example."""
        examples = SpectralGPTCodeExamples()
        api_example = examples.get_api_usage_example()
        
        assert "WaveGPT" in api_example
        assert "ResonantGradientDescent" in api_example
        assert "QuantumFieldEntanglementLoss" in api_example
    
    def test_generate_code_appendix(self):
        """Test generating complete code appendix."""
        examples = SpectralGPTCodeExamples()
        appendix = examples.generate_code_appendix()
        
        assert "# Appendix: Code Examples" in appendix
        assert "Wave Packet Embedding" in appendix
        assert "Resonant Gradient Descent" in appendix
        assert "```python" in appendix


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
