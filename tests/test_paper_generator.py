"""
Tests for paper generation infrastructure
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

from spectral_gpt.paper_generator import PaperGenerator, SpectralGPTPaperGenerator


def test_paper_generator_initialization():
    """Test PaperGenerator initialization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        assert generator.output_dir.exists()
        assert generator.figures_dir.exists()
        assert generator.figure_counter == 0


def test_figure_referencing():
    """Test figure reference system"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        # Add first figure
        ref1 = generator._add_figure_reference("test_fig1.png", "Test figure 1")
        assert "Figure 1" in ref1
        assert generator.figure_counter == 1
        
        # Add second figure
        ref2 = generator._add_figure_reference("test_fig2.png", "Test figure 2")
        assert "Figure 2" in ref2
        assert generator.figure_counter == 2
        
        # Reset counter
        generator._reset_figure_counter()
        assert generator.figure_counter == 0


def test_markdown_file_writing():
    """Test markdown file writing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        content = "# Test Document\n\nThis is a test."
        filepath = Path(tmpdir) / "test.md"
        
        generator._write_markdown_file(filepath, content)
        
        assert filepath.exists()
        with open(filepath, 'r') as f:
            assert f.read() == content


def test_generate_intuitive_guide():
    """Test intuitive guide generation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        output_path = generator.generate_intuitive_guide(experiments=[])
        
        assert Path(output_path).exists()
        with open(output_path, 'r') as f:
            content = f.read()
            assert "Spectral GPT" in content
            assert "Intuitive Guide" in content


def test_generate_technical_paper():
    """Test technical paper generation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        output_path = generator.generate_technical_paper(experiments=[])
        
        assert Path(output_path).exists()
        with open(output_path, 'r') as f:
            content = f.read()
            assert "Spectral GPT" in content
            assert "Abstract" in content
            assert "Introduction" in content


if __name__ == "__main__":
    test_paper_generator_initialization()
    test_figure_referencing()
    test_markdown_file_writing()
    test_generate_intuitive_guide()
    test_generate_technical_paper()
    print("✓ All paper generator tests passed!")
