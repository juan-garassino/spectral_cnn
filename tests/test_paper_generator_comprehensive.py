"""
Comprehensive tests for paper generation infrastructure
"""

import os
import tempfile
from pathlib import Path

from spectral_gpt.paper_generator import SpectralGPTPaperGenerator


def test_full_workflow():
    """Test complete paper generation workflow"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        # Generate intuitive guide
        guide_path = generator.generate_intuitive_guide(experiments=[])
        assert Path(guide_path).exists()
        
        with open(guide_path, 'r') as f:
            guide_content = f.read()
            
        # Verify key sections
        assert "Spectral GPT" in guide_content
        assert "Introduction" in guide_content
        assert "Visual Introduction" in guide_content
        assert "Layer-by-Layer Comparison" in guide_content
        assert "Wave Properties" in guide_content
        assert "Architecture Differences" in guide_content
        
        # Generate technical paper
        paper_path = generator.generate_technical_paper(experiments=[])
        assert Path(paper_path).exists()
        
        with open(paper_path, 'r') as f:
            paper_content = f.read()
        
        # Verify key sections
        assert "Abstract" in paper_content
        assert "Introduction" in paper_content
        assert "Related Work" in paper_content
        assert "Mathematical Formulation" in paper_content
        assert "Architecture Details" in paper_content
        assert "Experimental Methodology" in paper_content
        assert "Results" in paper_content
        assert "Analysis" in paper_content
        assert "Discussion" in paper_content
        assert "Conclusion" in paper_content
        
        # Verify mathematical content
        assert "Wave Packet Embeddings" in paper_content
        assert "Interference Attention" in paper_content
        assert "Resonant Gradient Descent" in paper_content
        
        print("✓ Full workflow test passed")


def test_layer_comparison():
    """Test layer comparison generation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        wave_info = {'num_parameters': 67_473_706}
        standard_info = {'num_parameters': 52_892_160}
        
        comparison = generator.generate_layer_comparison(wave_info, standard_info)
        
        # Verify content
        assert "Layer-by-Layer" in comparison
        assert "Architecture Diagrams" in comparison
        assert "Parameter Count" in comparison
        assert "Computational Complexity" in comparison
        assert "Standard Transformer" in comparison
        assert "Spectral GPT" in comparison
        
        print("✓ Layer comparison test passed")


def test_fitting_analysis():
    """Test fitting analysis generation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        exp_data = [
            {'name': 'Standard', 'loss_history': [8.3, 6.5, 4.44]},
            {'name': 'Spectral', 'loss_history': [7.9, 6.4, 4.48]}
        ]
        
        fitting = generator.generate_fitting_analysis(exp_data)
        
        # Verify content
        assert "Fitting Analysis" in fitting
        assert "Loss Landscape" in fitting
        assert "Convergence" in fitting
        assert "Frequency Spectrum" in fitting
        assert "Universal Function Approximation" in fitting
        
        print("✓ Fitting analysis test passed")


def test_figure_management():
    """Test figure referencing system"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        # Add figures
        ref1 = generator._add_figure_reference("fig1.png", "First figure")
        ref2 = generator._add_figure_reference("fig2.png", "Second figure")
        ref3 = generator._add_figure_reference("fig3.png", "Third figure")
        
        # Verify numbering
        assert "Figure 1" in ref1
        assert "Figure 2" in ref2
        assert "Figure 3" in ref3
        
        # Verify captions
        assert "First figure" in ref1
        assert "Second figure" in ref2
        assert "Third figure" in ref3
        
        # Test reset
        generator._reset_figure_counter()
        ref4 = generator._add_figure_reference("fig4.png", "Fourth figure")
        assert "Figure 1" in ref4  # Counter reset
        
        print("✓ Figure management test passed")


def test_content_quality():
    """Test quality of generated content"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        # Generate guide
        guide_path = generator.generate_intuitive_guide(experiments=[])
        
        with open(guide_path, 'r') as f:
            guide_content = f.read()
        
        # Check for quality indicators
        assert len(guide_content) > 10000  # Substantial content
        assert guide_content.count('\n\n') > 50  # Well-formatted paragraphs
        assert guide_content.count('```') >= 4  # Code blocks
        assert guide_content.count('|') > 20  # Tables
        
        # Generate paper
        paper_path = generator.generate_technical_paper(experiments=[])
        
        with open(paper_path, 'r') as f:
            paper_content = f.read()
        
        # Check for quality indicators
        assert len(paper_content) > 14000  # Substantial content
        assert paper_content.count('$$') >= 4  # Math equations
        assert paper_content.count('##') >= 10  # Multiple sections
        assert paper_content.count('|') > 30  # Tables
        
        print("✓ Content quality test passed")


def test_experiment_loading():
    """Test experiment data loading"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SpectralGPTPaperGenerator(output_dir=tmpdir)
        
        # Create mock experiment directory
        exp_dir = Path(tmpdir) / "mock_experiment"
        exp_dir.mkdir()
        
        # Create mock config
        import json
        config = {
            'experiment_id': 'test_exp',
            'model': {'type': 'wave', 'd_model': 768}
        }
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f)
        
        # Load config
        loaded_config = generator._load_experiment_config(str(exp_dir))
        assert loaded_config is not None
        assert loaded_config['experiment_id'] == 'test_exp'
        
        print("✓ Experiment loading test passed")


if __name__ == "__main__":
    test_full_workflow()
    test_layer_comparison()
    test_fitting_analysis()
    test_figure_management()
    test_content_quality()
    test_experiment_loading()
    print("\n✅ All comprehensive tests passed!")
