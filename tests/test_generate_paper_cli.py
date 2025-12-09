"""
Tests for the generate_paper.py CLI script.
"""

import subprocess
import sys
from pathlib import Path
import tempfile
import shutil


def test_cli_help():
    """Test that the CLI help option works."""
    result = subprocess.run(
        [sys.executable, 'spectral_gpt/generate_paper.py', '--help'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert 'Generate Spectral GPT documentation' in result.stdout
    assert '--type' in result.stdout
    assert '--experiments' in result.stdout
    assert '--output' in result.stdout
    assert '--format' in result.stdout


def test_cli_intuitive_guide():
    """Test generating intuitive guide via CLI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                'spectral_gpt/generate_paper.py',
                '--type', 'intuitive',
                '--output', tmpdir,
                '--verbose'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Generating Intuitive Guide' in result.stdout
        assert 'Documentation generation complete' in result.stdout
        
        # Check that output file was created
        output_file = Path(tmpdir) / 'spectral_gpt_intuitive_guide.md'
        assert output_file.exists()
        
        # Check that file has content
        content = output_file.read_text()
        assert len(content) > 0
        assert 'Spectral GPT' in content


def test_cli_technical_paper():
    """Test generating technical paper via CLI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                'spectral_gpt/generate_paper.py',
                '--type', 'technical',
                '--output', tmpdir,
                '--verbose'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Generating Technical Paper' in result.stdout
        assert 'Documentation generation complete' in result.stdout
        
        # Check that output file was created
        output_file = Path(tmpdir) / 'spectral_gpt_technical_paper.md'
        assert output_file.exists()
        
        # Check that file has content
        content = output_file.read_text()
        assert len(content) > 0
        assert 'Spectral GPT' in content


def test_cli_both_types():
    """Test generating both document types via CLI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                'spectral_gpt/generate_paper.py',
                '--type', 'both',
                '--output', tmpdir,
                '--verbose'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Generating Intuitive Guide' in result.stdout
        assert 'Generating Technical Paper' in result.stdout
        
        # Check that both output files were created
        intuitive_file = Path(tmpdir) / 'spectral_gpt_intuitive_guide.md'
        technical_file = Path(tmpdir) / 'spectral_gpt_technical_paper.md'
        
        assert intuitive_file.exists()
        assert technical_file.exists()


def test_cli_with_specific_experiments():
    """Test CLI with specific experiment directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock experiment directory
        exp_dir = Path(tmpdir) / 'mock_experiment'
        exp_dir.mkdir()
        
        # Create a minimal config file
        config_file = exp_dir / 'config.json'
        config_file.write_text('{"experiment_id": "test"}')
        
        result = subprocess.run(
            [
                sys.executable,
                'spectral_gpt/generate_paper.py',
                '--type', 'intuitive',
                '--experiments', str(exp_dir),
                '--output', tmpdir,
                '--verbose'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Using specified experiments: 1' in result.stdout
        assert 'Valid experiment' in result.stdout


def test_cli_with_nonexistent_experiments_dir():
    """Test CLI handles nonexistent experiments directory gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                'spectral_gpt/generate_paper.py',
                '--type', 'intuitive',
                '--experiments-dir', 'nonexistent_dir',
                '--output', tmpdir,
                '--verbose'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'No experiments found' in result.stdout
        assert 'Generating documentation without experiment data' in result.stdout


def test_cli_pdf_format_without_pandoc():
    """Test CLI handles PDF format gracefully when pandoc is not available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                'spectral_gpt/generate_paper.py',
                '--type', 'intuitive',
                '--format', 'pdf',
                '--output', tmpdir,
                '--verbose'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        
        # Should still generate markdown even if PDF fails
        output_file = Path(tmpdir) / 'spectral_gpt_intuitive_guide.md'
        assert output_file.exists()


def test_cli_template_option():
    """Test CLI with different template options."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for template in ['arxiv', 'neurips', 'icml']:
            result = subprocess.run(
                [
                    sys.executable,
                    'spectral_gpt/generate_paper.py',
                    '--type', 'technical',
                    '--template', template,
                    '--output', tmpdir,
                    '--verbose'
                ],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert f'Template: {template}' in result.stdout


if __name__ == '__main__':
    # Run tests
    import pytest
    pytest.main([__file__, '-v'])
