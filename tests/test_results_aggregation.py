"""
Tests for ResultsParser and ResultsAggregator
"""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spectral_gpt.monitoring import ResultsParser, ResultsAggregator


def test_results_parser_basic():
    """Test basic parsing of results file"""
    results_file = 'spectral_gpt/experiment_results/results.txt'
    
    if not os.path.exists(results_file):
        pytest.skip("Results file not found")
    
    parser = ResultsParser(results_file)
    experiments = parser.parse()
    
    # Should parse at least 3 experiments
    assert len(experiments) >= 3
    
    # Each experiment should have required fields
    for exp in experiments:
        assert 'name' in exp
        assert 'config' in exp
        assert 'training_metrics' in exp
        assert 'final_results' in exp
        assert 'generation_sample' in exp


def test_results_parser_experiment_details():
    """Test that parser extracts experiment details correctly"""
    results_file = 'spectral_gpt/experiment_results/results.txt'
    
    if not os.path.exists(results_file):
        pytest.skip("Results file not found")
    
    parser = ResultsParser(results_file)
    experiments = parser.parse()
    
    # Find Standard Transformer experiment
    standard_exp = None
    for exp in experiments:
        if 'Standard Transformer' in exp['name']:
            standard_exp = exp
            break
    
    assert standard_exp is not None
    
    # Check config
    assert 'parameters' in standard_exp['config']
    assert standard_exp['config']['parameters'] > 0
    assert 'optimizer' in standard_exp['config']
    assert 'loss' in standard_exp['config']
    
    # Check training metrics
    assert len(standard_exp['training_metrics']) > 0
    
    # Check final results
    assert 'best_val_loss' in standard_exp['final_results']


def test_results_parser_save_json():
    """Test saving parsed results to JSON"""
    results_file = 'spectral_gpt/experiment_results/results.txt'
    
    if not os.path.exists(results_file):
        pytest.skip("Results file not found")
    
    parser = ResultsParser(results_file)
    parser.parse()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        parser.save_to_json(temp_file)
        
        # Verify file was created and is valid JSON
        assert os.path.exists(temp_file)
        
        with open(temp_file, 'r') as f:
            data = json.load(f)
        
        assert 'experiments' in data
        assert 'parsed_at' in data
        assert 'source_file' in data
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_results_aggregator_comparison_table():
    """Test generating comparison table"""
    results_file = 'spectral_gpt/experiment_results/results.txt'
    
    if not os.path.exists(results_file):
        pytest.skip("Results file not found")
    
    parser = ResultsParser(results_file)
    parser.parse()
    
    aggregator = ResultsAggregator()
    aggregator.load_from_parser(parser)
    
    table = aggregator.generate_comparison_table()
    
    # Table should contain headers
    assert 'Experiment' in table
    assert 'Parameters' in table
    assert 'Val Loss' in table


def test_results_aggregator_statistics():
    """Test generating summary statistics"""
    results_file = 'spectral_gpt/experiment_results/results.txt'
    
    if not os.path.exists(results_file):
        pytest.skip("Results file not found")
    
    parser = ResultsParser(results_file)
    parser.parse()
    
    aggregator = ResultsAggregator()
    aggregator.load_from_parser(parser)
    
    stats = aggregator.generate_summary_statistics()
    
    assert 'num_experiments' in stats
    assert stats['num_experiments'] > 0
    assert 'val_loss' in stats
    assert 'mean' in stats['val_loss']


def test_results_aggregator_ablation_analysis():
    """Test ablation analysis"""
    results_file = 'spectral_gpt/experiment_results/results.txt'
    
    if not os.path.exists(results_file):
        pytest.skip("Results file not found")
    
    parser = ResultsParser(results_file)
    parser.parse()
    
    aggregator = ResultsAggregator()
    aggregator.load_from_parser(parser)
    
    ablation = aggregator.generate_ablation_analysis()
    
    assert 'baseline' in ablation
    assert 'components' in ablation
    
    # Should identify baseline
    assert ablation['baseline'] is not None
    
    # Should have component analysis
    assert len(ablation['components']) > 0


def test_results_aggregator_compare_experiments():
    """Test comparing two experiments"""
    results_file = 'spectral_gpt/experiment_results/results.txt'
    
    if not os.path.exists(results_file):
        pytest.skip("Results file not found")
    
    parser = ResultsParser(results_file)
    experiments = parser.parse()
    
    if len(experiments) < 2:
        pytest.skip("Need at least 2 experiments")
    
    aggregator = ResultsAggregator()
    aggregator.load_from_parser(parser)
    
    # Compare first two experiments
    exp1_name = experiments[0]['name'].split()[0]
    exp2_name = experiments[1]['name'].split()[0]
    
    comparison = aggregator.compare_experiments(exp1_name, exp2_name)
    
    assert 'experiment_1' in comparison
    assert 'experiment_2' in comparison


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
