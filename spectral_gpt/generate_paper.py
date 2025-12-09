#!/usr/bin/env python3
"""
Command-line interface for generating Spectral GPT documentation.

This script provides a CLI for generating intuitive guides and technical papers
from experiment results.

Usage:
    # Generate intuitive guide
    python generate_paper.py --type intuitive --output experiments/paper
    
    # Generate technical paper
    python generate_paper.py --type technical --output experiments/paper
    
    # Generate both
    python generate_paper.py --type both --output experiments/paper
    
    # Select specific experiments
    python generate_paper.py --type both --experiments exp1 exp2 --output experiments/paper
    
    # Generate PDF
    python generate_paper.py --type technical --format pdf --output experiments/paper
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from paper_generator import SpectralGPTPaperGenerator


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate Spectral GPT documentation from experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate intuitive guide only
  %(prog)s --type intuitive
  
  # Generate technical paper with PDF
  %(prog)s --type technical --format pdf
  
  # Generate both documents from specific experiments
  %(prog)s --type both --experiments exp1 exp2 exp3
  
  # Custom output directory
  %(prog)s --type both --output my_papers/
        """
    )
    
    parser.add_argument(
        '--type',
        choices=['intuitive', 'technical', 'both'],
        default='both',
        help='Type of documentation to generate (default: both)'
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        metavar='EXP',
        help='List of experiment directories to include (default: all in experiments/)'
    )
    
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='experiments',
        help='Base directory containing experiments (default: experiments/)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='experiments/paper',
        help='Output directory for generated papers (default: experiments/paper/)'
    )
    
    parser.add_argument(
        '--format',
        choices=['markdown', 'pdf', 'both'],
        default='markdown',
        help='Output format (default: markdown)'
    )
    
    parser.add_argument(
        '--template',
        choices=['arxiv', 'neurips', 'icml'],
        default='arxiv',
        help='Paper template for technical paper (default: arxiv)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def discover_experiments(experiments_dir: str) -> List[str]:
    """
    Discover all experiment directories in the base directory.
    
    Args:
        experiments_dir: Base directory containing experiments
        
    Returns:
        List of experiment directory paths
    """
    base_path = Path(experiments_dir)
    
    if not base_path.exists():
        return []
    
    # Look for directories with config.json or results.json
    experiments = []
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if it looks like an experiment directory
            has_config = (item / 'config.json').exists()
            has_results = (item / 'results.json').exists()
            has_logs = (item / 'logs').exists()
            
            if has_config or has_results or has_logs:
                experiments.append(str(item))
    
    return sorted(experiments)


def validate_experiments(experiment_paths: List[str], verbose: bool = False) -> List[str]:
    """
    Validate that experiment directories exist and contain data.
    
    Args:
        experiment_paths: List of experiment directory paths
        verbose: Whether to print validation details
        
    Returns:
        List of valid experiment paths
    """
    valid_experiments = []
    
    for exp_path in experiment_paths:
        path = Path(exp_path)
        
        if not path.exists():
            if verbose:
                print(f"Warning: Experiment directory not found: {exp_path}")
            continue
        
        if not path.is_dir():
            if verbose:
                print(f"Warning: Not a directory: {exp_path}")
            continue
        
        # Check for at least one data file
        has_data = (
            (path / 'config.json').exists() or
            (path / 'results.json').exists() or
            (path / 'logs' / 'metrics.jsonl').exists()
        )
        
        if not has_data:
            if verbose:
                print(f"Warning: No data files found in: {exp_path}")
            continue
        
        valid_experiments.append(exp_path)
        if verbose:
            print(f"✓ Valid experiment: {exp_path}")
    
    return valid_experiments


def generate_intuitive_guide(
    generator: SpectralGPTPaperGenerator,
    experiments: List[str],
    output_format: str,
    verbose: bool = False
) -> Optional[str]:
    """
    Generate intuitive guide.
    
    Args:
        generator: Paper generator instance
        experiments: List of experiment paths
        output_format: Output format ('markdown', 'pdf', or 'both')
        verbose: Whether to print progress
        
    Returns:
        Path to generated markdown file, or None on error
    """
    try:
        if verbose:
            print("\n" + "="*60)
            print("Generating Intuitive Guide")
            print("="*60)
        
        # Generate markdown
        md_path = generator.generate_intuitive_guide(experiments)
        
        if verbose:
            print(f"\n✓ Generated intuitive guide: {md_path}")
        
        # Generate PDF if requested
        if output_format in ['pdf', 'both']:
            if verbose:
                print("\nConverting to PDF...")
            
            pdf_path = generator.render_to_pdf(md_path)
            
            if pdf_path:
                if verbose:
                    print(f"✓ Generated PDF: {pdf_path}")
            else:
                print("Warning: PDF generation failed (is pandoc installed?)")
        
        return md_path
        
    except Exception as e:
        print(f"Error generating intuitive guide: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def generate_technical_paper(
    generator: SpectralGPTPaperGenerator,
    experiments: List[str],
    template: str,
    output_format: str,
    verbose: bool = False
) -> Optional[str]:
    """
    Generate technical paper.
    
    Args:
        generator: Paper generator instance
        experiments: List of experiment paths
        template: Paper template ('arxiv', 'neurips', 'icml')
        output_format: Output format ('markdown', 'pdf', or 'both')
        verbose: Whether to print progress
        
    Returns:
        Path to generated markdown file, or None on error
    """
    try:
        if verbose:
            print("\n" + "="*60)
            print("Generating Technical Paper")
            print("="*60)
        
        # Generate markdown
        md_path = generator.generate_technical_paper(experiments, template=template)
        
        if verbose:
            print(f"\n✓ Generated technical paper: {md_path}")
        
        # Generate PDF if requested
        if output_format in ['pdf', 'both']:
            if verbose:
                print("\nConverting to PDF...")
            
            pdf_path = generator.render_to_pdf(md_path)
            
            if pdf_path:
                if verbose:
                    print(f"✓ Generated PDF: {pdf_path}")
            else:
                print("Warning: PDF generation failed (is pandoc installed?)")
        
        return md_path
        
    except Exception as e:
        print(f"Error generating technical paper: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def main():
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    # Print header
    if args.verbose:
        print("="*60)
        print("Spectral GPT Paper Generator")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Type: {args.type}")
        print(f"  Format: {args.format}")
        print(f"  Output: {args.output}")
        print(f"  Template: {args.template}")
    
    # Determine which experiments to include
    if args.experiments:
        # Use specified experiments
        experiment_paths = args.experiments
        if args.verbose:
            print(f"\nUsing specified experiments: {len(experiment_paths)}")
    else:
        # Discover experiments automatically
        if args.verbose:
            print(f"\nDiscovering experiments in: {args.experiments_dir}")
        
        experiment_paths = discover_experiments(args.experiments_dir)
        
        if not experiment_paths:
            print(f"Warning: No experiments found in {args.experiments_dir}")
            print("Generating documentation without experiment data...")
            experiment_paths = []
        elif args.verbose:
            print(f"Found {len(experiment_paths)} experiment(s)")
    
    # Validate experiments
    if experiment_paths:
        valid_experiments = validate_experiments(experiment_paths, verbose=args.verbose)
        
        if not valid_experiments:
            print("Error: No valid experiments found")
            print("Generating documentation without experiment data...")
            valid_experiments = []
        elif args.verbose:
            print(f"\nValidated {len(valid_experiments)} experiment(s)")
    else:
        valid_experiments = []
    
    # Create paper generator
    if args.verbose:
        print(f"\nInitializing paper generator...")
        print(f"  Output directory: {args.output}")
    
    try:
        generator = SpectralGPTPaperGenerator(
            output_dir=args.output,
            experiments_base_dir=args.experiments_dir
        )
    except Exception as e:
        print(f"Error initializing paper generator: {e}")
        return 1
    
    # Generate documentation based on type
    success = True
    
    if args.type in ['intuitive', 'both']:
        result = generate_intuitive_guide(
            generator,
            valid_experiments,
            args.format,
            verbose=args.verbose
        )
        if result is None:
            success = False
    
    if args.type in ['technical', 'both']:
        result = generate_technical_paper(
            generator,
            valid_experiments,
            args.template,
            args.format,
            verbose=args.verbose
        )
        if result is None:
            success = False
    
    # Print summary
    if args.verbose:
        print("\n" + "="*60)
        if success:
            print("✓ Documentation generation complete!")
        else:
            print("⚠ Documentation generation completed with errors")
        print("="*60)
        print(f"\nOutput directory: {args.output}")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
