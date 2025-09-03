#!/usr/bin/env python3
"""
DEIM Dataset Visualization Entry Point
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Simple script to launch the dataset visualizer for any configured dataset.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.visualization.dataset_visualizer import DatasetVisualizer


def list_available_configs():
    """List all available dataset configuration files."""
    config_dir = project_root / "configs" / "dataset"
    if not config_dir.exists():
        print("No dataset configurations found.")
        return []
    
    configs = list(config_dir.glob("*.yml"))
    print("\nAvailable dataset configurations:")
    for i, config in enumerate(configs, 1):
        print(f"  {i:2d}. {config.stem}")
    
    return configs


def main():
    """Main entry point for dataset visualization."""
    parser = argparse.ArgumentParser(
        description='DEIM Dataset Visualizer - Interactive dataset browser with OpenCV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_dataset.py coco_detection          # Visualize COCO dataset (train split)
  python visualize_dataset.py coco_detection --split val  # Visualize COCO dataset (val split)
  python visualize_dataset.py --list                  # List all available configurations
  python visualize_dataset.py drone_detection --start-index 100  # Start from sample 100
  
  # Or use full path to config file:
  python visualize_dataset.py configs/dataset/custom_detection.yml
        """
    )
    
    parser.add_argument('config', nargs='?', help='Dataset configuration name or path to YAML file')
    parser.add_argument('--split', choices=['train', 'val'], default='train',
                        help='Dataset split to visualize (default: train)')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting index for visualization (default: 0)')
    parser.add_argument('--list', action='store_true',
                        help='List available dataset configurations')
    
    args = parser.parse_args()
    
    # Handle list option
    if args.list:
        list_available_configs()
        return
    
    # Validate config argument
    if not args.config:
        print("Error: Please specify a dataset configuration.")
        print("Use --list to see available configurations.")
        return
    
    # Determine config file path
    if args.config.endswith('.yml') or args.config.endswith('.yaml'):
        # Full path provided
        config_path = args.config
    else:
        # Configuration name provided
        config_path = project_root / "configs" / "dataset" / f"{args.config}.yml"
    
    # Validate config file exists
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found")
        print("\nAvailable configurations:")
        list_available_configs()
        return
    
    # Print information
    print("=" * 60)
    print("DEIM Dataset Visualizer")
    print("=" * 60)
    print(f"Configuration: {config_path}")
    print(f"Split: {args.split}")
    print(f"Starting index: {args.start_index}")
    print()
    
    try:
        # Create and run visualizer
        visualizer = DatasetVisualizer(str(config_path), args.split)
        
        # Validate starting index
        if args.start_index >= len(visualizer.dataset):
            print(f"Warning: Start index {args.start_index} is out of range. Using 0.")
            args.start_index = 0
        
        visualizer.current_index = args.start_index
        visualizer.run()
        
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTip: Make sure the dataset paths in the configuration file are correct")
        print("     and that all required dependencies are installed.")


if __name__ == '__main__':
    main()
