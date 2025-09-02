#!/usr/bin/env python3
"""
Script to split COCO formatted dataset into train and validation sets.

This script:
- Loads the existing train.json from /media/fast/drone_train/drone_ds
- Creates a validation set with 3000 images:
  - 1500 images from drone datasets (equally distributed)
  - 1500 images from objects365
- Saves the validation annotations as val.json
- Updates train.json to exclude validation images
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import logging
from tqdm import tqdm
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainValSplitter:
    def __init__(self, data_dir: str, val_size: int = 3000, seed: int = 42):
        """
        Initialize the train/validation splitter.
        
        Args:
            data_dir: Path to the dataset directory containing train.json
            val_size: Total number of images for validation set
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.train_json_path = self.data_dir / "train.json"
        self.val_json_path = self.data_dir / "val.json"
        self.val_size = val_size
        self.seed = seed
        
        # Define drone datasets based on coco_dataset.py
        self.drone_datasets = [
            "rgbt_drone_person",
            "search_and_rescue", 
            "stanford_drone",
            "visdrone2019",
            "vtsar",
            "vtuav",
            "wisard"
        ]
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Data structures
        self.data = None
        self.images_by_dataset = defaultdict(list)
        self.image_id_to_annotations = defaultdict(list)
        
    def load_data(self):
        """Load the original train.json file."""
        logger.info(f"Loading data from {self.train_json_path}")
        
        if not self.train_json_path.exists():
            raise FileNotFoundError(f"Train file not found: {self.train_json_path}")
        
        with open(self.train_json_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data['images'])} images and {len(self.data['annotations'])} annotations")
        
    def organize_data(self):
        """Organize images by dataset and create annotation mappings."""
        logger.info("Organizing data by dataset...")
        
        # Group images by dataset
        for img in tqdm(self.data['images'], desc="Grouping images"):
            dataset = img.get('dataset', 'unknown')
            self.images_by_dataset[dataset].append(img)
        
        # Map annotations to images
        for ann in tqdm(self.data['annotations'], desc="Mapping annotations"):
            self.image_id_to_annotations[ann['image_id']].append(ann)
        
        # Print statistics
        logger.info("\nDataset statistics:")
        for dataset, images in sorted(self.images_by_dataset.items()):
            logger.info(f"  {dataset}: {len(images)} images")
            
    def select_validation_images(self) -> Tuple[Set[int], Dict[str, List[int]]]:
        """
        Select images for validation set according to the rules.
        
        Returns:
            Tuple of (set of validation image IDs, dict of dataset to selected image IDs)
        """
        val_image_ids = set()
        val_images_by_dataset = defaultdict(list)
        
        # Calculate how many images to take from each drone dataset
        drone_val_total = self.val_size // 2  # 1500
        num_drone_datasets = len(self.drone_datasets)
        images_per_drone = drone_val_total // num_drone_datasets  # ~214
        
        logger.info(f"\nSelecting validation images:")
        logger.info(f"  Total validation size: {self.val_size}")
        logger.info(f"  Drone datasets allocation: {drone_val_total} images")
        logger.info(f"  Objects365 allocation: {self.val_size - drone_val_total} images")
        logger.info(f"  Images per drone dataset: ~{images_per_drone}")
        
        # Select from drone datasets
        drone_selected_total = 0
        for dataset in self.drone_datasets:
            if dataset not in self.images_by_dataset:
                logger.warning(f"  Dataset '{dataset}' not found in data")
                continue
                
            available_images = self.images_by_dataset[dataset]
            num_to_select = min(images_per_drone, len(available_images))
            
            if num_to_select < images_per_drone:
                logger.warning(f"  {dataset}: only {len(available_images)} images available, selecting all")
            
            selected = random.sample(available_images, num_to_select)
            for img in selected:
                val_image_ids.add(img['id'])
                val_images_by_dataset[dataset].append(img['id'])
            
            drone_selected_total += num_to_select
            logger.info(f"  {dataset}: selected {num_to_select} images")
        
        # Adjust objects365 selection based on actual drone selection
        objects365_needed = self.val_size - drone_selected_total
        
        # Select from objects365
        if 'objects365' in self.images_by_dataset:
            available_objects365 = self.images_by_dataset['objects365']
            num_to_select = min(objects365_needed, len(available_objects365))
            
            selected = random.sample(available_objects365, num_to_select)
            for img in selected:
                val_image_ids.add(img['id'])
                val_images_by_dataset['objects365'].append(img['id'])
            
            logger.info(f"  objects365: selected {num_to_select} images")
        else:
            logger.warning("  objects365 dataset not found")
        
        logger.info(f"\nTotal validation images selected: {len(val_image_ids)}")
        
        return val_image_ids, val_images_by_dataset
    
    def split_data(self, val_image_ids: Set[int]) -> Tuple[Dict, Dict]:
        """
        Split the data into train and validation sets.
        
        Args:
            val_image_ids: Set of image IDs selected for validation
            
        Returns:
            Tuple of (train_data, val_data) dictionaries
        """
        logger.info("\nSplitting data into train and validation sets...")
        
        # Initialize train and val data structures
        train_data = {
            "info": self.data.get("info", {}),
            "licenses": self.data.get("licenses", []),
            "categories": self.data.get("categories", []),
            "images": [],
            "annotations": []
        }
        
        val_data = {
            "info": self.data.get("info", {}),
            "licenses": self.data.get("licenses", []),
            "categories": self.data.get("categories", []),
            "images": [],
            "annotations": []
        }
        
        # Split images
        for img in tqdm(self.data['images'], desc="Splitting images"):
            if img['id'] in val_image_ids:
                val_data['images'].append(img)
            else:
                train_data['images'].append(img)
        
        # Split annotations
        for ann in tqdm(self.data['annotations'], desc="Splitting annotations"):
            if ann['image_id'] in val_image_ids:
                val_data['annotations'].append(ann)
            else:
                train_data['annotations'].append(ann)
        
        logger.info(f"Train set: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
        logger.info(f"Val set: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
        
        return train_data, val_data
    
    def save_splits(self, train_data: Dict, val_data: Dict, backup: bool = True):
        """
        Save the train and validation splits to JSON files.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            backup: Whether to create a backup of the original train.json
        """
        # Create backup of original train.json if requested
        if backup and self.train_json_path.exists():
            backup_path = self.train_json_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            logger.info(f"Creating backup: {backup_path}")
            shutil.copy2(self.train_json_path, backup_path)
        
        # Save validation set
        logger.info(f"Saving validation set to {self.val_json_path}")
        with open(self.val_json_path, 'w') as f:
            json.dump(val_data, f)
        
        # Save updated training set
        logger.info(f"Saving updated training set to {self.train_json_path}")
        with open(self.train_json_path, 'w') as f:
            json.dump(train_data, f)
        
        logger.info("Split completed successfully!")
    
    def print_statistics(self, val_images_by_dataset: Dict[str, List[int]]):
        """Print detailed statistics about the split."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SET STATISTICS")
        logger.info("="*60)
        
        total_val_images = sum(len(ids) for ids in val_images_by_dataset.values())
        
        # Drone datasets statistics
        drone_total = 0
        logger.info("\nDrone Datasets:")
        for dataset in self.drone_datasets:
            if dataset in val_images_by_dataset:
                count = len(val_images_by_dataset[dataset])
                drone_total += count
                percentage = (count / total_val_images) * 100
                logger.info(f"  {dataset:20s}: {count:5d} images ({percentage:5.2f}%)")
        
        logger.info(f"\nTotal Drone Images: {drone_total} ({(drone_total/total_val_images)*100:.1f}%)")
        
        # Objects365 statistics
        if 'objects365' in val_images_by_dataset:
            obj365_count = len(val_images_by_dataset['objects365'])
            logger.info(f"Objects365 Images: {obj365_count} ({(obj365_count/total_val_images)*100:.1f}%)")
        
        logger.info(f"\nTotal Validation Images: {total_val_images}")
        logger.info("="*60)
    
    def run(self):
        """Execute the train/validation split."""
        logger.info("Starting train/validation split...")
        logger.info(f"Random seed: {self.seed}")
        
        # Load data
        self.load_data()
        
        # Organize data
        self.organize_data()
        
        # Select validation images
        val_image_ids, val_images_by_dataset = self.select_validation_images()
        
        # Print statistics
        self.print_statistics(val_images_by_dataset)
        
        # Split data
        train_data, val_data = self.split_data(val_image_ids)
        
        # Save splits
        self.save_splits(train_data, val_data, backup=True)
        
        logger.info("\nProcess completed successfully!")


def main():
    """Main function to execute the train/validation split."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split COCO dataset into train and validation sets"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/media/fast/drone_train/drone_ds",
        help="Path to dataset directory containing train.json"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=3000,
        help="Number of images for validation set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of original train.json"
    )
    
    args = parser.parse_args()
    
    # Create splitter and run
    splitter = TrainValSplitter(
        data_dir=args.data_dir,
        val_size=args.val_size,
        seed=args.seed
    )
    
    try:
        splitter.run()
    except Exception as e:
        logger.error(f"Error during split: {e}")
        raise


if __name__ == "__main__":
    main()

