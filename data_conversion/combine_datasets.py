#!/usr/bin/env python3
"""
Script to combine multiple drone/person detection datasets into a single COCO format dataset.
All person-related categories are merged into a single "person" category.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import logging
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetCombiner:
    def __init__(self, output_dir: str, dry_run: bool = False, images_per_folder: int = 10000):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.annotations_file = self.output_dir / "annotations.json"
        self.dry_run = dry_run
        self.images_per_folder = images_per_folder
        
        if self.dry_run:
            logger.info("🔍 DRY RUN MODE - No files will be copied")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.dry_run:
            self.images_dir.mkdir(exist_ok=True)
        
        # Initialize COCO format structure
        self.combined_data = {
            "info": {
                "description": "Combined Person Detection Dataset from Multiple Drone Datasets",
                "version": "1.0",
                "year": 2024
            },
            "licenses": [],
            "categories": [{"id": 0, "name": "person", "supercategory": "person"}],
            "images": [],
            "annotations": []
        }
        
        # Tracking
        self.image_id_counter = 0
        self.annotation_id_counter = 0
        self.image_filename_mapping = {}  # old_path -> new_filename
        self.stats = defaultdict(lambda: {
            "total_images": 0,
            "total_annotations": 0,
            "images_with_persons": 0,
            "crowd_annotations": 0,
            "modalities": set(),
            "splits": set(),
            "missing_images": 0
        })
        
    def is_person_category(self, category_name: str) -> bool:
        """Check if a category name refers to a person."""
        person_keywords = [
            'person', 'people', 'pedestrian', 'human', 'crowd', 
            'rider', 'biker', 'skater'
        ]
        name_lower = category_name.lower()
        return any(keyword in name_lower for keyword in person_keywords)
    
    def is_crowd_category(self, category_name: str) -> bool:
        """Check if a category should be marked as crowd."""
        crowd_keywords = ['crowd', 'people', 'group']
        return any(keyword in category_name.lower() for keyword in crowd_keywords)
    
    def get_new_image_path(self, dataset_name: str, original_filename: str, image_id: int) -> Tuple[str, Path]:
        """Generate a new unique filename with pagination folder structure.
        
        Returns:
            Tuple of (relative_path_for_json, full_destination_path)
        """
        ext = Path(original_filename).suffix
        
        # Calculate folder number (0-based, but display as 1-based)
        folder_num = image_id // self.images_per_folder
        folder_name = f"{folder_num:07d}"  # 0000000, 0000001, etc.
        
        # Create filename
        filename = f"{dataset_name}_{image_id:08d}{ext}"
        
        # Relative path for JSON (images/0000001/filename.jpg)
        relative_path = f"{folder_name}/{filename}"
        
        # Full destination path
        folder_path = self.images_dir / folder_name
        if not self.dry_run:
            folder_path.mkdir(parents=True, exist_ok=True)
        
        full_path = folder_path / filename
        
        return relative_path, full_path
    
    def copy_image(self, source_path: Path, dest_path: Path) -> bool:
        """Copy image to the combined dataset directory."""
        if self.dry_run:
            # In dry run, just check if source exists
            if source_path.exists():
                return True
            else:
                logger.warning(f"Source image not found: {source_path}")
                return False
        
        try:
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                return True
            else:
                logger.warning(f"Source image not found: {source_path}")
                return False
        except Exception as e:
            logger.error(f"Error copying image {source_path}: {e}")
            return False
    
    def process_rgbt_drone_person(self):
        """Process RGBTDronePerson dataset."""
        dataset_name = "rgbt_drone_person"
        base_path = Path("/mnt/archive/person_drone/RGBTDronePerson-20250828T031729Z-1-001/RGBTDronePerson")
        
        annotation_files = [
            ("train_thermal.json", "train", "thermal"),
            ("val_thermal.json", "val", "thermal"),
            ("sub_train_thermal.json", "sub_train", "thermal"),
            ("sub_train_visible.json", "sub_train", "visible")
        ]
        
        for ann_file, split, modality in annotation_files:
            ann_path = base_path / ann_file
            if not ann_path.exists():
                logger.warning(f"Annotation file not found: {ann_path}")
                continue
                
            logger.info(f"Processing {dataset_name} - {split} - {modality}")
            
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            # Map category IDs
            category_mapping = {}
            for cat in data.get('categories', []):
                if self.is_person_category(cat['name']):
                    category_mapping[cat['id']] = 0
            
            # Process images and annotations
            image_id_mapping = {}
            images_with_persons = set()
            
            # First pass: identify images with person annotations
            for ann in data.get('annotations', []):
                if ann['category_id'] in category_mapping:
                    images_with_persons.add(ann['image_id'])
            
            # Process ALL images (drone dataset - keep all images)
            for img in tqdm(data.get('images', []), desc=f"Processing {dataset_name} {split} {modality} images"):
                
                old_id = img['id']
                new_id = self.image_id_counter
                self.image_id_counter += 1
                image_id_mapping[old_id] = new_id
                
                # Determine image path based on split and modality
                img_filename = img['file_name']
                # RGBTDronePerson has structure: RGBTDronePerson/{split}/{modality}/{filename}
                if split == "sub_train":
                    # sub_train doesn't have its own folder, uses train folder
                    source_path = base_path / "RGBTDronePerson" / "train" / modality / img_filename
                else:
                    source_path = base_path / "RGBTDronePerson" / split / modality / img_filename
                
                relative_path, dest_path = self.get_new_image_path(dataset_name, img_filename, new_id)
                
                if not source_path.exists():
                    self.stats[dataset_name]["missing_images"] += 1
                    if not self.dry_run:
                        continue
                
                if self.copy_image(source_path, dest_path):
                    new_img = {
                        "id": new_id,
                        "file_name": relative_path,
                        "width": img.get('width', 0),
                        "height": img.get('height', 0),
                        "dataset": dataset_name,
                        "split": split,
                        "modality": modality,
                        "original_filename": img_filename
                    }
                    self.combined_data['images'].append(new_img)
                    self.stats[dataset_name]["total_images"] += 1
                    if old_id in images_with_persons:
                        self.stats[dataset_name]["images_with_persons"] += 1
                    self.stats[dataset_name]["modalities"].add(modality)
                    self.stats[dataset_name]["splits"].add(split)
            
            # Process annotations
            for ann in data.get('annotations', []):
                if ann['category_id'] not in category_mapping:
                    continue
                if ann['image_id'] not in image_id_mapping:
                    continue
                
                old_cat_name = next((c['name'] for c in data['categories'] if c['id'] == ann['category_id']), '')
                is_crowd = self.is_crowd_category(old_cat_name) or ann.get('iscrowd', 0) == 1
                
                new_ann = {
                    "id": self.annotation_id_counter,
                    "image_id": image_id_mapping[ann['image_id']],
                    "category_id": 0,  # All person categories -> 0
                    "bbox": ann['bbox'],
                    "area": ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                    "segmentation": ann.get('segmentation', []),
                    "iscrowd": 1 if is_crowd else 0
                }
                self.combined_data['annotations'].append(new_ann)
                self.annotation_id_counter += 1
                self.stats[dataset_name]["total_annotations"] += 1
                if is_crowd:
                    self.stats[dataset_name]["crowd_annotations"] += 1
    
    def process_search_and_rescue(self):
        """Process Search and Rescue dataset."""
        dataset_name = "search_and_rescue"
        base_path = Path("/mnt/archive/person_drone/search-and-rescue")
        
        splits = ["train", "valid", "test"]
        
        for split in splits:
            ann_path = base_path / f"{split}.json"
            if not ann_path.exists():
                logger.warning(f"Annotation file not found: {ann_path}")
                continue
            
            logger.info(f"Processing {dataset_name} - {split}")
            
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            # Map category IDs (human -> person)
            category_mapping = {}
            for cat in data.get('categories', []):
                if self.is_person_category(cat['name']):
                    category_mapping[cat['id']] = 0
            
            # Process images and annotations
            image_id_mapping = {}
            images_with_persons = set()
            
            # First pass: identify images with person annotations
            for ann in data.get('annotations', []):
                if ann['category_id'] in category_mapping:
                    images_with_persons.add(ann['image_id'])
            
            # Process ALL images (drone dataset - keep all images)
            for img in tqdm(data.get('images', []), desc=f"Processing {dataset_name} {split} images"):
                
                old_id = img['id']
                new_id = self.image_id_counter
                self.image_id_counter += 1
                image_id_mapping[old_id] = new_id
                
                img_filename = img['file_name']
                # Search and rescue has images in train/images, valid/images, test/images folders
                source_path = base_path / split / "images" / img_filename
                
                relative_path, dest_path = self.get_new_image_path(dataset_name, img_filename, new_id)
                
                if self.copy_image(source_path, dest_path):
                    new_img = {
                        "id": new_id,
                        "file_name": relative_path,
                        "width": img.get('width', 0),
                        "height": img.get('height', 0),
                        "dataset": dataset_name,
                        "split": split,
                        "original_filename": img_filename
                    }
                    self.combined_data['images'].append(new_img)
                    self.stats[dataset_name]["total_images"] += 1
                    if old_id in images_with_persons:
                        self.stats[dataset_name]["images_with_persons"] += 1
                    self.stats[dataset_name]["splits"].add(split)
            
            # Process annotations
            for ann in data.get('annotations', []):
                if ann['category_id'] not in category_mapping:
                    continue
                if ann['image_id'] not in image_id_mapping:
                    continue
                
                new_ann = {
                    "id": self.annotation_id_counter,
                    "image_id": image_id_mapping[ann['image_id']],
                    "category_id": 0,
                    "bbox": ann['bbox'],
                    "area": ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                    "segmentation": ann.get('segmentation', []),
                    "iscrowd": ann.get('iscrowd', 0)
                }
                self.combined_data['annotations'].append(new_ann)
                self.annotation_id_counter += 1
                self.stats[dataset_name]["total_annotations"] += 1
                if new_ann['iscrowd']:
                    self.stats[dataset_name]["crowd_annotations"] += 1
    
    def process_stanford_drone(self):
        """Process Stanford Drone dataset."""
        dataset_name = "stanford_drone"
        base_path = Path("/mnt/archive/person_drone/stanford_drone_coco")
        
        ann_path = base_path / "train.json"
        if not ann_path.exists():
            logger.warning(f"Annotation file not found: {ann_path}")
            return
        
        logger.info(f"Processing {dataset_name}")
        
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        # Map category IDs (Pedestrian, Biker, Skater -> person)
        category_mapping = {}
        for cat in data.get('categories', []):
            if self.is_person_category(cat['name']):
                category_mapping[cat['id']] = 0
        
        # Process images and annotations
        image_id_mapping = {}
        images_with_persons = set()
        
        # First pass: identify images with person annotations
        for ann in data.get('annotations', []):
            if ann['category_id'] in category_mapping:
                images_with_persons.add(ann['image_id'])
        
        # Process ALL images (drone dataset - keep all images)
        for img in tqdm(data.get('images', []), desc=f"Processing {dataset_name} images"):
            old_id = img['id']
            new_id = self.image_id_counter
            self.image_id_counter += 1
            image_id_mapping[old_id] = new_id
            
            img_filename = img['file_name']
            # Stanford drone has images in train_images folder
            possible_paths = [
                base_path / "train_images" / img_filename,
                base_path / img_filename,
                base_path / "images" / img_filename
            ]
            
            source_path = None
            for path in possible_paths:
                if path.exists():
                    source_path = path
                    break
            
            if source_path is None:
                logger.warning(f"Image not found in any expected location: {img_filename}")
                continue
            
            relative_path, dest_path = self.get_new_image_path(dataset_name, img_filename, new_id)
            
            if self.copy_image(source_path, dest_path):
                new_img = {
                    "id": new_id,
                    "file_name": relative_path,
                    "width": img.get('width', 0),
                    "height": img.get('height', 0),
                    "dataset": dataset_name,
                    "split": "train",
                    "original_filename": img_filename
                }
                self.combined_data['images'].append(new_img)
                self.stats[dataset_name]["total_images"] += 1
                if old_id in images_with_persons:
                    self.stats[dataset_name]["images_with_persons"] += 1
                self.stats[dataset_name]["splits"].add("train")
        
        # Process annotations
        for ann in data.get('annotations', []):
            if ann['category_id'] not in category_mapping:
                continue
            if ann['image_id'] not in image_id_mapping:
                continue
            
            new_ann = {
                "id": self.annotation_id_counter,
                "image_id": image_id_mapping[ann['image_id']],
                "category_id": 0,
                "bbox": ann['bbox'],
                "area": ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                "segmentation": ann.get('segmentation', []),
                "iscrowd": ann.get('iscrowd', 0)
            }
            self.combined_data['annotations'].append(new_ann)
            self.annotation_id_counter += 1
            self.stats[dataset_name]["total_annotations"] += 1
    
    def process_coco_format_dataset(self, dataset_name: str, base_path: Path, ann_filename: str = "annotations.json"):
        """Generic processor for COCO format datasets."""
        ann_path = base_path / ann_filename
        if not ann_path.exists():
            logger.warning(f"Annotation file not found: {ann_path}")
            return
        
        logger.info(f"Processing {dataset_name}")
        
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        # Map category IDs
        category_mapping = {}
        for cat in data.get('categories', []):
            if self.is_person_category(cat['name']):
                category_mapping[cat['id']] = 0
        
        # Process images and annotations
        image_id_mapping = {}
        images_with_persons = set()
        
        # First pass: identify images with person annotations
        for ann in data.get('annotations', []):
            if ann['category_id'] in category_mapping:
                images_with_persons.add(ann['image_id'])
        
        # Process ALL images (drone dataset - keep all images)
        for img in tqdm(data.get('images', []), desc=f"Processing {dataset_name} images"):
            old_id = img['id']
            new_id = self.image_id_counter
            self.image_id_counter += 1
            image_id_mapping[old_id] = new_id
            
            img_filename = img['file_name']
            # Most COCO datasets have the path included in file_name
            # Try different possible paths
            possible_paths = [
                base_path / img_filename,  # Full path as specified in JSON
                base_path / "images" / img_filename,
                base_path / Path(img_filename).name  # Just filename without path
            ]
            
            source_path = None
            for path in possible_paths:
                if path.exists():
                    source_path = path
                    break
            
            if source_path is None:
                logger.warning(f"Image not found: {img_filename}")
                continue
            
            relative_path, dest_path = self.get_new_image_path(dataset_name, Path(img_filename).name, new_id)
            
            if self.copy_image(source_path, dest_path):
                new_img = {
                    "id": new_id,
                    "file_name": relative_path,
                    "width": img.get('width', 0),
                    "height": img.get('height', 0),
                    "dataset": dataset_name,
                    "original_filename": img_filename
                }
                # Add split info if available
                if 'split' in img:
                    new_img['split'] = img['split']
                    self.stats[dataset_name]["splits"].add(img['split'])
                
                self.combined_data['images'].append(new_img)
                self.stats[dataset_name]["total_images"] += 1
                if old_id in images_with_persons:
                    self.stats[dataset_name]["images_with_persons"] += 1
        
        # Process annotations
        for ann in data.get('annotations', []):
            if ann['category_id'] not in category_mapping:
                continue
            if ann['image_id'] not in image_id_mapping:
                continue
            
            # Check for crowd based on category name
            old_cat = next((c for c in data['categories'] if c['id'] == ann['category_id']), None)
            is_crowd = ann.get('iscrowd', 0)
            if old_cat and self.is_crowd_category(old_cat['name']):
                is_crowd = 1
            
            new_ann = {
                "id": self.annotation_id_counter,
                "image_id": image_id_mapping[ann['image_id']],
                "category_id": 0,
                "bbox": ann['bbox'],
                "area": ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                "segmentation": ann.get('segmentation', []),
                "iscrowd": is_crowd
            }
            self.combined_data['annotations'].append(new_ann)
            self.annotation_id_counter += 1
            self.stats[dataset_name]["total_annotations"] += 1
            if is_crowd:
                self.stats[dataset_name]["crowd_annotations"] += 1
    
    def process_visdrone(self):
        """Process VisDrone2019-DET dataset."""
        dataset_name = "visdrone2019"
        base_path = Path("/mnt/archive/person_drone/VisDrone2019-DET")
        
        ann_path = base_path / "annotations.json"
        if not ann_path.exists():
            logger.warning(f"Annotation file not found: {ann_path}")
            return
        
        logger.info(f"Processing {dataset_name}")
        
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        # Map category IDs (pedestrian, people -> person)
        category_mapping = {}
        for cat in data.get('categories', []):
            if self.is_person_category(cat['name']):
                category_mapping[cat['id']] = 0
        
        # Process images and annotations
        image_id_mapping = {}
        images_with_persons = set()
        
        # First pass: identify images with person annotations
        for ann in data.get('annotations', []):
            if ann['category_id'] in category_mapping:
                images_with_persons.add(ann['image_id'])
        
        # Process ALL images (drone dataset - keep all images)
        for img in tqdm(data.get('images', []), desc=f"Processing {dataset_name} images"):
            old_id = img['id']
            new_id = self.image_id_counter
            self.image_id_counter += 1
            image_id_mapping[old_id] = new_id
            
            img_filename = img['file_name']
            # VisDrone has images in train_images, val_images, test_images folders
            # The file_name already includes the folder (e.g., "train_images/xxx.jpg")
            possible_paths = [
                base_path / img_filename,  # This should work as file_name includes the folder
                base_path / "images" / img_filename
            ]
            
            source_path = None
            for path in possible_paths:
                if path.exists():
                    source_path = path
                    break
            
            if source_path is None:
                logger.warning(f"Image not found: {img_filename}")
                continue
            
            relative_path, dest_path = self.get_new_image_path(dataset_name, Path(img_filename).name, new_id)
            
            if self.copy_image(source_path, dest_path):
                new_img = {
                    "id": new_id,
                    "file_name": relative_path,
                    "width": img.get('width', 0),
                    "height": img.get('height', 0),
                    "dataset": dataset_name,
                    "original_filename": img_filename
                }
                self.combined_data['images'].append(new_img)
                self.stats[dataset_name]["total_images"] += 1
                if old_id in images_with_persons:
                    self.stats[dataset_name]["images_with_persons"] += 1
        
        # Process annotations
        for ann in data.get('annotations', []):
            if ann['category_id'] not in category_mapping:
                continue
            if ann['image_id'] not in image_id_mapping:
                continue
            
            # Check if it's a crowd annotation (people category)
            old_cat = next((c for c in data['categories'] if c['id'] == ann['category_id']), None)
            is_crowd = ann.get('iscrowd', 0)
            if old_cat and old_cat['name'].lower() == 'people':
                is_crowd = 1
            
            new_ann = {
                "id": self.annotation_id_counter,
                "image_id": image_id_mapping[ann['image_id']],
                "category_id": 0,
                "bbox": ann['bbox'],
                "area": ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                "segmentation": ann.get('segmentation', []),
                "iscrowd": is_crowd
            }
            self.combined_data['annotations'].append(new_ann)
            self.annotation_id_counter += 1
            self.stats[dataset_name]["total_annotations"] += 1
            if is_crowd:
                self.stats[dataset_name]["crowd_annotations"] += 1
    
    def process_objects365(self):
        """Process Objects365 dataset (only person category)."""
        dataset_name = "objects365"
        base_path = Path("/mnt/archive/datasets/OpenDataLab___Objects365")
        
        # Process train split
        ann_path = base_path / "raw/Objects365/data/train/zhiyuan_objv2_train.json"
        if ann_path.exists():
            logger.info(f"Processing {dataset_name} - train split (this may take a while...)")
            
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            # Find person category ID
            person_cat_id = None
            for cat in data.get('categories', []):
                if cat['name'].lower() == 'person':
                    person_cat_id = cat['id']
                    break
            
            if person_cat_id is None:
                logger.warning("Person category not found in Objects365")
                return
            
            # Process images and annotations
            image_id_mapping = {}
            images_with_persons = set()
            
            # First pass: identify images with person annotations
            logger.info("Identifying images with person annotations...")
            for ann in tqdm(data.get('annotations', []), desc="Scanning annotations"):
                if ann['category_id'] == person_cat_id:
                    images_with_persons.add(ann['image_id'])
            
            logger.info(f"Found {len(images_with_persons)} images with persons")
            
            # Create image ID to image dict for faster lookup
            id_to_image = {img['id']: img for img in data.get('images', [])}
            
            # Process only images with persons
            processed = 0
            
            for img_id in tqdm(images_with_persons, desc=f"Processing {dataset_name} images"):
                if img_id not in id_to_image:
                    continue
                    
                img = id_to_image[img_id]
                old_id = img['id']
                new_id = self.image_id_counter
                self.image_id_counter += 1
                image_id_mapping[old_id] = new_id
                
                img_filename = img['file_name']
                # Objects365 image paths need adjustment
                # JSON has: "images/v1/patch8/objects365_v1_00420917.jpg"
                # Actual path: "train/patch8/objects365_v1_00420917.jpg"
                
                # Extract patch and filename from the path
                path_parts = Path(img_filename).parts
                if len(path_parts) >= 3:
                    # Get patch directory and filename
                    patch_dir = path_parts[-2]  # e.g., "patch8"
                    filename = path_parts[-1]   # e.g., "objects365_v1_00420917.jpg"
                    source_path = base_path / "raw/Objects365/data/train" / patch_dir / filename
                else:
                    source_path = base_path / "raw/Objects365/data" / img_filename
                
                if not source_path.exists():
                    # In dry run, we still want to count the image even if file doesn't exist
                    if not self.dry_run:
                        continue
                    else:
                        self.stats[dataset_name]["missing_images"] = self.stats[dataset_name].get("missing_images", 0) + 1
                
                relative_path, dest_path = self.get_new_image_path(dataset_name, Path(img_filename).name, new_id)
                
                # In dry run or if file exists, add to dataset
                if self.dry_run or self.copy_image(source_path, dest_path):
                    new_img = {
                        "id": new_id,
                        "file_name": relative_path,
                        "width": img.get('width', 0),
                        "height": img.get('height', 0),
                        "dataset": dataset_name,
                        "split": "train",
                        "original_filename": img_filename
                    }
                    self.combined_data['images'].append(new_img)
                    self.stats[dataset_name]["total_images"] += 1
                    self.stats[dataset_name]["images_with_persons"] += 1
                    self.stats[dataset_name]["splits"].add("train")
                    processed += 1
            
            # Process annotations
            for ann in tqdm(data.get('annotations', []), desc="Processing annotations"):
                if ann['category_id'] != person_cat_id:
                    continue
                if ann['image_id'] not in image_id_mapping:
                    continue
                
                new_ann = {
                    "id": self.annotation_id_counter,
                    "image_id": image_id_mapping[ann['image_id']],
                    "category_id": 0,
                    "bbox": ann['bbox'],
                    "area": ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                    "segmentation": ann.get('segmentation', []),
                    "iscrowd": ann.get('iscrowd', 0)
                }
                self.combined_data['annotations'].append(new_ann)
                self.annotation_id_counter += 1
                self.stats[dataset_name]["total_annotations"] += 1
    
    def visualize_samples(self, num_samples: int = 10, dataset_filter: str = None):
        """Visualize random samples with bounding boxes using cv2.imshow."""
        if len(self.combined_data['images']) == 0:
            logger.warning("No images to visualize")
            return
        
        # Create image_id to annotations mapping
        img_to_anns = defaultdict(list)
        for ann in self.combined_data['annotations']:
            img_to_anns[ann['image_id']].append(ann)
        
        # Sample random images that have annotations
        images_with_anns = [img for img in self.combined_data['images'] 
                           if img['id'] in img_to_anns]
        
        # Filter by dataset if specified
        if dataset_filter:
            images_with_anns = [img for img in images_with_anns 
                               if img.get('dataset', '').lower() == dataset_filter.lower()]
            if not images_with_anns:
                logger.warning(f"No images with annotations from dataset '{dataset_filter}'")
                return
        
        if not images_with_anns:
            logger.warning("No images with annotations to visualize")
            return
        
        num_samples = min(num_samples, len(images_with_anns))
        sampled_images = random.sample(images_with_anns, num_samples)
        
        logger.info(f"Visualizing {num_samples} sample images with bounding boxes...")
        logger.info("Press any key to see next image, 'q' to quit")
        
        for idx, img_info in enumerate(sampled_images):
            # Determine the actual image path based on dataset
            dataset_name = img_info.get('dataset', '')
            original_filename = img_info.get('original_filename', img_info['file_name'])
            
            # Find the source image path
            if self.dry_run or not (self.images_dir / img_info['file_name']).exists():
                # In dry-run mode or if copied image doesn't exist, load from original location
                source_path = None
                
                if dataset_name == "rgbt_drone_person":
                    base = Path("/mnt/archive/person_drone/RGBTDronePerson-20250828T031729Z-1-001/RGBTDronePerson/RGBTDronePerson")
                    split = img_info.get('split', 'train')
                    modality = img_info.get('modality', 'thermal')
                    if split == "sub_train":
                        split = "train"
                    source_path = base / split / modality / original_filename
                    
                elif dataset_name == "search_and_rescue":
                    base = Path("/mnt/archive/person_drone/search-and-rescue")
                    split = img_info.get('split', 'train')
                    source_path = base / split / "images" / Path(original_filename).name
                    
                elif dataset_name == "stanford_drone":
                    base = Path("/mnt/archive/person_drone/stanford_drone_coco")
                    source_path = base / "train_images" / original_filename
                    
                elif dataset_name == "vtsar":
                    base = Path("/mnt/archive/person_drone/vtsar_coco")
                    source_path = base / original_filename
                    
                elif dataset_name == "vtuav":
                    base = Path("/mnt/archive/person_drone/vtuav_coco")
                    source_path = base / original_filename
                    
                elif dataset_name == "wisard":
                    base = Path("/mnt/archive/person_drone/wisard_coco")
                    source_path = base / original_filename
                    
                elif dataset_name == "visdrone2019":
                    base = Path("/mnt/archive/person_drone/VisDrone2019-DET")
                    source_path = base / original_filename
                    
                elif dataset_name == "objects365":
                    base = Path("/mnt/archive/datasets/OpenDataLab___Objects365")
                    path_parts = Path(original_filename).parts
                    if len(path_parts) >= 3:
                        patch_dir = path_parts[-2]
                        filename = path_parts[-1]
                        source_path = base / "raw/Objects365/data/train" / patch_dir / filename
                    else:
                        source_path = base / "raw/Objects365/data" / original_filename
                
                if source_path and source_path.exists():
                    img = cv2.imread(str(source_path))
                else:
                    # Create placeholder if image not found
                    img = np.zeros((img_info.get('height', 480), 
                                  img_info.get('width', 640), 3), dtype=np.uint8)
                    img[:] = (50, 50, 50)
                    cv2.putText(img, "Image not found", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Load from copied location
                img_path = self.images_dir / img_info['file_name']
                img = cv2.imread(str(img_path))
            
            if img is None:
                # Create placeholder if loading failed
                img = np.zeros((img_info.get('height', 480), 
                              img_info.get('width', 640), 3), dtype=np.uint8)
                img[:] = (50, 50, 50)
                cv2.putText(img, "Failed to load image", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw annotations
            annotations = img_to_anns[img_info['id']]
            
            for ann in annotations:
                # Get bbox
                x, y, w, h = ann['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Choose color based on crowd flag
                if ann.get('iscrowd', 0):
                    color = (0, 165, 255)  # Orange for crowd
                    label = "crowd"
                else:
                    color = (0, 255, 0)  # Green for individual person
                    label = "person"
                
                # Draw rectangle
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x, y - label_size[1] - 4), 
                            (x + label_size[0], y), color, -1)
                cv2.putText(img, label, (x, y - 2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add image info
            info_text = [
                f"Dataset: {img_info.get('dataset', 'unknown')}",
                f"Image ID: {img_info['id']}",
                f"Annotations: {len(annotations)}",
                f"Size: {img_info.get('width', 0)}x{img_info.get('height', 0)}"
            ]
            
            if 'split' in img_info:
                info_text.append(f"Split: {img_info['split']}")
            if 'modality' in img_info:
                info_text.append(f"Modality: {img_info['modality']}")
            
            # Draw info background
            y_offset = 10
            for text in info_text:
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (5, y_offset), 
                            (10 + text_size[0], y_offset + text_size[1] + 5),
                            (0, 0, 0), -1)
                cv2.putText(img, text, (10, y_offset + text_size[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += text_size[1] + 10
            
            # Display image
            window_name = f"Sample {idx+1}/{num_samples} - {img_info.get('dataset', 'unknown')} - ID: {img_info['id']}"
            cv2.imshow(window_name, img)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(window_name)
            
            if key == ord('q'):
                logger.info("Visualization stopped by user")
                break
        
        cv2.destroyAllWindows()
        logger.info("Visualization complete")
    
    def save_combined_dataset(self):
        """Save the combined dataset to disk."""
        if self.dry_run:
            logger.info(f"DRY RUN: Would save combined dataset to {self.annotations_file}")
            logger.info(f"DRY RUN: Dataset would contain {len(self.combined_data['images'])} images and {len(self.combined_data['annotations'])} annotations")
        else:
            logger.info(f"Saving combined dataset to {self.annotations_file}")
            with open(self.annotations_file, 'w') as f:
                json.dump(self.combined_data, f)
            logger.info("Dataset saved successfully")
    
    def print_statistics(self):
        """Print detailed statistics about the combined dataset."""
        print("\n" + "="*80)
        if self.dry_run:
            print("COMBINED DATASET STATISTICS (DRY RUN)")
        else:
            print("COMBINED DATASET STATISTICS")
        print("="*80)
        
        total_images = len(self.combined_data['images'])
        total_annotations = len(self.combined_data['annotations'])
        
        print(f"\nOVERALL:")
        print(f"  Total Images: {total_images:,}")
        print(f"  Total Annotations: {total_annotations:,}")
        print(f"  Average Annotations per Image: {total_annotations/max(total_images, 1):.2f}")
        
        print("\n" + "-"*80)
        print("PER-DATASET BREAKDOWN:")
        print("-"*80)
        
        # Calculate percentages
        for dataset_name, stats in sorted(self.stats.items()):
            img_pct = 100 * stats['images_with_persons'] / max(total_images, 1)
            ann_pct = 100 * stats['total_annotations'] / max(total_annotations, 1)
            
            print(f"\n{dataset_name.upper()}:")
            print(f"  Total images: {stats['total_images']:,}")
            print(f"  Images with persons: {stats['images_with_persons']:,} ({img_pct:.1f}% of total)")
            print(f"  Total annotations: {stats['total_annotations']:,} ({ann_pct:.1f}% of total)")
            if stats.get('missing_images', 0) > 0:
                print(f"  ⚠️  Missing images: {stats['missing_images']:,}")
            if stats['total_annotations'] > 0:
                print(f"  Crowd annotations: {stats['crowd_annotations']:,} ({100*stats['crowd_annotations']/stats['total_annotations']:.1f}%)")
            if stats['modalities']:
                print(f"  Modalities: {', '.join(sorted(stats['modalities']))}")
            if stats['splits']:
                print(f"  Splits: {', '.join(sorted(stats['splits']))}")
            if stats['images_with_persons'] > 0:
                print(f"  Avg annotations/image: {stats['total_annotations']/stats['images_with_persons']:.2f}")
        
        # Image statistics
        if self.combined_data['images']:
            widths = [img['width'] for img in self.combined_data['images'] if img['width'] > 0]
            heights = [img['height'] for img in self.combined_data['images'] if img['height'] > 0]
            if widths and heights:
                print("\n" + "-"*80)
                print("IMAGE DIMENSIONS:")
                print("-"*80)
                print(f"  Width range: {min(widths)} - {max(widths)} pixels")
                print(f"  Height range: {min(heights)} - {max(heights)} pixels")
                print(f"  Average width: {sum(widths)/len(widths):.0f} pixels")
                print(f"  Average height: {sum(heights)/len(heights):.0f} pixels")
        
        # Annotation statistics
        if self.combined_data['annotations']:
            areas = [ann['area'] for ann in self.combined_data['annotations'] if ann['area'] > 0]
            crowd_count = sum(1 for ann in self.combined_data['annotations'] if ann['iscrowd'] == 1)
            
            print("\n" + "-"*80)
            print("ANNOTATION STATISTICS:")
            print("-"*80)
            print(f"  Total bounding boxes: {len(self.combined_data['annotations']):,}")
            print(f"  Crowd annotations: {crowd_count:,} ({100*crowd_count/len(self.combined_data['annotations']):.1f}%)")
            if areas:
                print(f"  Area range: {min(areas):.0f} - {max(areas):.0f} pixels²")
                print(f"  Average area: {sum(areas)/len(areas):.0f} pixels²")
        
        # Dataset contribution summary table
        print("\n" + "-"*80)
        print("DATASET CONTRIBUTION SUMMARY:")
        print("-"*80)
        print(f"{'Dataset':<25} {'Total Images':>12} {'With Person':>12} {'%':>7} {'Annotations':>12} {'%':>7}")
        print("-"*100)
        
        # Only show datasets that actually have images
        datasets_with_images = [(name, stats) for name, stats in self.stats.items() if stats['total_images'] > 0]
        
        for dataset_name, stats in sorted(datasets_with_images, key=lambda x: x[1]['total_images'], reverse=True):
            img_pct = 100 * stats['total_images'] / max(total_images, 1)
            ann_pct = 100 * stats['total_annotations'] / max(total_annotations, 1)
            print(f"{dataset_name:<25} {stats['total_images']:>12,} {stats['images_with_persons']:>12,} {img_pct:>6.1f}% {stats['total_annotations']:>12,} {ann_pct:>6.1f}%")
        
        print("-"*100)
        
        # Calculate totals for images with persons
        total_with_persons = sum(stats['images_with_persons'] for stats in self.stats.values())
        print(f"{'TOTAL':<25} {total_images:>12,} {total_with_persons:>12,} {'100.0%':>7} {total_annotations:>12,} {'100.0%':>7}")
        
        print("\n" + "="*80)
    
    def check_datasets(self):
        """Check which datasets are available."""
        datasets = {
            "RGBTDronePerson": Path("/mnt/archive/person_drone/RGBTDronePerson-20250828T031729Z-1-001/RGBTDronePerson"),
            "search-and-rescue": Path("/mnt/archive/person_drone/search-and-rescue"),
            "stanford_drone_coco": Path("/mnt/archive/person_drone/stanford_drone_coco"),
            "vtsar_coco": Path("/mnt/archive/person_drone/vtsar_coco"),
            "vtuav_coco": Path("/mnt/archive/person_drone/vtuav_coco"),
            "wisard_coco": Path("/mnt/archive/person_drone/wisard_coco"),
            "VisDrone2019-DET": Path("/mnt/archive/person_drone/VisDrone2019-DET"),
            "Objects365": Path("/mnt/archive/datasets/OpenDataLab___Objects365")
        }
        
        print("\n" + "="*80)
        print("CHECKING DATASET AVAILABILITY")
        print("="*80)
        
        available = []
        missing = []
        
        for name, path in datasets.items():
            if path.exists():
                available.append(name)
                print(f"✅ {name}: Found at {path}")
            else:
                missing.append(name)
                print(f"❌ {name}: Not found at {path}")
        
        print(f"\nSummary: {len(available)} available, {len(missing)} missing")
        print("="*80 + "\n")
        
        return available, missing
    
    def run(self, visualize: bool = False, num_vis_samples: int = 10, vis_dataset: str = None):
        """Run the complete dataset combination pipeline."""
        # Check dataset availability
        available, missing = self.check_datasets()
        
        if missing and not self.dry_run:
            response = input(f"\n⚠️  {len(missing)} dataset(s) missing. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                logger.info("Aborted by user")
                return
        
        logger.info("Starting dataset combination process...")
        
        # Process each dataset
        self.process_rgbt_drone_person()
        self.process_search_and_rescue()
        self.process_stanford_drone()
        
        # Process COCO format datasets
        self.process_coco_format_dataset("vtsar", Path("/mnt/archive/person_drone/vtsar_coco"))
        self.process_coco_format_dataset("vtuav", Path("/mnt/archive/person_drone/vtuav_coco"))
        self.process_coco_format_dataset("wisard", Path("/mnt/archive/person_drone/wisard_coco"))
        
        # Process VisDrone
        self.process_visdrone()
        
        # Process Objects365 (limited due to size)
        self.process_objects365()
        
        # Save combined dataset
        self.save_combined_dataset()
        
        # Create visualizations if requested
        if visualize:
            self.visualize_samples(num_vis_samples, dataset_filter=vis_dataset)
        
        # Print statistics
        self.print_statistics()
        
        logger.info("Dataset combination complete!")


def main():
    parser = argparse.ArgumentParser(description="Combine multiple person detection datasets into a single COCO format dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/archive/person_drone/combined_dataset",
        help="Output directory for the combined dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without copying images to check correctness"
    )
    parser.add_argument(
        "--skip-objects365",
        action="store_true",
        help="Skip Objects365 dataset (it's very large)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization samples with bounding boxes"
    )
    parser.add_argument(
        "--num-vis-samples",
        type=int,
        default=200,
        help="Number of samples to visualize with cv2.imshow (default: 20)"
    )
    parser.add_argument(
        "--vis-dataset",
        type=str,
        default=None,
        help="Visualize samples only from specific dataset (e.g., stanford_drone, visdrone2019)"
    )
    parser.add_argument(
        "--images-per-folder",
        type=int,
        default=10000,
        help="Number of images per folder for pagination (default: 10000)"
    )
    
    args = parser.parse_args()
    
    combiner = DatasetCombiner(args.output_dir, dry_run=args.dry_run, images_per_folder=args.images_per_folder)
    
    if args.skip_objects365:
        # Override the process_objects365 method to skip it
        combiner.process_objects365 = lambda: logger.info("Skipping Objects365 dataset")
    
    combiner.run(visualize=args.visualize, num_vis_samples=args.num_vis_samples, vis_dataset=args.vis_dataset)


if __name__ == "__main__":
    main()
