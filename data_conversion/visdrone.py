#!/usr/bin/env python3
"""
Convert VisDrone dataset from three separate splits to a single unified COCO format dataset.

Dataset structure:
/mnt/archive/person_drone/VisDrone2019-DET-train/
├── images/
└── annotations/
/mnt/archive/person_drone/VisDrone2019-DET-val/
├── images/
└── annotations/
/mnt/archive/person_drone/VisDrone2019-DET-test-dev/
├── images/
└── annotations/

VisDrone annotation format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
- bbox_left, bbox_top: Top-left corner coordinates (absolute pixels)
- bbox_width, bbox_height: Box dimensions (absolute pixels)
- score: Confidence score (usually 0 or 1)
- object_category: Object class ID (0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor)
- truncation: Truncation flag (0=not truncated, 1=truncated)
- occlusion: Occlusion flag (0=not occluded, 1=occluded)

COCO format: [x, y, width, height] (absolute coordinates for top-left corner)
"""

import json
import os
import argparse
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from datetime import datetime


def get_visdrone_categories() -> Dict[int, Dict]:
    """Get VisDrone category mapping."""
    # VisDrone categories - 0 is ignored regions, so we skip it
    class_names = {
        1: "pedestrian",
        2: "people", 
        3: "bicycle",
        4: "car",
        5: "van",
        6: "truck",
        7: "tricycle",
        8: "awning-tricycle", 
        9: "bus",
        10: "motor"
    }
    
    category_mapping = {}
    for class_id, name in class_names.items():
        category_mapping[class_id] = {
            "id": class_id,  # Keep original VisDrone IDs
            "name": name,
            "supercategory": "vehicle" if name not in ["pedestrian", "people"] else "person"
        }
    
    return category_mapping


def parse_visdrone_annotation(label_file: str) -> List[List[int]]:
    """
    Parse VisDrone annotation file.
    
    Args:
        label_file: Path to .txt annotation file
    
    Returns:
        List of [bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion] for each detection
    """
    annotations = []
    
    if not os.path.exists(label_file):
        return annotations
    
    with open(label_file, 'r') as f:
        content = f.read().strip()
        if not content:
            return annotations
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse comma-separated values, filter out empty strings from trailing commas
                parts = [part.strip() for part in line.split(',') if part.strip()]
                values = list(map(int, parts))
                
                # Each detection has 8 values: bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion
                if len(values) != 8:
                    print(f"Warning: Unexpected format in {label_file} line {line_num + 1}, expected 8 values, got {len(values)}: {line}")
                    continue
                
                annotations.append(values)
                
            except ValueError as e:
                print(f"Warning: Could not parse line {line_num + 1} in {label_file}: {line} - {e}")
                continue
    
    return annotations


def process_split_to_coco(
    split_dir: str,
    split_name: str, 
    category_mapping: Dict[int, Dict],
    start_img_id: int = 1,
    start_ann_id: int = 1
) -> Tuple[List[Dict], List[Dict], int, int]:
    """
    Process a VisDrone split and convert to COCO format components.
    
    Args:
        split_dir: Directory containing images/ and annotations/ subdirectories
        split_name: Name of the split (train/val/test-dev)
        category_mapping: Mapping of class_id to category info
        start_img_id: Starting image ID for this split
        start_ann_id: Starting annotation ID for this split
    
    Returns:
        images_list, annotations_list, next_img_id, next_ann_id
    """
    split_path = Path(split_dir)
    images_dir = split_path / "images"
    annotations_dir = split_path / "annotations"
    
    if not images_dir.exists() or not annotations_dir.exists():
        print(f"Warning: Missing images or annotations directory in {split_dir}")
        return [], [], start_img_id, start_ann_id
    
    images_list = []
    annotations_list = []
    img_id = start_img_id
    ann_id = start_ann_id
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    print(f"Processing {len(image_files)} images in {split_name} split...")
    
    for img_file in image_files:
        # Get corresponding annotation file
        annotation_file = annotations_dir / f"{img_file.stem}.txt"
        
        # Open image to get dimensions
        try:
            with Image.open(img_file) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {img_file}: {e}")
            continue
        
        # Add image info with split prefix to avoid filename conflicts
        relative_path = f"{split_name}_images/{img_file.name}"
        images_list.append({
            "id": img_id,
            "file_name": relative_path,
            "width": img_width,
            "height": img_height,
            "license": 1
        })
        
        # Parse annotations
        visdrone_annotations = parse_visdrone_annotation(str(annotation_file))
        
        for visdrone_ann in visdrone_annotations:
            bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion = visdrone_ann
            
            # Skip ignored regions (category 0)
            if object_category == 0:
                continue
                
            # Skip unknown categories
            if object_category not in category_mapping:
                continue
            
            # Ensure bbox is within image bounds
            bbox_left = max(0, min(bbox_left, img_width - 1))
            bbox_top = max(0, min(bbox_top, img_height - 1))
            bbox_width = min(bbox_width, img_width - bbox_left)
            bbox_height = min(bbox_height, img_height - bbox_top)
            
            # Skip invalid bboxes
            if bbox_width <= 0 or bbox_height <= 0:
                continue
            
            area = bbox_width * bbox_height
            
            annotations_list.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_mapping[object_category]["id"],
                "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],  # COCO format: [x, y, width, height]
                "area": area,
                "iscrowd": 0,
                "segmentation": [],
                # Additional VisDrone specific fields
                "score": score,
                "truncation": truncation,
                "occlusion": occlusion
            })
            ann_id += 1
        
        img_id += 1
    
    return images_list, annotations_list, img_id, ann_id


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone dataset splits to single unified COCO format")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/mnt/archive/person_drone/VisDrone2019-DET-train",
        help="Path to VisDrone train split directory"
    )
    parser.add_argument(
        "--val_dir", 
        type=str,
        default="/mnt/archive/person_drone/VisDrone2019-DET-val",
        help="Path to VisDrone val split directory"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/mnt/archive/person_drone/VisDrone2019-DET-test-dev",
        help="Path to VisDrone test-dev split directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/svakhreev/projects/DEIM/data/visdrone_coco",
        help="Output directory for unified COCO format files"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get category mapping
    category_mapping = get_visdrone_categories()
    print(f"Categories: {category_mapping}")
    
    # Initialize COCO data structure
    coco_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "VisDrone Dataset - Combined train/val/test-dev splits in COCO format",
            "contributor": "VisDrone Dataset",
            "url": "https://github.com/VisDrone/VisDrone-Dataset",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Academic Use",
                "url": ""
            }
        ],
        "categories": list(category_mapping.values()),
        "images": [],
        "annotations": []
    }
    
    # Process each split
    splits = [
        (args.train_dir, "train"),
        (args.val_dir, "val"), 
        (args.test_dir, "test")
    ]
    
    img_id = 1
    ann_id = 1
    
    for split_dir, split_name in splits:
        if not Path(split_dir).exists():
            print(f"Warning: {split_dir} does not exist, skipping {split_name}")
            continue
        
        print(f"\nProcessing {split_name} split from {split_dir}...")
        
        images_list, annotations_list, next_img_id, next_ann_id = process_split_to_coco(
            split_dir,
            split_name,
            category_mapping,
            img_id,
            ann_id
        )
        
        # Add to combined dataset
        coco_data["images"].extend(images_list)
        coco_data["annotations"].extend(annotations_list)
        
        # Update IDs for next split
        img_id = next_img_id
        ann_id = next_ann_id
        
        print(f"Added {len(images_list)} images and {len(annotations_list)} annotations from {split_name}")
    
    # Save unified COCO annotation file
    output_file = output_dir / "annotations.json"
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total: {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    print(f"Saved unified COCO format to {output_file}")
    
    # Create symlinks to original image directories for easy access
    for split_dir, split_name in splits:
        if not Path(split_dir).exists():
            continue
            
        original_images_dir = Path(split_dir) / "images"
        symlink_dir = output_dir / f"{split_name}_images"
        
        if original_images_dir.exists() and not symlink_dir.exists():
            try:
                symlink_dir.symlink_to(original_images_dir.resolve())
                print(f"Created symlink: {symlink_dir} -> {original_images_dir}")
            except Exception as e:
                print(f"Warning: Could not create symlink for {split_name}: {e}")
    
    # Save dataset information for reference
    info_file = output_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            'splits_processed': [split_name for split_dir, split_name in splits if Path(split_dir).exists()],
            'total_images': len(coco_data['images']),
            'total_annotations': len(coco_data['annotations']),
            'categories': category_mapping,
            'args': vars(args)
        }, f, indent=2)
    
    print(f"Dataset info saved to {info_file}")


if __name__ == "__main__":
    main()
