#!/usr/bin/env python3
"""
Convert VTSaR dataset from YOLO format to COCO format.

Dataset structure:
/mnt/archive/person_drone/VTSaR/VTSaR_Crop_640/
├── rgb/
│   ├── train/
│   └── val/
├── ir/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/

YOLO format: class_id center_x center_y width height (normalized 0-1)
COCO format: x y width height (absolute coordinates)

VTSaR is a single-class dataset for person detection (class 0 = person).
"""

import json
import os
import argparse
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from datetime import datetime


def yolo_to_coco_bbox(yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert YOLO bbox format to COCO format.
    
    Args:
        yolo_bbox: [center_x, center_y, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        [x, y, width, height] in absolute coordinates for COCO format
    """
    center_x, center_y, width, height = yolo_bbox
    
    # Convert normalized coordinates to absolute
    abs_center_x = center_x * img_width
    abs_center_y = center_y * img_height
    abs_width = width * img_width
    abs_height = height * img_height
    
    # Convert center coordinates to top-left corner
    x = abs_center_x - abs_width / 2
    y = abs_center_y - abs_height / 2
    
    # Ensure coordinates are within image bounds
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    abs_width = min(abs_width, img_width - x)
    abs_height = min(abs_height, img_height - y)
    
    return [x, y, abs_width, abs_height]


def parse_yolo_annotation(label_file: str) -> List[List[float]]:
    """
    Parse YOLO annotation file.
    
    Args:
        label_file: Path to .txt annotation file
    
    Returns:
        List of [class_id, center_x, center_y, width, height] for each detection
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
                # Parse space-separated values
                values = list(map(float, line.split()))
                
                # Each detection has 5 values: class_id, center_x, center_y, width, height
                if len(values) != 5:
                    print(f"Warning: Unexpected format in {label_file} line {line_num + 1}, expected 5 values, got {len(values)}: {line}")
                    continue
                
                annotations.append(values)
                
            except ValueError as e:
                print(f"Warning: Could not parse line {line_num + 1} in {label_file}: {line} - {e}")
                continue
    
    return annotations


def convert_split_to_coco(
    dataset_root: str,
    split_name: str,
    modality: str,
    category_mapping: Dict[int, Dict],
    start_img_id: int = 1,
    start_ann_id: int = 1
) -> Tuple[List[Dict], List[Dict], int, int]:
    """
    Convert a VTSaR dataset split and modality to COCO format components.
    
    Args:
        dataset_root: Root directory of VTSaR dataset
        split_name: Name of the split (train/val)
        modality: Modality to use ('rgb' or 'ir')
        category_mapping: Mapping of class_id to category info
        start_img_id: Starting image ID for this split
        start_ann_id: Starting annotation ID for this split
    
    Returns:
        images_list, annotations_list, next_img_id, next_ann_id
    """
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / modality / split_name
    labels_dir = dataset_root / "labels" / split_name
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Warning: Missing images or labels directory for {split_name} {modality}")
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
    
    print(f"Processing {len(image_files)} images in {split_name} split ({modality} modality)...")
    
    for img_file in image_files:
        # Get corresponding label file
        label_file = labels_dir / f"{img_file.stem}.txt"
        
        # Open image to get dimensions
        try:
            with Image.open(img_file) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {img_file}: {e}")
            continue
        
        # Add image info with modality prefix to avoid filename conflicts
        relative_path = f"{split_name}_{modality}_images/{img_file.name}"
        images_list.append({
            "id": img_id,
            "file_name": relative_path,
            "width": img_width,
            "height": img_height,
            "license": 1,
            "modality": modality  # Add modality info
        })
        
        # Parse annotations
        yolo_annotations = parse_yolo_annotation(str(label_file))
        
        for yolo_ann in yolo_annotations:
            class_id, center_x, center_y, width, height = yolo_ann
            class_id = int(class_id)
            
            # Skip unknown classes
            if class_id not in category_mapping:
                continue
            
            # Convert bbox to COCO format
            coco_bbox = yolo_to_coco_bbox([center_x, center_y, width, height], img_width, img_height)
            
            # Calculate area
            area = coco_bbox[2] * coco_bbox[3]
            
            if area > 0:  # Only add valid annotations
                annotations_list.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_mapping[class_id]["id"],
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                    "modality": modality  # Add modality info
                })
                ann_id += 1
        
        img_id += 1
    
    return images_list, annotations_list, img_id, ann_id


def main():
    parser = argparse.ArgumentParser(description="Convert VTSaR dataset to COCO format")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/archive/person_drone/VTSaR/VTSaR_Crop_640",
        help="Path to VTSaR dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/svakhreev/projects/DEIM/data/vtsar_coco",
        help="Output directory for COCO format files"
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs='+',
        default=["rgb", "ir"],
        choices=["rgb", "ir"],
        help="Modalities to convert (default: both rgb and ir)"
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create category mapping for VTSaR (single class: person)
    category_mapping = {
        0: {
            "id": 1,  # COCO categories start from 1
            "name": "person",
            "supercategory": "person"
        }
    }
    
    print(f"Categories: {category_mapping}")
    print(f"Converting modalities: {args.modalities}")
    
    # Initialize combined COCO data structure
    coco_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": f"VTSaR Dataset - Combined train/val splits (RGB+IR modalities) in COCO format",
            "contributor": "VTSaR Dataset",
            "url": "",
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
    
    # Process each split and modality combination
    splits = ["train", "val"]
    img_id = 1
    ann_id = 1
    
    for split in splits:
        print(f"\nConverting {split} split...")
        
        for modality in args.modalities:
            split_images_dir = dataset_root / modality / split
            if not split_images_dir.exists():
                print(f"Warning: {split_images_dir} does not exist, skipping {split} {modality}")
                continue
            
            images_list, annotations_list, next_img_id, next_ann_id = convert_split_to_coco(
                str(dataset_root),
                split,
                modality,
                category_mapping,
                img_id,
                ann_id
            )
            
            # Add to combined dataset
            coco_data["images"].extend(images_list)
            coco_data["annotations"].extend(annotations_list)
            
            # Update IDs for next modality/split
            img_id = next_img_id
            ann_id = next_ann_id
            
            print(f"Added {len(images_list)} images and {len(annotations_list)} annotations from {split} {modality}")
    
    # Save unified COCO annotation file
    output_file = output_dir / "annotations.json"
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total: {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    print(f"Saved unified COCO format to {output_file}")
    
    # Create symlinks to original image directories for easy access
    for split in splits:
        for modality in args.modalities:
            original_images_dir = dataset_root / modality / split
            symlink_dir = output_dir / f"{split}_{modality}_images"
            
            if original_images_dir.exists() and not symlink_dir.exists():
                try:
                    symlink_dir.symlink_to(original_images_dir.resolve())
                    print(f"Created symlink: {symlink_dir} -> {original_images_dir}")
                except Exception as e:
                    print(f"Warning: Could not create symlink for {split} {modality}: {e}")
    
    # Save dataset information for reference
    info_file = output_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            'modalities': args.modalities,
            'splits_processed': [(split, modality) for split in splits for modality in args.modalities 
                               if (dataset_root / modality / split).exists()],
            'total_images': len(coco_data['images']),
            'total_annotations': len(coco_data['annotations']),
            'categories': category_mapping,
            'args': vars(args)
        }, f, indent=2)
    
    print(f"Dataset info saved to {info_file}")


if __name__ == "__main__":
    main()
