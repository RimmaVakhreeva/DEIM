#!/usr/bin/env python3
"""
Convert WiSARDv1 (Wildland Search and Rescue Dataset) from YOLO format to COCO format.

Dataset structure:
/mnt/archive/person_drone/WiSARDv1/
├── [DATE]_[LOCATION]_[SENSOR]_[TYPE]/
│   ├── [IMAGE_NAME].jpg
│   ├── [IMAGE_NAME].txt  (YOLO annotations)
│   └── count.txt  (optional statistics)

Where:
- DATE: Format YYMMDD (e.g., 200321)
- LOCATION: Location name (e.g., Baker, Carnation, etc.)
- SENSOR: Sensor type (Phantom, Inspire, Mavic_Mini, FLIR, Enterprise)
- TYPE: VIS (visible/RGB), IR (infrared), or sequence number

YOLO format: class_id center_x center_y width height (normalized 0-1)
COCO format: x y width height (absolute coordinates)

WiSARD is a single-class dataset focused on person detection in search and rescue scenarios.
Class 0 = person (human)
"""

import json
import os
import argparse
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from datetime import datetime
import re


def get_wisard_categories() -> Dict[int, Dict]:
    """
    Get WiSARD category mapping.
    WiSARD is a single-class dataset for person detection in search and rescue scenarios.
    """
    category_mapping = {
        0: {
            "id": 1,  # COCO categories start from 1
            "name": "person",
            "supercategory": "person"
        }
    }
    
    return category_mapping


def parse_sequence_info(sequence_name: str) -> Dict[str, str]:
    """
    Parse sequence directory name to extract metadata.
    
    Args:
        sequence_name: Directory name like "200321_Baker_Phantom_VIS"
    
    Returns:
        Dictionary with parsed metadata
    """
    # Extract basic information from sequence name
    parts = sequence_name.split('_')
    
    info = {
        'sequence_name': sequence_name,
        'date': '',
        'location': '',
        'sensor': '',
        'modality': 'unknown'
    }
    
    if len(parts) >= 2:
        info['date'] = parts[0]
        info['location'] = parts[1]
    
    # Determine modality (VIS = RGB, IR = infrared)
    if 'VIS' in sequence_name:
        info['modality'] = 'rgb'
    elif 'IR' in sequence_name:
        info['modality'] = 'ir'
    
    # Extract sensor information
    for part in parts:
        if part in ['Phantom', 'Inspire', 'Mavic', 'FLIR', 'Enterprise']:
            info['sensor'] = part
            break
    
    return info


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


def convert_sequence_to_coco(
    sequence_path: str,
    sequence_info: Dict[str, str],
    category_mapping: Dict[int, Dict],
    start_img_id: int = 1,
    start_ann_id: int = 1
) -> Tuple[List[Dict], List[Dict], int, int]:
    """
    Convert a WiSARD sequence to COCO format components.
    
    Args:
        sequence_path: Path to sequence directory
        sequence_info: Parsed sequence metadata
        category_mapping: Mapping of class_id to category info
        start_img_id: Starting image ID for this sequence
        start_ann_id: Starting annotation ID for this sequence
    
    Returns:
        images_list, annotations_list, next_img_id, next_ann_id
    """
    sequence_path = Path(sequence_path)
    
    if not sequence_path.exists():
        print(f"Warning: Sequence path does not exist: {sequence_path}")
        return [], [], start_img_id, start_ann_id
    
    images_list = []
    annotations_list = []
    img_id = start_img_id
    ann_id = start_ann_id
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(sequence_path.glob(f"*{ext}"))
        image_files.extend(sequence_path.glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    print(f"Processing {len(image_files)} images in sequence {sequence_info['sequence_name']} ({sequence_info['modality']} modality)...")
    
    for img_file in image_files:
        # Get corresponding annotation file
        label_file = sequence_path / f"{img_file.stem}.txt"
        
        # Open image to get dimensions
        try:
            with Image.open(img_file) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {img_file}: {e}")
            continue
        
        # Add image info with sequence prefix to organize and avoid filename conflicts
        relative_path = f"{sequence_info['sequence_name']}/{img_file.name}"
        images_list.append({
            "id": img_id,
            "file_name": relative_path,
            "width": img_width,
            "height": img_height,
            "license": 1,
            "sequence": sequence_info['sequence_name'],
            "modality": sequence_info['modality'],
            "location": sequence_info['location'],
            "sensor": sequence_info['sensor'],
            "date": sequence_info['date']
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
                    "sequence": sequence_info['sequence_name'],
                    "modality": sequence_info['modality']
                })
                ann_id += 1
        
        img_id += 1
    
    return images_list, annotations_list, img_id, ann_id


def get_all_sequences(dataset_root: str) -> List[str]:
    """Get all sequence directories in the dataset."""
    dataset_path = Path(dataset_root)
    sequences = []
    
    for item in dataset_path.iterdir():
        if item.is_dir():
            sequences.append(item.name)
    
    return sorted(sequences)


def main():
    parser = argparse.ArgumentParser(description="Convert WiSARD v1 dataset to COCO format")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/archive/person_drone/WiSARDv1",
        help="Path to WiSARD dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/svakhreev/projects/DEIM/data/wisard_coco",
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
    
    # Get category mapping
    category_mapping = get_wisard_categories()
    print(f"Categories: {category_mapping}")
    print(f"Converting modalities: {args.modalities}")
    
    # Get all sequences
    all_sequences = get_all_sequences(str(dataset_root))
    print(f"Found {len(all_sequences)} sequences")
    
    # Filter sequences by modality
    sequences_to_process = []
    for sequence in all_sequences:
        seq_info = parse_sequence_info(sequence)
        if seq_info['modality'] in args.modalities:
            sequences_to_process.append(sequence)
    
    print(f"Processing {len(sequences_to_process)} sequences matching requested modalities")
    
    # Initialize combined COCO data structure
    coco_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "WiSARD v1 Dataset - Wildland Search and Rescue in COCO format",
            "contributor": "WiSARD Dataset",
            "url": "https://github.com/castacks/WiSARDdataset",
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
    
    # Process each sequence
    img_id = 1
    ann_id = 1
    modality_counts = {"rgb": 0, "ir": 0}
    
    for sequence in sequences_to_process:
        sequence_path = dataset_root / sequence
        sequence_info = parse_sequence_info(sequence)
        
        if not sequence_path.exists():
            print(f"Warning: Sequence path does not exist: {sequence_path}")
            continue
        
        print(f"\nConverting sequence: {sequence}")
        
        images_list, annotations_list, next_img_id, next_ann_id = convert_sequence_to_coco(
            str(sequence_path),
            sequence_info,
            category_mapping,
            img_id,
            ann_id
        )
        
        # Add to combined dataset
        coco_data["images"].extend(images_list)
        coco_data["annotations"].extend(annotations_list)
        
        # Update IDs for next sequence
        img_id = next_img_id
        ann_id = next_ann_id
        
        # Track modality counts
        modality_counts[sequence_info['modality']] += len(images_list)
        
        print(f"Added {len(images_list)} images and {len(annotations_list)} annotations from {sequence}")
    
    # Save unified COCO annotation file
    output_file = output_dir / "annotations.json"
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total: {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    print(f"Modality distribution: RGB={modality_counts['rgb']}, IR={modality_counts['ir']}")
    print(f"Saved unified COCO format to {output_file}")
    
    # Create symlinks to original sequence directories for easy access
    symlinks_created = 0
    for sequence in sequences_to_process:
        original_sequence_dir = dataset_root / sequence
        symlink_dir = output_dir / sequence
        
        if original_sequence_dir.exists() and not symlink_dir.exists():
            try:
                symlink_dir.symlink_to(original_sequence_dir.resolve())
                symlinks_created += 1
            except Exception as e:
                print(f"Warning: Could not create symlink for {sequence}: {e}")
    
    print(f"Created {symlinks_created} symlinks to original sequence directories")
    
    # Save dataset information for reference
    info_file = output_dir / "dataset_info.json"
    
    # Group sequences by modality and location for analysis
    sequences_by_modality = {"rgb": [], "ir": []}
    sequences_by_location = {}
    
    for sequence in sequences_to_process:
        seq_info = parse_sequence_info(sequence)
        sequences_by_modality[seq_info['modality']].append(sequence)
        
        location = seq_info['location']
        if location not in sequences_by_location:
            sequences_by_location[location] = []
        sequences_by_location[location].append(sequence)
    
    with open(info_file, 'w') as f:
        json.dump({
            'modalities': args.modalities,
            'total_sequences': len(sequences_to_process),
            'sequences_by_modality': sequences_by_modality,
            'sequences_by_location': sequences_by_location,
            'total_images': len(coco_data['images']),
            'total_annotations': len(coco_data['annotations']),
            'modality_counts': modality_counts,
            'categories': category_mapping,
            'args': vars(args)
        }, f, indent=2)
    
    print(f"Dataset info saved to {info_file}")
    print(f"Sequences by location: {dict(sorted(sequences_by_location.items()))}")


if __name__ == "__main__":
    main()
