#!/usr/bin/env python3
"""
Convert VTUAV_v1.0-001 dataset from Pascal VOC format to COCO format.

Dataset structure:
/mnt/archive/person_drone/VTUAV_v1.0-001/
├── train/
│   ├── rgb/
│   ├── ir/
│   └── anno/
└── test/
    ├── rgb/
    ├── ir/
    └── anno/

Annotation format: Pascal VOC XML format
- XML files contain bounding box annotations with <xmin>, <ymin>, <xmax>, <ymax> coordinates
- Class names are numeric (0, 1, 2, 3) - need to map to meaningful names
- Both RGB and IR modalities available for each image

VTUAV appears to be a multi-class UAV detection dataset focusing on pedestrian detection.
Based on the sequence names, classes likely represent different object types in UAV imagery.
"""

import json
import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from datetime import datetime


def get_vtuav_categories() -> Dict[str, Dict]:
    """
    Get VTUAV category mapping.
    Based on common UAV detection datasets, classes likely represent:
    0: person/pedestrian (most common based on sequence names)
    1: vehicle/car  
    2: bicycle/bike
    3: other/background
    """
    class_mapping = {
        "0": {
            "id": 1,  # COCO categories start from 1
            "name": "person",
            "supercategory": "person"
        },
        "1": {
            "id": 2,
            "name": "vehicle",
            "supercategory": "vehicle"
        },
        "2": {
            "id": 3,
            "name": "bicycle", 
            "supercategory": "vehicle"
        },
        "3": {
            "id": 4,
            "name": "other",
            "supercategory": "other"
        }
    }
    
    return class_mapping


def parse_voc_annotation(xml_file: str) -> Tuple[Dict, List[Dict]]:
    """
    Parse Pascal VOC format XML annotation file.
    
    Args:
        xml_file: Path to XML annotation file
    
    Returns:
        image_info dict, list of annotation dicts
    """
    # Default image info structure
    default_image_info = {
        'filename': "",
        'width': 0,
        'height': 0
    }
    
    if not os.path.exists(xml_file):
        return default_image_info.copy(), []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image information
        filename = root.find('filename').text if root.find('filename') is not None else ""
        
        size = root.find('size')
        if size is not None:
            width_elem = size.find('width')
            height_elem = size.find('height')
            width = int(width_elem.text) if width_elem is not None and width_elem.text else 0
            height = int(height_elem.text) if height_elem is not None and height_elem.text else 0
        else:
            width = height = 0
        
        image_info = {
            'filename': filename,
            'width': width,
            'height': height
        }
        
        # Get object annotations
        annotations = []
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            name = name_elem.text if name_elem is not None and name_elem.text else ""
            
            # Get bounding box
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                try:
                    xmin_elem = bndbox.find('xmin')
                    ymin_elem = bndbox.find('ymin')
                    xmax_elem = bndbox.find('xmax')
                    ymax_elem = bndbox.find('ymax')
                    
                    if all(elem is not None and elem.text for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                        xmin = float(xmin_elem.text)
                        ymin = float(ymin_elem.text)
                        xmax = float(xmax_elem.text)
                        ymax = float(ymax_elem.text)
                        
                        # Convert to COCO format: [x, y, width, height]
                        # Round coordinates to integers to reduce sub-pixel misalignments
                        x = round(xmin)
                        y = round(ymin)
                        width = round(xmax - xmin)
                        height = round(ymax - ymin)
                        
                        # Skip invalid bboxes
                        if width <= 0 or height <= 0:
                            continue
                        
                        annotations.append({
                            'class_name': name,
                            'bbox': [x, y, width, height],
                            'area': width * height
                        })
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not parse bounding box in {xml_file}: {e}")
                    continue
        
        return image_info, annotations
        
    except (ET.ParseError, Exception) as e:
        print(f"Warning: Could not parse XML file {xml_file}: {e}")
        return default_image_info.copy(), []


def convert_split_to_coco(
    dataset_root: str,
    split_name: str,
    modality: str,
    category_mapping: Dict[str, Dict],
    start_img_id: int = 1,
    start_ann_id: int = 1
) -> Tuple[List[Dict], List[Dict], int, int]:
    """
    Convert a VTUAV dataset split and modality to COCO format components.
    
    Args:
        dataset_root: Root directory of VTUAV dataset
        split_name: Name of the split (train/test)
        modality: Modality to use ('rgb' or 'ir')
        category_mapping: Mapping of class names to category info
        start_img_id: Starting image ID for this split
        start_ann_id: Starting annotation ID for this split
    
    Returns:
        images_list, annotations_list, next_img_id, next_ann_id
    """
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / split_name / modality
    annotations_dir = dataset_root / split_name / "anno"
    
    if not images_dir.exists() or not annotations_dir.exists():
        print(f"Warning: Missing images or annotations directory for {split_name} {modality}")
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
        # Get corresponding annotation file
        annotation_file = annotations_dir / f"{img_file.stem}.xml"
        
        # Parse XML annotation
        if not annotation_file.exists():
            print(f"Warning: No annotation file found for {img_file}, skipping")
            continue
            
        image_info, annotations = parse_voc_annotation(str(annotation_file))
        
        # Verify image dimensions by opening the image
        try:
            with Image.open(img_file) as img:
                actual_width, actual_height = img.size
                
            # Use actual dimensions if XML dimensions are missing or incorrect
            if image_info.get('width', 0) == 0 or image_info.get('height', 0) == 0:
                image_info['width'] = actual_width
                image_info['height'] = actual_height
                
        except Exception as e:
            print(f"Error opening image {img_file}: {e}")
            # Try to get dimensions from XML, if that fails too, skip this image
            if image_info.get('width', 0) == 0 or image_info.get('height', 0) == 0:
                print(f"Skipping {img_file} - no valid dimensions available")
                continue
        
        # Add image info with modality prefix to avoid filename conflicts
        relative_path = f"{split_name}_{modality}_images/{img_file.name}"
        images_list.append({
            "id": img_id,
            "file_name": relative_path,
            "width": image_info['width'],
            "height": image_info['height'],
            "license": 1,
            "modality": modality  # Add modality info
        })
        
        # Process annotations
        for annotation in annotations:
            class_name = annotation['class_name']
            
            # Skip unknown classes
            if class_name not in category_mapping:
                print(f"Warning: Unknown class '{class_name}' in {annotation_file}")
                continue
            
            bbox = annotation['bbox']
            x, y, width, height = bbox
            
            # Ensure bbox is within image bounds
            x = max(0, min(x, image_info['width'] - 1))
            y = max(0, min(y, image_info['height'] - 1))
            width = min(width, image_info['width'] - x)
            height = min(height, image_info['height'] - y)
            
            # Skip invalid bboxes
            if width <= 0 or height <= 0:
                continue
            
            area = width * height
            
            annotations_list.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_mapping[class_name]["id"],
                "bbox": [x, y, width, height],
                "area": area,
                "iscrowd": 0,
                "segmentation": [],
                "modality": modality  # Add modality info
            })
            ann_id += 1
        
        img_id += 1
    
    return images_list, annotations_list, img_id, ann_id


def main():
    parser = argparse.ArgumentParser(description="Convert VTUAV dataset to COCO format")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/archive/person_drone/VTUAV_v1.0-001",
        help="Path to VTUAV dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/svakhreev/projects/DEIM/data/vtuav_coco",
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
    category_mapping = get_vtuav_categories()
    print(f"Categories: {category_mapping}")
    print(f"Converting modalities: {args.modalities}")
    
    # Initialize combined COCO data structure
    coco_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "VTUAV Dataset v1.0-001 - Combined train/test splits (RGB+IR modalities) in COCO format",
            "contributor": "VTUAV Dataset",
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
    splits = ["train", "test"]
    img_id = 1
    ann_id = 1
    
    for split in splits:
        print(f"\nConverting {split} split...")
        
        for modality in args.modalities:
            split_images_dir = dataset_root / split / modality
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
            original_images_dir = dataset_root / split / modality
            symlink_dir = output_dir / f"{split}_{modality}_images"
            
            if original_images_dir.exists() and not symlink_dir.exists():
                try:
                    symlink_dir.symlink_to(original_images_dir.resolve())
                    print(f"Created symlink: {symlink_dir} -> {original_images_dir}")
                except Exception as e:
                    print(f"Warning: Could not create symlink for {split} {modality}: {e}")
    
    # Save dataset information for reference
    info_file = output_dir / "dataset_info.json"
    
    # Calculate class distribution
    class_distribution = {}
    for annotation in coco_data["annotations"]:
        cat_id = annotation["category_id"]
        cat_name = next(cat["name"] for cat in coco_data["categories"] if cat["id"] == cat_id)
        class_distribution[cat_name] = class_distribution.get(cat_name, 0) + 1
    
    with open(info_file, 'w') as f:
        json.dump({
            'modalities': args.modalities,
            'splits_processed': [(split, modality) for split in splits for modality in args.modalities 
                               if (dataset_root / split / modality).exists()],
            'total_images': len(coco_data['images']),
            'total_annotations': len(coco_data['annotations']),
            'class_distribution': class_distribution,
            'categories': category_mapping,
            'args': vars(args)
        }, f, indent=2)
    
    print(f"Dataset info saved to {info_file}")
    print(f"Class distribution: {class_distribution}")


if __name__ == "__main__":
    main()
