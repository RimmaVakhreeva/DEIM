#!/usr/bin/env python3
"""
Convert RGB-Drone Person Search and Rescue dataset from YOLO format to COCO format.

Dataset structure:
/mnt/archive/person_drone/search-and-rescue/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

YOLO format: class_id center_x center_y width height (normalized 0-1)
COCO format: x y width height (absolute coordinates)
"""

import json
import os
import argparse
from pathlib import Path
from PIL import Image
import yaml
from typing import Dict, List, Tuple
from datetime import datetime


def load_dataset_info(data_yaml_path: str) -> Dict:
    """Load dataset information from data.yaml file."""
    with open(data_yaml_path, 'r') as f:
        data_info = yaml.safe_load(f)
    return data_info


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

        # Split by lines and then by spaces to handle multiple detections per line
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse numbers from the line
            values = list(map(float, line.split()))

            # Each detection has 5 values: class_id, center_x, center_y, width, height
            if len(values) % 5 != 0:
                print(f"Warning: Unexpected format in {label_file}, skipping line: {line}")
                continue

            # Process each detection in groups of 5
            for i in range(0, len(values), 5):
                if i + 4 < len(values):
                    detection = values[i:i + 5]
                    annotations.append(detection)

    return annotations


def convert_split_to_coco(
        images_dir: str,
        labels_dir: str,
        split_name: str,
        category_mapping: Dict[int, Dict],
        start_img_id: int = 1,
        start_ann_id: int = 1
) -> Tuple[Dict, int, int]:
    """
    Convert a dataset split to COCO format.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO annotation files
        split_name: Name of the split (train/valid/test)
        category_mapping: Mapping of class_id to category info
        start_img_id: Starting image ID for this split
        start_ann_id: Starting annotation ID for this split

    Returns:
        COCO format dictionary, next_img_id, next_ann_id
    """
    coco_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": f"RGB-Drone Person Search and Rescue Dataset - {split_name} split",
            "contributor": "Search and Rescue Dataset",
            "url": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Private",
                "url": ""
            }
        ],
        "categories": list(category_mapping.values()),
        "images": [],
        "annotations": []
    }

    img_id = start_img_id
    ann_id = start_ann_id

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))
        image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)

    print(f"Processing {len(image_files)} images in {split_name} split...")

    for img_file in image_files:
        # Get corresponding label file
        label_file = Path(labels_dir) / f"{img_file.stem}.txt"

        # Open image to get dimensions
        try:
            with Image.open(img_file) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {img_file}: {e}")
            continue

        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": img_width,
            "height": img_height,
            "license": 1
        })

        # Parse annotations
        yolo_annotations = parse_yolo_annotation(str(label_file))

        for yolo_ann in yolo_annotations:
            class_id, center_x, center_y, width, height = yolo_ann
            class_id = int(class_id)

            # Convert bbox to COCO format
            coco_bbox = yolo_to_coco_bbox([center_x, center_y, width, height], img_width, img_height)

            # Calculate area
            area = coco_bbox[2] * coco_bbox[3]

            if area > 0:  # Only add valid annotations
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_mapping[class_id]["id"],
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1

        img_id += 1

    return coco_data, img_id, ann_id


def main():
    parser = argparse.ArgumentParser(description="Convert RGB-Drone Person dataset to COCO format")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/archive/person_drone/search-and-rescue",
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/svakhreev/projects/DEIM/data/rgbtdrone_person_coco",
        help="Output directory for COCO format files"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset info
    data_yaml_path = dataset_root / "data.yaml"
    if not data_yaml_path.exists():
        print(f"Error: {data_yaml_path} not found")
        return

    dataset_info = load_dataset_info(str(data_yaml_path))
    print(f"Dataset info: {dataset_info}")

    # Create category mapping
    category_mapping = {}
    for i, name in enumerate(dataset_info["names"]):
        category_mapping[i] = {
            "id": i + 1,  # COCO categories start from 1
            "name": name,
            "supercategory": "person" if name == "human" else name
        }

    print(f"Categories: {category_mapping}")

    # Process each split
    splits = ["train", "valid", "test"]
    img_id = 1
    ann_id = 1

    for split in splits:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"

        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping {split}")
            continue

        print(f"\nConverting {split} split...")

        coco_data, next_img_id, next_ann_id = convert_split_to_coco(
            str(images_dir),
            str(labels_dir),
            split,
            category_mapping,
            img_id,
            ann_id
        )

        # Update IDs for next split
        img_id = next_img_id
        ann_id = next_ann_id

        # Save COCO annotation file
        output_file = output_dir / f"{split}.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(
            f"Saved {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations to {output_file}")

    print(f"\nConversion complete! COCO format files saved in {output_dir}")

    # Create symlinks to original images for easy access
    for split in splits:
        original_images_dir = dataset_root / split / "images"
        symlink_dir = output_dir / f"{split}_images"

        if original_images_dir.exists() and not symlink_dir.exists():
            try:
                symlink_dir.symlink_to(original_images_dir.resolve())
                print(f"Created symlink: {symlink_dir} -> {original_images_dir}")
            except Exception as e:
                print(f"Warning: Could not create symlink for {split}: {e}")


if __name__ == "__main__":
    main()
