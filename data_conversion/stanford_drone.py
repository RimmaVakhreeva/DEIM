#!/usr/bin/env python3
"""
Convert Stanford Drone dataset to COCO format.

Dataset structure:
/mnt/archive/person_drone/stanford_drone/
├── annotations/
│   ├── scene1/
│   │   └── video0/
│   │       ├── annotations.txt
│   │       └── reference.jpg
└── video/
    ├── scene1/
    │   └── video0/
    │       └── video.mp4

Annotation format: frame_id xmin ymin xmax ymax track_id object_lost occluded generated class_name
- frame_id: Frame number (0-indexed)
- xmin, ymin, xmax, ymax: Bounding box coordinates in pixels
- track_id: Unique track ID for multi-object tracking
- object_lost: 1 if object is lost, 0 otherwise
- occluded: 1 if object is occluded, 0 otherwise  
- generated: 1 if bounding box is generated/interpolated, 0 otherwise
- class_name: Object class in quotes (e.g., "Pedestrian", "Biker", "Car", "Cart", "Skater", "Bus")
"""

import json
import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
from collections import defaultdict
import random


def get_all_scenes_and_videos(dataset_root: str) -> List[Tuple[str, str]]:
    """Get all scene/video combinations in the dataset."""
    annotations_dir = Path(dataset_root) / "annotations"
    scene_videos = []
    
    for scene_dir in annotations_dir.iterdir():
        if scene_dir.is_dir():
            for video_dir in scene_dir.iterdir():
                if video_dir.is_dir():
                    scene_videos.append((scene_dir.name, video_dir.name))
    
    return sorted(scene_videos)


def parse_stanford_annotations(annotation_file: str) -> Dict[int, List[Dict]]:
    """
    Parse Stanford Drone annotation file.
    
    Args:
        annotation_file: Path to annotations.txt file
    
    Returns:
        Dictionary mapping frame_id to list of annotations
    """
    frame_annotations = defaultdict(list)
    
    with open(annotation_file, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split()
                if len(parts) < 10:
                    continue
                
                frame_id = int(parts[0])
                xmin = float(parts[1])
                ymin = float(parts[2])
                xmax = float(parts[3])
                ymax = float(parts[4])
                track_id = int(parts[5])
                object_lost = int(parts[6])
                occluded = int(parts[7])
                generated = int(parts[8])
                class_name = parts[9].strip('"')
                
                # Calculate width and height
                width = xmax - xmin
                height = ymax - ymin
                
                # Skip invalid bboxes
                if width <= 0 or height <= 0:
                    continue
                
                frame_annotations[frame_id].append({
                    'bbox': [xmin, ymin, width, height],
                    'track_id': track_id,
                    'class_name': class_name,
                    'object_lost': object_lost,
                    'occluded': occluded,
                    'generated': generated
                })
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {line_num + 1} in {annotation_file}: {line}")
                continue
    
    return dict(frame_annotations)


def extract_frames_from_video(video_path: str, output_dir: str, sample_rate: int = 30) -> List[Tuple[int, str]]:
    """
    Extract frames from video at specified sample rate.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        sample_rate: Extract every Nth frame
    
    Returns:
        List of (frame_id, image_path) tuples for successfully extracted frames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    extracted_frames = []
    frame_id = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if not success:
            break
        
        # Sample frames at specified rate
        if frame_id % sample_rate == 0:
            frame_filename = f"frame_{frame_id:06d}.jpg"
            frame_path = output_dir / frame_filename
            
            if cv2.imwrite(str(frame_path), frame):
                extracted_frames.append((frame_id, str(frame_path)))
            else:
                print(f"Warning: Could not save frame {frame_id} to {frame_path}")
        
        frame_id += 1
    
    cap.release()
    print(f"Extracted {len(extracted_frames)} frames from {video_path}")
    return extracted_frames


def create_train_val_test_splits(scene_videos: List[Tuple[str, str]], 
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.2,
                                test_ratio: float = 0.1) -> Dict[str, List[Tuple[str, str]]]:
    """Create train/val/test splits from scene/video combinations."""
    random.shuffle(scene_videos)
    
    n_total = len(scene_videos)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    splits = {
        'train': scene_videos[:n_train],
        'val': scene_videos[n_train:n_train + n_val],
        'test': scene_videos[n_train + n_val:]
    }
    
    print(f"Dataset splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


def convert_split_to_coco(
    dataset_root: str,
    scene_videos: List[Tuple[str, str]],
    split_name: str,
    output_dir: str,
    category_mapping: Dict[str, Dict],
    sample_rate: int = 30,
    start_img_id: int = 1,
    start_ann_id: int = 1
) -> Tuple[Dict, int, int]:
    """
    Convert a dataset split to COCO format.
    
    Args:
        dataset_root: Root directory of Stanford Drone dataset
        scene_videos: List of (scene, video) tuples for this split
        split_name: Name of the split (train/val/test)
        output_dir: Output directory for images and annotations
        category_mapping: Mapping of class names to category info
        sample_rate: Extract every Nth frame from videos
        start_img_id: Starting image ID
        start_ann_id: Starting annotation ID
    
    Returns:
        COCO format dictionary, next_img_id, next_ann_id
    """
    coco_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": f"Stanford Drone Dataset - {split_name} split",
            "contributor": "Stanford Drone Dataset",
            "url": "http://cvgl.stanford.edu/projects/uav_data/",
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
    
    img_id = start_img_id
    ann_id = start_ann_id
    
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    
    # Create images directory for this split
    images_output_dir = output_dir / f"{split_name}_images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(scene_videos)} scene/video combinations for {split_name} split...")
    
    for scene, video in scene_videos:
        print(f"Processing {scene}/{video}...")
        
        # Paths
        video_path = dataset_root / "video" / scene / video / "video.mp4"
        annotation_file = dataset_root / "annotations" / scene / video / "annotations.txt"
        
        if not video_path.exists() or not annotation_file.exists():
            print(f"Warning: Missing video or annotation for {scene}/{video}")
            continue
        
        # Parse annotations
        frame_annotations = parse_stanford_annotations(str(annotation_file))
        
        # Extract frames
        scene_video_dir = images_output_dir / f"{scene}_{video}"
        extracted_frames = extract_frames_from_video(str(video_path), str(scene_video_dir), sample_rate)
        
        # Process extracted frames
        for frame_id, frame_path in extracted_frames:
            # Get frame dimensions
            img = cv2.imread(frame_path)
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Relative path for COCO annotation
            relative_frame_path = Path(frame_path).relative_to(images_output_dir)
            
            # Add image info
            coco_data["images"].append({
                "id": img_id,
                "file_name": str(relative_frame_path),
                "width": img_width,
                "height": img_height,
                "license": 1
            })
            
            # Add annotations for this frame
            if frame_id in frame_annotations:
                for annotation in frame_annotations[frame_id]:
                    class_name = annotation['class_name']
                    
                    if class_name not in category_mapping:
                        continue  # Skip unknown classes
                    
                    bbox = annotation['bbox']
                    xmin, ymin, width, height = bbox
                    
                    # Ensure bbox is within image bounds
                    xmin = max(0, min(xmin, img_width - 1))
                    ymin = max(0, min(ymin, img_height - 1))
                    width = min(width, img_width - xmin)
                    height = min(height, img_height - ymin)
                    
                    if width <= 0 or height <= 0:
                        continue
                    
                    area = width * height
                    
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": category_mapping[class_name]["id"],
                        "bbox": [xmin, ymin, width, height],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                        # Additional Stanford Drone specific fields
                        "track_id": annotation['track_id'],
                        "object_lost": annotation['object_lost'],
                        "occluded": annotation['occluded'],
                        "generated": annotation['generated']
                    })
                    ann_id += 1
            
            img_id += 1
    
    return coco_data, img_id, ann_id


def main():
    parser = argparse.ArgumentParser(description="Convert Stanford Drone dataset to COCO format")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/archive/person_drone/stanford_drone",
        help="Path to Stanford Drone dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/svakhreev/projects/DEIM/data/stanford_drone_coco",
        help="Output directory for COCO format files"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=30,
        help="Extract every Nth frame from videos (default: 30)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="Proportion of data for training (default: 0.7)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="Proportion of data for validation (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducible splits
    random.seed(args.seed)
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all scene/video combinations
    scene_videos = get_all_scenes_and_videos(str(dataset_root))
    print(f"Found {len(scene_videos)} scene/video combinations")
    
    # Create category mapping for Stanford Drone classes
    class_names = ["Pedestrian", "Biker", "Car", "Cart", "Skater", "Bus"]
    category_mapping = {}
    for i, name in enumerate(class_names):
        category_mapping[name] = {
            "id": i + 1,  # COCO categories start from 1
            "name": name,
            "supercategory": "object"
        }
    
    print(f"Categories: {category_mapping}")
    
    # Create train/val/test splits
    splits = create_train_val_test_splits(
        scene_videos, 
        args.train_ratio, 
        args.val_ratio, 
        1.0 - args.train_ratio - args.val_ratio
    )
    
    # Process each split
    img_id = 1
    ann_id = 1
    
    for split_name, split_scene_videos in splits.items():
        if not split_scene_videos:
            continue
        
        print(f"\nConverting {split_name} split...")
        
        coco_data, next_img_id, next_ann_id = convert_split_to_coco(
            str(dataset_root),
            split_scene_videos,
            split_name,
            str(output_dir),
            category_mapping,
            args.sample_rate,
            img_id,
            ann_id
        )
        
        # Update IDs for next split
        img_id = next_img_id
        ann_id = next_ann_id
        
        # Save COCO annotation file
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Saved {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations to {output_file}")
    
    print(f"\nConversion complete! COCO format files saved in {output_dir}")
    
    # Save split information for reproducibility
    splits_file = output_dir / "splits_info.json"
    with open(splits_file, 'w') as f:
        json.dump({
            'splits': splits,
            'args': vars(args),
            'categories': category_mapping
        }, f, indent=2)
    
    print(f"Split information saved to {splits_file}")


if __name__ == "__main__":
    main()
