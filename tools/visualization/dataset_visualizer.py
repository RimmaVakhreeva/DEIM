"""
Dataset Visualizer for DEIM Project
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Interactive dataset visualization tool using OpenCV for browsing datasets
with bounding box annotations.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from engine.core import YAMLConfig
from engine.data.dataset import CocoDetection, VOCDetection
from engine.data.dataset import mscoco_category2name, mscoco_category2label


class DatasetVisualizer:
    """Interactive dataset visualizer with OpenCV display."""
    
    def __init__(self, config_path: str, split: str = 'train'):
        """
        Initialize the dataset visualizer.
        
        Args:
            config_path: Path to dataset configuration YAML file
            split: Dataset split to visualize ('train' or 'val')
        """
        self.config_path = config_path
        self.split = split
        self.current_index = 0
        self.window_name = "DEIM Dataset Visualizer"
        
        # Color palette for bounding boxes (BGR format for OpenCV)
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Light Green
            (255, 192, 203), # Pink
            (220, 220, 220), # Light Gray
        ]
        
        # Load configuration and initialize dataset
        self.config = self._load_config()
        self.dataset = self._create_dataset()
        self.category_names = self._get_category_names()
        
        print(f"Loaded dataset: {len(self.dataset)} samples")
        print(f"Categories: {len(self.category_names)}")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load dataset configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
        
    def _create_dataset(self):
        """Create dataset instance based on configuration."""
        dataloader_key = f"{self.split}_dataloader"
        if dataloader_key not in self.config:
            raise ValueError(f"No {dataloader_key} found in config")
            
        dataset_config = self.config[dataloader_key]['dataset']
        dataset_type = dataset_config['type']
        
        if dataset_type == 'CocoDetection':
            return CocoDetection(
                img_folder=dataset_config['img_folder'],
                ann_file=dataset_config['ann_file'],
                transforms=None,  # No transforms for visualization
                return_masks=dataset_config.get('return_masks', False),
                remap_mscoco_category=self.config.get('remap_mscoco_category', False)
            )
        elif dataset_type == 'VOCDetection':
            return VOCDetection(
                root=dataset_config['root'],
                ann_file=dataset_config['ann_file'],
                label_file=dataset_config['label_file'],
                transforms=None  # No transforms for visualization
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def _get_category_names(self) -> Dict[int, str]:
        """Get category ID to name mapping."""
        if hasattr(self.dataset, 'category2name'):
            return self.dataset.category2name
        elif hasattr(self.dataset, 'labels_map'):
            # VOC dataset - reverse the labels_map
            return {v: k for k, v in self.dataset.labels_map.items()}
        elif self.config.get('remap_mscoco_category', False):
            return mscoco_category2name
        else:
            # Try to get from COCO dataset categories
            if hasattr(self.dataset, 'categories'):
                return {cat['id']: cat['name'] for cat in self.dataset.categories}
            else:
                # Fallback to generic names
                num_classes = self.config.get('num_classes', 80)
                return {i: f'class_{i}' for i in range(num_classes)}
    
    def _pil_to_opencv(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to OpenCV format."""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        opencv_image = np.array(pil_image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        return opencv_image
    
    def _draw_bounding_boxes(self, image: np.ndarray, target: Dict[str, Any]) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        if 'boxes' not in target or len(target['boxes']) == 0:
            return image
            
        boxes = target['boxes']
        labels = target['labels'] if 'labels' in target else None
        
        # Handle different box formats and tensor types
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(labels) and labels is not None:
            labels = labels.cpu().numpy()
            
        image_copy = image.copy()
        
        for i, box in enumerate(boxes):
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Get color for this category
            color_idx = (labels[i] if labels is not None else i) % len(self.colors)
            color = self.colors[color_idx]
            
            # Draw bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if labels is not None:
                label_id = int(labels[i])
                label_text = self.category_names.get(label_id, f'class_{label_id}')
                
                # Calculate text size and background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, thickness
                )
                
                # Draw text background
                cv2.rectangle(
                    image_copy,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width + 5, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    image_copy,
                    label_text,
                    (x1 + 2, y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
        
        return image_copy
    
    def _add_info_overlay(self, image: np.ndarray, target: Dict[str, Any]) -> np.ndarray:
        """Add information overlay to the image."""
        info_text = [
            f"Sample: {self.current_index + 1}/{len(self.dataset)}",
            f"Image ID: {target.get('image_id', 'N/A')}",
            f"Objects: {len(target.get('boxes', []))}",
            "",
            "Controls:",
            "'n' or '→' - Next image",
            "'p' or '←' - Previous image", 
            "'s' - Save current image",
            "'q' or ESC - Quit",
            "'h' - Show/hide help",
        ]
        
        # Add info panel
        panel_width = 300
        panel_height = len(info_text) * 25 + 20
        
        # Create info panel
        info_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        info_panel.fill(50)  # Dark gray background
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        for i, text in enumerate(info_text):
            y_pos = 20 + i * 25
            if text == "":  # Empty line for spacing
                continue
                
            color = (255, 255, 255) if not text.startswith("'") else (0, 255, 255)
            cv2.putText(info_panel, text, (10, y_pos), font, font_scale, color, thickness)
        
        # Combine with original image
        h, w = image.shape[:2]
        if w > panel_width:
            # Overlay on the right side
            combined = image.copy()
            combined[10:10+panel_height, w-panel_width-10:w-10] = info_panel
        else:
            # Concatenate horizontally if image is too small
            combined = np.hstack([image, info_panel])
            
        return combined
    
    def visualize_sample(self, index: int, show_info: bool = True) -> Tuple[int, np.ndarray]:
        """Visualize a single sample from the dataset."""
        if index < 0 or index >= len(self.dataset):
            return index, None
            
        # Load image and target
        while True:
            image, target = self.dataset.load_item(index)
            if len(target['boxes']) < 300:
                print(f"skipping {index}")
                index += 1
            else:
                break

        # Convert PIL to OpenCV
        cv_image = self._pil_to_opencv(image)

        # Draw bounding boxes
        cv_image = self._draw_bounding_boxes(cv_image, target)
        
        # Add info overlay
        if show_info:
            cv_image = self._add_info_overlay(cv_image, target)
            
        return index, cv_image
    
    def save_current_image(self) -> None:
        """Save the current image with annotations."""
        output_dir = "dataset_visualization_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        self.current_index, image = self.visualize_sample(self.current_index, show_info=False)
        if image is not None:
            filename = f"sample_{self.current_index:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, image)
            print(f"Saved image to: {filepath}")
    
    def run(self):
        """Run the interactive visualization."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        
        show_info = True
        
        print("\n=== DEIM Dataset Visualizer ===")
        print(f"Dataset: {self.config_path}")
        print(f"Split: {self.split}")
        print(f"Total samples: {len(self.dataset)}")
        print("\nControls:")
        print("  'n' or Right Arrow - Next image")
        print("  'p' or Left Arrow  - Previous image")
        print("  's' - Save current image")
        print("  'h' - Toggle help overlay")
        print("  'q' or ESC - Quit")
        print("\nPress any key to start...")
        
        while True:
            # Display current image
            self.current_index, image = self.visualize_sample(self.current_index, show_info)
            if image is None:
                print("Failed to load image")
                break
                
            cv2.imshow(self.window_name, image)
            
            # Handle keyboard input
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('n') or key == 83:  # 'n' or Right arrow
                self.current_index = (self.current_index + 1) % len(self.dataset)
            elif key == ord('p') or key == 81:  # 'p' or Left arrow
                self.current_index = (self.current_index - 1) % len(self.dataset)
            elif key == ord('s'):  # 's' - Save
                self.save_current_image()
            elif key == ord('h'):  # 'h' - Toggle help
                show_info = not show_info
            elif key == ord('g'):  # 'g' - Go to specific index
                try:
                    new_index = int(input(f"Enter index (0-{len(self.dataset)-1}): "))
                    if 0 <= new_index < len(self.dataset):
                        self.current_index = new_index
                    else:
                        print("Index out of range")
                except ValueError:
                    print("Invalid index")
        
        cv2.destroyAllWindows()
        print("Visualization ended.")


def main():
    """Main function for the dataset visualizer."""
    parser = argparse.ArgumentParser(description='DEIM Dataset Visualizer')
    parser.add_argument('config', help='Path to dataset configuration YAML file')
    parser.add_argument('--split', choices=['train', 'val'], default='train',
                        help='Dataset split to visualize (default: train)')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting index for visualization (default: 0)')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        return
    
    try:
        # Create and run visualizer
        visualizer = DatasetVisualizer(args.config, args.split)
        visualizer.current_index = args.start_index
        visualizer.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
