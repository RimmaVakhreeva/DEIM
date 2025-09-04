#!/usr/bin/env python3
"""
DEIM Debug Script for Interactive Bbox Detection and Visualization
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

This script provides interactive debugging capabilities for DEIM models:
- Load model from config and checkpoint
- Process images and videos
- Interactive OpenCV visualization with imshow
- Adjustable confidence thresholds
- Keyboard controls for video playback
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import os
import sys
import argparse
import time
from pathlib import Path
from PIL import Image

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from engine.core import YAMLConfig

# Default class names - will be overridden by dataset configuration
DEFAULT_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'aeroplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'trafficlight',
    11: 'firehydrant', 13: 'stopsign', 14: 'parkingmeter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie',
    33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sportsball', 38: 'kite', 39: 'baseballbat', 40: 'baseballglove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennisracket', 44: 'bottle',
    46: 'wineglass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hotdog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'sofa', 64: 'pottedplant', 65: 'bed',
    67: 'diningtable', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cellphone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddybear',
    89: 'hairdrier', 90: 'toothbrush'
}

def load_class_names_from_config(cfg):
    """Load class names from dataset configuration"""
    try:
        # Import here to avoid circular imports
        from engine.data.dataset.coco_dataset import mscoco_category2name
        
        # Check if we can access dataset configuration
        if hasattr(cfg, 'val_dataloader') and cfg.val_dataloader is not None:
            dataset_cfg = cfg.val_dataloader.dataset
            
            # Try to instantiate dataset to get class names
            try:
                # Get the number of classes from config
                num_classes = getattr(cfg, 'num_classes', 80)
                
                # Check if using COCO remapping
                remap_mscoco = getattr(cfg, 'remap_mscoco_category', False)
                
                if remap_mscoco:
                    print(f"Using COCO class names (remapped)")
                    return mscoco_category2name
                    
                # Try to create dataset instance to get category names
                if hasattr(dataset_cfg, 'ann_file') and dataset_cfg.ann_file:
                    # For COCO-style datasets, try to load annotations
                    try:
                        from pycocotools.coco import COCO
                        if os.path.exists(dataset_cfg.ann_file):
                            coco = COCO(dataset_cfg.ann_file)
                            categories = coco.dataset.get('categories', [])
                            if categories:
                                class_names = {}
                                for i, cat in enumerate(categories):
                                    # Use category ID as key for proper mapping
                                    class_names[cat['id']] = cat['name']
                                print(f"Loaded {len(class_names)} class names from annotation file")
                                return class_names
                    except Exception as e:
                        print(f"Could not load classes from annotation file: {e}")
                
                # Generate generic class names based on number of classes
                print(f"Generating generic class names for {num_classes} classes")
                if num_classes == 80:
                    return mscoco_category2name
                elif num_classes == 1:
                    return {1: 'object'}
                elif num_classes == 2:
                    return {1: 'person', 2: 'object'}  # Common for crowd detection
                elif num_classes == 20:
                    # VOC classes
                    voc_classes = {
                        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
                        6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
                        11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
                        16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
                    }
                    return voc_classes
                else:
                    # Generic class names
                    return {i+1: f'class_{i+1}' for i in range(num_classes)}
                    
            except Exception as e:
                print(f"Could not instantiate dataset: {e}")
                
    except Exception as e:
        print(f"Could not load class names from config: {e}")
    
    # Fallback to default COCO classes
    print("Using default COCO class names")
    return DEFAULT_CLASSES

# Color palette for bounding boxes (BGR format for OpenCV)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (192, 192, 192), (64, 64, 64)
]

class DEIMModel(nn.Module):
    """Wrapper for DEIM model with postprocessing"""
    
    def __init__(self, config_path, checkpoint_path, device='cuda', input_size=640):
        super().__init__()
        self.device = device
        
        # Create initial config to check current eval_spatial_size
        temp_cfg = YAMLConfig(config_path, resume=None)
        current_eval_size = temp_cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
        
        # Prepare config overrides based on input_size
        config_overrides = {}
        if current_eval_size != [input_size, input_size]:
            print(f"Adjusting eval_spatial_size from {current_eval_size} to [{input_size}, {input_size}]")
            if input_size != 640:
                print(f"Warning: Using non-standard input size ({input_size}). This will force dynamic positional embedding computation, which may be slower.")
            
            config_overrides['eval_spatial_size'] = [input_size, input_size]
            
            # Force dynamic positional embeddings by setting HybridEncoder eval_spatial_size to None
            config_overrides['HybridEncoder'] = {'eval_spatial_size': None}
        
        # Disable pretrained weights loading
        config_overrides['HGNetv2'] = {'pretrained': False}
        
        # Create the final config with overrides
        self.cfg = YAMLConfig(config_path, resume=checkpoint_path, **config_overrides)
        
        # Load checkpoint
        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle EMA or regular model state
            if 'ema' in checkpoint:
                print("Loading EMA model weights")
                state_dict = checkpoint['ema']['module']
            else:
                print("Loading regular model weights")
                state_dict = checkpoint['model']
            
            # If we changed the spatial size, skip loading parameters that depend on spatial dimensions
            if config_overrides and 'eval_spatial_size' in config_overrides:
                print("Filtering out spatial-size-dependent parameters (anchors, valid_mask)")
                # These parameters depend on spatial size and will be regenerated
                spatial_dependent_keys = ['decoder.anchors', 'decoder.valid_mask']
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                     if not any(key in k for key in spatial_dependent_keys)}
                
                print(f"Skipped {len(state_dict) - len(filtered_state_dict)} spatial-dependent parameters")
                self.cfg.model.load_state_dict(filtered_state_dict, strict=False)
            else:
                self.cfg.model.load_state_dict(state_dict)
        else:
            print("Warning: No checkpoint provided, using random weights")
        
        # Set up model and postprocessor
        self.model = self.cfg.model.eval().to(device)
        self.postprocessor = self.cfg.postprocessor.eval().to(device)
        
        # Load class names from configuration
        self.class_names = load_class_names_from_config(self.cfg)
        self.num_classes = getattr(self.cfg, 'num_classes', len(self.class_names))
        
        print(f"Model loaded successfully on {device}")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Sample classes: {dict(list(self.class_names.items())[:5])}...")
    
    def forward(self, images, orig_sizes):
        """Forward pass through model and postprocessor"""
        with torch.no_grad():
            # Model inference
            outputs = self.model(images)
            
            # Postprocessing
            results = self.postprocessor(outputs, orig_sizes)
            
        return results
    
    def get_class_name(self, class_id):
        """Get class name for given class ID"""
        return self.class_names.get(class_id, f'class_{class_id}')

class DebugVisualizer:
    """Interactive visualizer with OpenCV"""
    
    def __init__(self, model, confidence_threshold=0.5, window_name="DEIM Debug"):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.window_name = window_name
        self.paused = False
        self.show_info = True
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        
        print("\n=== Debug Controls ===")
        print("SPACE: Pause/Resume video")
        print("'q' or ESC: Quit")
        print("'i': Toggle info display")
        print("'+'/'-': Increase/Decrease confidence threshold")
        print("'s': Save current frame")
        print("'n': Next file (in folder mode)")
        print("=====================\n")
    
    def draw_detections(self, image, results, frame_info=None):
        """Draw bounding boxes and labels on image"""
        vis_image = image.copy()
        
        if len(results) == 0:
            return vis_image
            
        # Extract results
        result = results[0] if isinstance(results, list) else results
        labels = result['labels'].cpu().numpy()
        boxes = result['boxes'].cpu().numpy()  
        scores = result['scores'].cpu().numpy()
        
        # Filter by confidence threshold
        valid_indices = scores >= self.confidence_threshold
        labels = labels[valid_indices]
        boxes = boxes[valid_indices] 
        scores = scores[valid_indices]
        
        # Draw bounding boxes
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get class name and color
            class_name = self.model.get_class_name(label)
            color = COLORS[label % len(COLORS)]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_text = f'{class_name}: {score:.2f}'
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(vis_image, (x1, y1 - text_h - baseline), 
                         (x1 + text_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label_text, (x1, y1 - baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw info overlay
        if self.show_info:
            self._draw_info_overlay(vis_image, labels, scores, frame_info)
        
        return vis_image
    
    def _draw_info_overlay(self, image, labels, scores, frame_info=None):
        """Draw information overlay on image"""
        h, w = image.shape[:2]
        overlay_y = 30
        
        # Detection count and confidence info
        info_lines = [
            f"Detections: {len(labels)} (conf >= {self.confidence_threshold:.2f})",
            f"Avg Confidence: {scores.mean():.3f}" if len(scores) > 0 else "Avg Confidence: N/A"
        ]
        
        # Add frame info for videos
        if frame_info:
            info_lines.extend([
                f"Frame: {frame_info.get('frame_num', 'N/A')}",
                f"FPS: {frame_info.get('fps', 'N/A'):.1f}",
                f"Status: {'PAUSED' if self.paused else 'PLAYING'}"
            ])
            # Add file progress if available
            if 'file_progress' in frame_info:
                info_lines.append(f"File: {frame_info['file_progress']}")
        
        # Draw background
        overlay_height = len(info_lines) * 25 + 20
        cv2.rectangle(image, (10, 10), (350, 10 + overlay_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (350, 10 + overlay_height), 
                     (255, 255, 255), 1)
        
        # Draw text
        for i, line in enumerate(info_lines):
            cv2.putText(image, line, (20, overlay_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def show_image(self, image, results, title=None):
        """Display single image with detections"""
        vis_image = self.draw_detections(image, results)
        
        if title:
            cv2.setWindowTitle(self.window_name, f"{self.window_name} - {title}")
        
        cv2.imshow(self.window_name, vis_image)
        
        # Wait for key press
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                return False
            elif key == ord('n'):  # Next file
                return True
            elif key == ord('s'):  # Save image
                save_path = f"debug_output_{int(time.time())}.jpg"
                cv2.imwrite(save_path, vis_image)
                print(f"Image saved as {save_path}")
            elif key == ord('i'):  # Toggle info
                self.show_info = not self.show_info
                vis_image = self.draw_detections(image, results)
                cv2.imshow(self.window_name, vis_image)
            elif key == ord('+') or key == ord('='):  # Increase threshold
                self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                vis_image = self.draw_detections(image, results)
                cv2.imshow(self.window_name, vis_image)
            elif key == ord('-') or key == ord('_'):  # Decrease threshold  
                self.confidence_threshold = max(0.0, self.confidence_threshold - 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                vis_image = self.draw_detections(image, results)
                cv2.imshow(self.window_name, vis_image)
            else:
                break
                
        return True
    
    def show_video_frame(self, image, results, frame_info):
        """Display video frame with detections"""
        vis_image = self.draw_detections(image, results, frame_info)
        
        cv2.setWindowTitle(self.window_name, 
                          f"{self.window_name} - Frame {frame_info.get('frame_num', 'N/A')}")
        cv2.imshow(self.window_name, vis_image)
        
        # Handle keyboard input
        wait_time = 1 if self.paused else max(1, int(1000 / frame_info.get('fps', 30)))
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q') or key == 27:  # Quit
            return False
        elif key == ord('n'):  # Next file (skip rest of video)
            return 'next'
        elif key == ord(' '):  # Pause/Resume
            self.paused = not self.paused
            print("PAUSED" if self.paused else "RESUMED")
        elif key == ord('s'):  # Save frame
            save_path = f"debug_frame_{frame_info.get('frame_num', int(time.time()))}.jpg"
            cv2.imwrite(save_path, vis_image)
            print(f"Frame saved as {save_path}")
        elif key == ord('i'):  # Toggle info
            self.show_info = not self.show_info
        elif key == ord('+') or key == ord('='):  # Increase threshold
            self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
            print(f"Confidence threshold: {self.confidence_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):  # Decrease threshold
            self.confidence_threshold = max(0.0, self.confidence_threshold - 0.05)
            print(f"Confidence threshold: {self.confidence_threshold:.2f}")
        
        return True
    
    def close(self):
        """Close visualization windows"""
        cv2.destroyAllWindows()

def find_media_files(folder_path):
    """Recursively find all image and video files in folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
    
    media_files = []
    folder_path = Path(folder_path)
    
    if folder_path.is_file():
        # Single file provided
        if folder_path.suffix.lower() in image_extensions | video_extensions:
            media_files.append(folder_path)
    else:
        # Recursively find all media files
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions | video_extensions:
                media_files.append(file_path)
    
    # Sort files for consistent ordering
    media_files.sort()
    
    # Separate images and videos
    images = [f for f in media_files if f.suffix.lower() in image_extensions]
    videos = [f for f in media_files if f.suffix.lower() in video_extensions]
    
    print(f"Found {len(images)} images and {len(videos)} videos")
    return images, videos

def process_image(model, image_path, visualizer, input_size=640, file_index=None, total_files=None):
    """Process single image"""
    progress_str = f"[{file_index+1}/{total_files}] " if file_index is not None else ""
    print(f"{progress_str}Processing image: {image_path}")
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    h, w = image.shape[:2]
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(model.device)
    
    # Convert to PIL for transforms
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Apply transforms
    transforms = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])
    
    tensor_image = transforms(pil_image).unsqueeze(0).to(model.device)
    
    # Run inference
    start_time = time.time()
    results = model(tensor_image, orig_size)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.3f}s")
    
    # Show results
    title = f"{progress_str}{Path(image_path).name} ({inference_time:.3f}s)"
    return visualizer.show_image(image, results, title)

def process_video(model, video_path, visualizer, input_size=640, file_index=None, total_files=None):
    """Process video file"""
    progress_str = f"[{file_index+1}/{total_files}] " if file_index is not None else ""
    print(f"{progress_str}Processing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    
    # Apply transforms
    transforms = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])
    
    frame_num = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            for _ in range(10):
                ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            h, w = frame.shape[:2]
            orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(model.device)
            
            # Convert to PIL for transforms
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor_frame = transforms(pil_frame).unsqueeze(0).to(model.device)
            
            # Run inference
            frame_start = time.time()
            results = model(tensor_frame, orig_size)
            inference_time = time.time() - frame_start
            
            # Calculate average FPS
            elapsed_time = time.time() - start_time
            avg_fps = frame_num / elapsed_time if elapsed_time > 0 else 0
            
            # Prepare frame info
            frame_info = {
                'frame_num': frame_num,
                'fps': avg_fps,
                'inference_time': inference_time,
                'total_frames': total_frames,
                'file_progress': f"{progress_str}{Path(video_path).name}"
            }
            
            # Show frame
            result = visualizer.show_video_frame(frame, results, frame_info)
            if result == False:
                break
            elif result == 'next':
                print("Skipping to next file...")
                break
            
            # Print progress periodically
            if frame_num % 30 == 0:
                print(f"Processed {frame_num}/{total_frames} frames, "
                      f"Avg FPS: {avg_fps:.1f}, "
                      f"Inference: {inference_time:.3f}s")
    
    finally:
        cap.release()
        
    print(f"\nVideo processing completed!")
    print(f"Total frames processed: {frame_num}")
    print(f"Average FPS: {frame_num / (time.time() - start_time):.2f}")
    
    return True

def process_folder(model, folder_path, visualizer, input_size=640, process_videos=True):
    """Process all images and videos in a folder recursively"""
    print(f"Scanning folder: {folder_path}")
    
    # Find all media files
    images, videos = find_media_files(folder_path)
    
    if not images and not videos:
        print("No image or video files found!")
        return False
    
    all_files = []
    
    # Add images first
    if images:
        print(f"\nFound {len(images)} images:")
        for img in images[:10]:  # Show first 10
            print(f"  {img}")
        if len(images) > 10:
            print(f"  ... and {len(images) - 10} more")
        all_files.extend([(img, 'image') for img in images])
    
    # Add videos if requested
    if videos and process_videos:
        print(f"\nFound {len(videos)} videos:")
        for vid in videos[:10]:  # Show first 10
            print(f"  {vid}")
        if len(videos) > 10:
            print(f"  ... and {len(videos) - 10} more")
        all_files.extend([(vid, 'video') for vid in videos])
    elif videos and not process_videos:
        print(f"\nSkipping {len(videos)} videos (use --process-videos to include them)")
    
    if not all_files:
        print("No files to process!")
        return False
    
    print(f"\nProcessing {len(all_files)} files total...")
    print("Use SPACE to pause/resume, 'q' to quit, 'n' for next file")
    
    # Process all files
    for i, (file_path, file_type) in enumerate(all_files):
        print(f"\n{'='*60}")
        
        try:
            if file_type == 'image':
                success = process_image(model, file_path, visualizer, input_size, i, len(all_files))
            else:  # video
                success = process_video(model, file_path, visualizer, input_size, i, len(all_files))
            
            if not success:
                print(f"Stopping processing at user request or error")
                break
                
        except KeyboardInterrupt:
            print(f"\nProcessing interrupted by user")
            break
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            
            # Ask user if they want to continue
            response = input("Continue with next file? (y/n): ")
            if response.lower() != 'y':
                break
    
    print(f"\nFinished processing folder: {folder_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="DEIM Debug Script")
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Path to input image, video, or folder')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--input-size', type=int, default=1024,
                       help='Input image size')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='Confidence threshold for detections')
    parser.add_argument('--process-videos', action='store_true',
                       help='Process video files when scanning folders')
    parser.add_argument('--images-only', action='store_true',
                       help='Process only images (skip videos)')
    parser.add_argument('--videos-only', action='store_true',
                       help='Process only videos (skip images)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
        
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
        
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=== DEIM Debug Script ===")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input}")
    print(f"Device: {args.device}")
    print(f"Input size: {args.input_size}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print("========================\n")
    
    try:
        # Initialize model
        print("Loading model...")
        model = DEIMModel(args.config, args.checkpoint, args.device, args.input_size)
        
        # Initialize visualizer
        visualizer = DebugVisualizer(model, args.conf_threshold)
        
        # Determine input type and process
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single file
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
                if not args.videos_only:
                    success = process_image(model, args.input, visualizer, args.input_size)
                else:
                    print("Skipping image file (videos-only mode)")
                    success = True
            elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']:
                if not args.images_only:
                    success = process_video(model, args.input, visualizer, args.input_size)
                else:
                    print("Skipping video file (images-only mode)")
                    success = True
            else:
                print(f"Error: Unsupported file format: {input_path.suffix}")
                success = False
        elif input_path.is_dir():
            # Folder - process recursively
            process_videos = args.process_videos or not args.images_only
            if args.videos_only:
                # Only process videos, skip images
                success = process_folder(model, args.input, visualizer, args.input_size, process_videos=True)
            elif args.images_only:
                # Only process images, skip videos
                success = process_folder(model, args.input, visualizer, args.input_size, process_videos=False)
            else:
                # Process based on --process-videos flag
                success = process_folder(model, args.input, visualizer, args.input_size, process_videos)
        else:
            print(f"Error: Input path does not exist: {args.input}")
            success = False
        
        if success:
            print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'visualizer' in locals():
            visualizer.close()
        print("Debug session ended.")

if __name__ == '__main__':
    main()
