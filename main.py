import argparse
import os
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from video_processor import VideoProcessor
from object_detector import ObjectDetector
from object_tracker import ObjectTracker
from crop_calculator import CropCalculator
from smoothing import CropWindowSmoother

def parse_args():
    parser = argparse.ArgumentParser(description='Content-aware video cropping')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--target_ratio', type=float, default=9/16, help='Target aspect ratio (width/height)')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'ssd', 'faster_rcnn'], 
                        help='Object detection model to use')
    parser.add_argument('--smoothing_window', type=int, default=30, 
                        help='Number of frames for temporal smoothing')
    parser.add_argument('--skip_frames', type=int, default=10, 
                        help='Process every nth frame for detection (1 = process all frames)')
    parser.add_argument('--model_size', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--use_saliency', action='store_true',
                        help='Use saliency detection for regions of interest')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads for parallel processing')
    return parser.parse_args()

def process_keyframe(frame_idx, frame, detector, tracker, tracked_objects_by_frame):
    """Process a keyframe with detection and tracking."""
    # Detect objects in frame
    detected_objects = detector.detect(frame)
    
    # Update tracker with new detections
    tracked_objects = tracker.update(frame, detected_objects)
    tracked_objects_by_frame[frame_idx] = tracked_objects
    
    return frame_idx

def main():
    args = parse_args()
    
    # Initialize components with YOLOv8
    video_processor = VideoProcessor()
    detector = ObjectDetector(
        confidence_threshold=args.conf_threshold,
        model_size=args.model_size  # Pass the model size argument here
    )
    tracker = ObjectTracker()
    crop_calculator = CropCalculator(target_ratio=args.target_ratio)
    smoother = CropWindowSmoother(window_size=args.smoothing_window)
    
    # Load video and get properties
    video_info = video_processor.load_video(args.input)
    total_frames = video_info['total_frames']
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']
    
    print(f"Processing video: {args.input}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    
    # Process frames
    tracked_objects_by_frame = {}
    
    start_time = time.time()
    
    # First pass: detect and track objects on keyframes only
    print("Phase 1: Detecting and tracking objects...")
    
    # Determine keyframes
    keyframes = list(range(0, total_frames, args.skip_frames))
    
    # Use ThreadPoolExecutor for parallel processing of keyframes
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a list to store futures
        futures = []
        
        # Process keyframes
        for frame_idx in keyframes:
            # Set position to keyframe
            video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_processor.cap.read()
            
            if not ret:
                continue
            
            # Submit task to executor
            future = executor.submit(
                process_keyframe, 
                frame_idx, 
                frame, 
                detector, 
                tracker, 
                tracked_objects_by_frame
            )
            futures.append(future)
            
            # Print progress every 10 keyframes
            if len(futures) % 10 == 0:
                print(f"Submitted {len(futures)}/{len(keyframes)} keyframes for processing")
        
        # Wait for all futures to complete
        for i, future in enumerate(futures):
            future.result()  # This will raise any exceptions that occurred
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(keyframes)} keyframes")
    
    # Second pass: calculate crop windows for keyframes
    print("Phase 2: Calculating crop windows for keyframes...")
    
    # Pre-allocate crop windows array
    crop_windows = [None] * total_frames
    
    # Process keyframes
    for frame_idx in keyframes:
        if frame_idx not in tracked_objects_by_frame:
            continue
            
        objects = tracked_objects_by_frame[frame_idx]
        
        # Get the actual frame for additional analysis
        video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_processor.cap.read()
        
        if not ret:
            continue
        
        # Calculate optimal crop window
        crop_window = crop_calculator.calculate(objects, width, height, frame)
        crop_windows[frame_idx] = crop_window
        
        if frame_idx % 100 == 0:
            print(f"Calculated crop window for keyframe {frame_idx}/{total_frames}")
    
    # Phase 3: Interpolate crop windows for non-keyframes
    print("Phase 3: Interpolating crop windows for non-keyframes...")
    
    # Fast interpolation using numpy
    keyframe_indices = np.array(keyframes)
    keyframe_crop_windows = np.array([crop_windows[i] for i in keyframes if crop_windows[i] is not None])
    
    if len(keyframe_crop_windows) > 1:
        # For each frame, find the nearest keyframes and interpolate
        for i in range(total_frames):
            if crop_windows[i] is not None:
                continue
                
            # Find nearest keyframes
            next_idx = keyframe_indices[keyframe_indices > i]
            prev_idx = keyframe_indices[keyframe_indices < i]
            
            if len(next_idx) == 0 and len(prev_idx) > 0:
                # After last keyframe, use last keyframe
                crop_windows[i] = crop_windows[prev_idx[-1]]
            elif len(prev_idx) == 0 and len(next_idx) > 0:
                # Before first keyframe, use first keyframe
                crop_windows[i] = crop_windows[next_idx[0]]
            elif len(prev_idx) > 0 and len(next_idx) > 0:
                # Interpolate between keyframes
                prev_frame = prev_idx[-1]
                next_frame = next_idx[0]
                
                if crop_windows[prev_frame] is not None and crop_windows[next_frame] is not None:
                    # Calculate interpolation factor
                    alpha = (i - prev_frame) / (next_frame - prev_frame)
                    
                    # Linear interpolation
                    prev_crop = np.array(crop_windows[prev_frame])
                    next_crop = np.array(crop_windows[next_frame])
                    interp_crop = prev_crop * (1 - alpha) + next_crop * alpha
                    crop_windows[i] = [int(x) for x in interp_crop]
    
    # Fill any remaining None values with center crop
    for i in range(total_frames):
        if crop_windows[i] is None:
            # Use center crop as fallback
            crop_height = height
            crop_width = int(crop_height * args.target_ratio)
            if crop_width > width:
                crop_width = width
                crop_height = int(crop_width / args.target_ratio)
            x = int((width - crop_width) / 2)
            y = int((height - crop_height) / 2)
            crop_windows[i] = [x, y, crop_width, crop_height]
    
    # Apply temporal smoothing to crop windows
    print("Phase 4: Applying temporal smoothing...")
    smoothed_windows = smoother.smooth(crop_windows)
    
    # Generate output video with cropped frames
    print("Phase 5: Generating output video...")
    video_processor.generate_output_video(
        output_path=args.output,
        crop_windows=smoothed_windows,
        fps=fps
    )
    
    elapsed_time = time.time() - start_time
    print(f"Video processing completed in {elapsed_time:.2f} seconds")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main() 