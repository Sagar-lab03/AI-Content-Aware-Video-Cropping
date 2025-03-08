import cv2
import numpy as np
import os

class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.writer = None
        self.frames = None
        self.current_frame_idx = 0
        self.video_info = {}
    
    def load_video(self, video_path):
        """Load video file and extract basic information."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(video_path)
        # self.cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
        self.video_info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'path': video_path
        }
        
        self.current_frame_idx = 0
        return self.video_info
    
    def frame_generator(self):
        """Generator that yields frames from the video."""
        if self.cap is None:
            raise ValueError("No video loaded. Call load_video() first.")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            yield frame
            self.current_frame_idx += 1
    
    def apply_crop(self, frame, crop_window):
        """Apply crop window to a frame."""
        x, y, w, h = crop_window
        
        # Ensure crop window is within frame boundaries
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))
        
        # Apply crop
        return frame[y:y+h, x:x+w]
    
    def generate_output_video(self, output_path, crop_windows, fps=None):
        """Generate output video with the specified crop windows."""
        if self.cap is None:
            raise ValueError("No video loaded. Call load_video() first.")
        
        if fps is None:
            fps = self.video_info['fps']
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get first crop window to determine output dimensions
        first_crop = crop_windows[0]
        crop_width, crop_height = first_crop[2], first_crop[3]
        
        # Create video writer with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            fps, 
            (crop_width, crop_height)
        )
        
        # Check if writer was successfully created
        if not self.writer.isOpened():
            print(f"Error: Could not create video writer for {output_path}")
            print(f"Trying alternative codec...")
            # Try with a different codec
            self.writer = cv2.VideoWriter(
                output_path, 
                cv2.VideoWriter_fourcc(*'XVID'), 
                fps, 
                (crop_width, crop_height)
            )
            if not self.writer.isOpened():
                raise ValueError(f"Failed to create video writer with multiple codecs")
        
        # Process each frame
        for i, frame in enumerate(self.frame_generator()):
            if i >= len(crop_windows):
                break
            
            # Apply crop with boundary checking
            crop_window = crop_windows[i]
            try:
                cropped_frame = self.apply_crop(frame, crop_window)
                
                # Ensure cropped frame has the expected dimensions
                if cropped_frame.shape[1] != crop_width or cropped_frame.shape[0] != crop_height:
                    cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
                    
                # Write to output video
                self.writer.write(cropped_frame)
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                continue
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(crop_windows)} frames for output video")
        
        # Release resources
        if self.writer:
            self.writer.release()
    
    def __del__(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release() 