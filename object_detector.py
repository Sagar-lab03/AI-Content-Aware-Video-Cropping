from ultralytics import YOLO
import cv2
import numpy as np
import os

class ObjectDetector:
    """Class for detecting objects in video frames using YOLOv8."""
    
    def __init__(self, confidence_threshold=0.5, model_size='n'):
        """
        Initialize YOLOv8 detector.
        
        Args:
            confidence_threshold: Minimum confidence score for detections
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_size = model_size
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the YOLOv8 model."""
        try:
            model_path = f'models/yolov8{self.model_size}.pt'
            if not os.path.exists(model_path):
                print(f"Model {model_path} not found, using default model from Ultralytics")
                model_path = f'yolov8{self.model_size}.pt'
            
            self.model = YOLO(model_path)
            print(f"YOLOv8-{self.model_size} model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise
    
    def detect(self, frame):
        """
        Detect objects in a frame.
        
        Args:
            frame: The input frame (numpy array)
            
        Returns:
            List of detected objects, each as a dictionary with:
            - 'box': [x, y, width, height]
            - 'confidence': detection confidence
            - 'class_name': class name
            - 'class_id': class ID
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False)[0]
        
        # Process detections
        detections = []
        
        for det in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            
            if conf < self.confidence_threshold:
                continue
                
            # Convert to [x, y, width, height] format
            width = x2 - x1
            height = y2 - y1
            
            detection = {
                'box': [int(x1), int(y1), int(width), int(height)],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': results.names[int(cls)]
            }
            
            detections.append(detection)
        
        return detections
    
    def get_class_names(self):
        """Get list of class names the model can detect."""
        return self.model.names if self.model else {}

    def _detect_yolo(self, frame):
        """YOLO object detection implementation."""
        height, width, _ = frame.shape
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.model.setInput(blob)
        
        # Get detections
        outs = self.model.forward(self.output_layers)
        
        # Process detections
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
        detected_objects = []
        for i in indices:
            if isinstance(i, list):  # Handle different OpenCV versions
                i = i[0]
                
            box = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]
            class_name = self.classes[class_id]
            
            detected_objects.append({
                'box': box,  # [x, y, width, height]
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            })
        
        return detected_objects
    
    def _detect_ssd(self, frame):
        """SSD object detection implementation."""
        # Implementation for SSD model
        pass
    
    def _detect_faster_rcnn(self, frame):
        """Faster R-CNN object detection implementation."""
        # Implementation for Faster R-CNN model
        pass 