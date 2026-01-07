"""
Road Sign Detection - Video Testing with Generic Fallback
Handles 21 trained classes with fallback to generic "Road Sign" for unknown signs
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import torch

# Define the 21 trained road sign classes (update based on your actual classes)
TRAINED_CLASSES = {
    0: "Speed Limit 20",
    1: "Speed Limit 30", 
    2: "Speed Limit 40",
    3: "Speed Limit 50",
    4: "Speed Limit 60",
    5: "Speed Limit 70",
    6: "Speed Limit 80",
    7: "Stop Sign",
    8: "No Entry",
    9: "Yield",
    10: "Priority Road",
    11: "End of Priority Road",
    12: "No Overtaking",
    13: "No Parking",
    14: "Pedestrian Crossing",
    15: "School Zone",
    16: "Traffic Light Ahead",
    17: "Roundabout",
    18: "One Way",
    19: "Keep Right",
    20: "Keep Left"
}

class RoadSignDetector:
    def __init__(self, model_path, confidence_threshold=0.3, generic_confidence=0.15):
        """
        Initialize the road sign detector
        
        Args:
            model_path: Path to the trained YOLO model
            confidence_threshold: Confidence threshold for known classes
            generic_confidence: Lower threshold for generic "Road Sign" detection
        """
        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        self.generic_confidence = generic_confidence
        
        # Get class names from model
        try:
            self.class_names = self.model.names
            print(f"Loaded {len(self.class_names)} classes from model")
        except:
            self.class_names = TRAINED_CLASSES
            print(f"Using predefined {len(self.class_names)} classes")
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'known_class_detections': 0,
            'generic_detections': 0,
            'per_class_count': {},
            'frames_processed': 0
        }
    
    def detect_video(self, video_path, output_path=None, show_display=True):
        """
        Process video and detect road signs
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_display: Whether to display video while processing
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection (GPU accelerated if available)
            results = self.model(frame, conf=self.generic_confidence, verbose=False, device=self.device)[0]
            
            # Process detections
            annotated_frame = self._process_detections(frame, results)
            
            # Display progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames ({fps_current:.1f} fps)")
            
            # Save frame
            if writer:
                writer.write(annotated_frame)
            
            # Display frame
            if show_display:
                cv2.imshow('Road Sign Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        self._print_statistics(frame_count, time.time() - start_time)
    
    def _process_detections(self, frame, results):
        """Process detection results and annotate frame"""
        annotated_frame = frame.copy()
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Extract box info
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Determine label and color
                if conf >= self.confidence_threshold:
                    # High confidence - use specific class
                    label = self.class_names.get(cls_id, f"Class {cls_id}")
                    color = (0, 255, 0)  # Green for known classes
                    self.stats['known_class_detections'] += 1
                    self.stats['per_class_count'][label] = self.stats['per_class_count'].get(label, 0) + 1
                else:
                    # Low confidence - mark as generic road sign
                    label = "Road Sign"
                    color = (0, 165, 255)  # Orange for generic
                    self.stats['generic_detections'] += 1
                
                self.stats['total_detections'] += 1
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label_text = f"{label} {conf:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(annotated_frame, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _print_statistics(self, total_frames, elapsed_time):
        """Print detection statistics"""
        print("\n" + "="*60)
        print("DETECTION STATISTICS")
        print("="*60)
        print(f"Total frames processed: {total_frames}")
        print(f"Total time: {elapsed_time:.2f}s ({total_frames/elapsed_time:.1f} fps)")
        print(f"\nTotal detections: {self.stats['total_detections']}")
        print(f"Known class detections (conf >= {self.confidence_threshold}): {self.stats['known_class_detections']}")
        print(f"Generic 'Road Sign' detections (conf < {self.confidence_threshold}): {self.stats['generic_detections']}")
        
        if self.stats['per_class_count']:
            print("\nPer-Class Detection Count:")
            sorted_classes = sorted(self.stats['per_class_count'].items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes:
                print(f"  {class_name}: {count}")
        print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Road Sign Detection on Video")
    parser.add_argument("--model", type=str, default="weights/best.pt", 
                       help="Path to YOLO model")
    parser.add_argument("--video", type=str, 
                       default="4K _ First time in Colombo, Sri Lanka - Driving Tour.mp4",
                       help="Path to input video")
    parser.add_argument("--output", type=str, default="output_detection.mp4",
                       help="Path to output video")
    parser.add_argument("--conf", type=float, default=0.3,
                       help="Confidence threshold for known classes")
    parser.add_argument("--generic-conf", type=float, default=0.15,
                       help="Confidence threshold for generic road sign")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't display video while processing")
    
    args = parser.parse_args()
    
    # Check if files exist
    model_path = Path(args.model)
    video_path = Path(args.video)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    # Create detector and process video
    detector = RoadSignDetector(
        model_path=model_path,
        confidence_threshold=args.conf,
        generic_confidence=args.generic_conf
    )
    
    output_path = Path(args.output) if not args.no_display else None
    
    detector.detect_video(
        video_path=video_path,
        output_path=output_path,
        show_display=not args.no_display
    )

if __name__ == "__main__":
    main()
