"""
FUSION MODEL - INTEGRATED POTHOLE & ROAD SIGN DETECTION SYSTEM
==============================================================

Combines:
1. Optimized Pothole Detection with Recurrent Validation
2. Road Sign Detection with Generic Fallback

Features:
- Dual-model inference (pothole + road sign)
- Kalman filtering for pothole tracking
- Distance estimation for potholes
- Adaptive frame skipping
- Smart multi-scale detection
- Comprehensive visualization

Author: Enhanced Fusion Detection System
Date: 2025
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import torch
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import argparse

# ============================================
# CONFIGURATION
# ============================================

# Model paths
POTHOLE_MODEL_PATH = r'pothole_weights\best.pt'
SIGN_MODEL_PATH = r'sign_weights\best.pt'

# Detection parameters
BASE_CONF_THRESHOLD_POTHOLE = 0.20
BASE_CONF_THRESHOLD_SIGN = 0.30
GENERIC_SIGN_CONF = 0.15
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300

# Distance-aware confidence thresholds for potholes
ZONE_CONF_THRESHOLDS = {
    "CRITICAL": 0.18,
    "NEAR": 0.19,
    "MEDIUM": 0.22,
    "FAR": 0.25,
}

# Adaptive frame skipping
ENABLE_ADAPTIVE_SKIP = True
MIN_SKIP_FRAMES = 1
MAX_SKIP_FRAMES = 2
SKIP_DECAY = 0.8

# Multi-scale detection
ENABLE_SMART_MULTISCALE = True
MULTISCALE_CONF_TRIGGER = 0.45
SCALES = [640, 960]

# Kalman tracking for potholes
ENABLE_KALMAN_TRACKING = True
MIN_HITS_TO_CONFIRM = 2
ENABLE_SOFT_ADMISSION = True
SOFT_ADMISSION_IOU = 0.35
SOFT_ADMISSION_CONF = 0.30
MAX_AGE_FRAMES = 12
IOU_TRACKING_THRESHOLD = 0.25

# Distance estimation (camera calibration)
CAMERA_FOCAL_LENGTH = 700
AVERAGE_POTHOLE_WIDTH = 0.4
CAMERA_HEIGHT = 1.0
CAMERA_ANGLE = 10

# Alert zones
DISTANCE_ZONES = {
    "CRITICAL": (0, 10),
    "NEAR": (10, 25),
    "MEDIUM": (25, 50),
    "FAR": (50, 100),
}

ZONE_COLORS = {
    "CRITICAL": (255, 0, 255),   # Magenta
    "NEAR": (0, 255, 255),       # Yellow
    "MEDIUM": (0, 165, 255),     # Orange
    "FAR": (0, 0, 255),          # Red
}

# Ground-level road sign filtering parameters
SIGN_MIN_Y_RATIO = 0.2           # Signs below top 20% of frame (filter aerial signs)
SIGN_MAX_AREA_RATIO = 0.15       # Max 15% of frame (filter billboards)
SIGN_MIN_SIZE = 20               # Minimum 20 pixels width/height
SIGN_MIN_ASPECT_RATIO = 0.3      # Minimum aspect ratio
SIGN_MAX_ASPECT_RATIO = 3.0      # Maximum aspect ratio
SIGN_MIN_X_RATIO = 0.05          # Minimum 5% from left edge
SIGN_MAX_X_RATIO = 0.95          # Maximum 95% from left edge
SIGN_CLOSE_MIN_AREA = 800        # Min area for close signs (bottom of frame)
SIGN_FAR_MAX_AREA_RATIO = 0.08  # Max area ratio for far signs (top of frame)

# Road sign classes (21 classes)
TRAINED_SIGN_CLASSES = {
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

# ============================================
# KALMAN FILTER FOR POTHOLE TRACKING
# ============================================

@dataclass
class Track:
    """Represents a tracked pothole detection over time"""
    id: int
    bbox: np.ndarray
    confidence: float
    hits: int = 1
    age: int = 0
    distance: float = 0.0
    velocity: float = 0.0
    last_seen_frame: int = 0
    confirmed: bool = False

class KalmanBoxTracker:
    """Kalman Filter for tracking bounding boxes over time"""
    count = 0
    
    def __init__(self, bbox, confidence):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        x, y, w, h = self.bbox_to_xyxy(bbox)
        self.kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(-1, 1)
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.confidence = confidence
        self.history_confidences = deque([confidence], maxlen=10)
        
    def bbox_to_xyxy(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    
    def xyxy_to_bbox(self, x, y, w, h):
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])
    
    def update(self, bbox, confidence):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence
        self.history_confidences.append(confidence)
        
        x, y, w, h = self.bbox_to_xyxy(bbox)
        measurement = np.array([x, y, w, h], dtype=np.float32).reshape(-1, 1)
        self.kf.correct(measurement)
    
    def predict(self):
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        state = self.kf.statePost.flatten()
        x, y, w, h = state[0], state[1], state[2], state[3]
        return self.xyxy_to_bbox(x, y, w, h)
    
    def get_state(self):
        state = self.kf.statePost.flatten()
        x, y, w, h = state[0], state[1], state[2], state[3]
        return self.xyxy_to_bbox(x, y, w, h)
    
    def get_avg_confidence(self):
        return float(np.mean(self.history_confidences))

# ============================================
# RECURRENT TRACKER
# ============================================

class RecurrentTracker:
    """Manages multiple Kalman trackers for pothole detection"""
    def __init__(self, min_hits=2, max_age=12, iou_threshold=0.25, 
                 enable_soft_admission=True, soft_iou=0.35, soft_conf=0.30):
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.enable_soft_admission = enable_soft_admission
        self.soft_iou = soft_iou
        self.soft_conf = soft_conf
        self.trackers = []
        self.frame_count = 0
        self.next_id = 0
        
    def update(self, detections):
        self.frame_count += 1
        
        for trk in self.trackers:
            trk.predict()
        
        matched, unmatched_dets, unmatched_trks = self.associate_detections(detections)
        
        for trk_idx, det_idx in matched:
            self.trackers[trk_idx].update(
                detections[det_idx]['bbox'],
                detections[det_idx]['confidence']
            )
        
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(
                detections[det_idx]['bbox'],
                detections[det_idx]['confidence']
            )
            self.trackers.append(trk)
        
        self.trackers = [
            trk for trk in self.trackers
            if trk.time_since_update <= self.max_age
        ]
        
        confirmed_tracks = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits and trk.time_since_update == 0:
                confirmed_tracks.append({
                    'id': trk.id,
                    'bbox': trk.get_state(),
                    'confidence': trk.get_avg_confidence(),
                    'hits': trk.hits,
                    'confirmed': True
                })
            elif (self.enable_soft_admission and 
                  trk.hits == 1 and 
                  trk.time_since_update == 0 and
                  trk.confidence >= self.soft_conf):
                confirmed_tracks.append({
                    'id': trk.id,
                    'bbox': trk.get_state(),
                    'confidence': trk.get_avg_confidence(),
                    'hits': trk.hits,
                    'confirmed': False
                })
        
        return confirmed_tracks
    
    def associate_detections(self, detections):
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        
        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = self.calculate_iou(det['bbox'], trk.get_state())
        
        matched_indices = []
        unmatched_detections = []
        unmatched_trackers = list(range(len(self.trackers)))
        
        for d in range(len(detections)):
            if len(unmatched_trackers) == 0:
                unmatched_detections.append(d)
                continue
            
            ious = iou_matrix[d, unmatched_trackers]
            best_idx = np.argmax(ious)
            best_iou = ious[best_idx]
            
            if best_iou >= self.iou_threshold:
                matched_indices.append((unmatched_trackers[best_idx], d))
                unmatched_trackers.pop(best_idx)
            else:
                unmatched_detections.append(d)
        
        return matched_indices, unmatched_detections, unmatched_trackers
    
    @staticmethod
    def calculate_iou(bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

# ============================================
# DISTANCE ESTIMATION
# ============================================

class DistanceEstimator:
    """Estimate real-world distance to potholes"""
    def __init__(self, focal_length, avg_width, camera_height, camera_angle):
        self.focal_length = focal_length
        self.avg_width = avg_width
        self.camera_height = camera_height
        self.camera_angle = np.radians(camera_angle)
        
    def estimate_distance(self, bbox, frame_height):
        x1, y1, x2, y2 = bbox
        y_bottom = y2
        y_ratio = y_bottom / frame_height
        
        # Perspective-based distance estimation
        if y_ratio < 0.4:
            distance_perspective = 50 + (0.4 - y_ratio) * 125
        elif y_ratio < 0.6:
            distance_perspective = 25 + (0.6 - y_ratio) * 125
        elif y_ratio < 0.75:
            distance_perspective = 10 + (0.75 - y_ratio) * 100
        else:
            distance_perspective = 2 + (1.0 - y_ratio) * 32
        
        # Width-based estimation
        bbox_width_pixels = x2 - x1
        if bbox_width_pixels > 15:
            distance_width = (self.avg_width * self.focal_length) / bbox_width_pixels
            distance_width = max(2, min(distance_width, 150))
        else:
            distance_width = distance_perspective
        
        # Combine methods
        if y_ratio > 0.8:
            distance = distance_width * 0.6 + distance_perspective * 0.4
        elif y_ratio > 0.65:
            distance = distance_width * 0.4 + distance_perspective * 0.6
        else:
            distance = distance_width * 0.2 + distance_perspective * 0.8
        
        distance = max(2, min(distance, 120))
        return distance
    
    def get_zone(self, distance):
        for zone, (min_dist, max_dist) in DISTANCE_ZONES.items():
            if min_dist <= distance < max_dist:
                return zone
        return "FAR"

# ============================================
# ADAPTIVE FRAME SKIPPER
# ============================================

class AdaptiveFrameSkipper:
    """Intelligently skip frames based on detection activity"""
    def __init__(self, min_skip=1, max_skip=3, decay=0.8):
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.decay = decay
        self.current_skip = max_skip
        self.frames_since_detection = 0
        
    def update(self, has_detections):
        if has_detections:
            self.current_skip = self.min_skip
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
            if self.frames_since_detection > 5:
                self.current_skip = min(
                    self.current_skip + 1,
                    self.max_skip
                )
        
        return int(self.current_skip)
    
    def should_process(self, frame_num):
        return frame_num % max(1, int(self.current_skip)) == 0

# ============================================
# SMART MULTI-SCALE DETECTION
# ============================================

def smart_multiscale_detect(model, frame, base_conf, scales, trigger_conf):
    """Adaptive multi-scale detection for potholes"""
    results_640 = model(frame, imgsz=scales[0], conf=base_conf, iou=IOU_THRESHOLD, verbose=False)[0]
    
    detections = []
    max_conf = 0.0
    
    for box in results_640.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        max_conf = max(max_conf, conf)
        detections.append({
            'bbox': np.array([x1, y1, x2, y2]),
            'confidence': conf,
            'scale': scales[0]
        })
    
    if (max_conf < trigger_conf or len(detections) < 2) and len(scales) > 1:
        for scale in scales[1:]:
            results = model(frame, imgsz=scale, conf=base_conf*0.7, iou=IOU_THRESHOLD, verbose=False)[0]
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                
                is_new = True
                for existing in detections:
                    iou = RecurrentTracker.calculate_iou(
                        np.array([x1, y1, x2, y2]),
                        existing['bbox']
                    )
                    if iou > 0.4:
                        if conf > existing['confidence']:
                            existing['confidence'] = conf
                            existing['bbox'] = np.array([x1, y1, x2, y2])
                            existing['scale'] = scale
                        is_new = False
                        break
                
                if is_new:
                    detections.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'confidence': conf,
                        'scale': scale
                    })
    
    return detections

# ============================================
# FUSION DETECTION SYSTEM
# ============================================

class FusionDetector:
    """Integrated Pothole and Road Sign Detection System"""
    
    def __init__(self, pothole_model_path, sign_model_path):
        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load models
        print(f"\n📦 Loading pothole model from: {pothole_model_path}")
        self.pothole_model = YOLO(pothole_model_path)
        self.pothole_model.to(self.device)
        
        print(f"📦 Loading sign model from: {sign_model_path}")
        self.sign_model = YOLO(sign_model_path)
        self.sign_model.to(self.device)
        
        # Get sign class names
        try:
            self.sign_class_names = self.sign_model.names
            print(f"   Loaded {len(self.sign_class_names)} sign classes")
        except:
            self.sign_class_names = TRAINED_SIGN_CLASSES
            print(f"   Using predefined {len(self.sign_class_names)} sign classes")
        
        # Initialize components
        self.pothole_tracker = RecurrentTracker(
            min_hits=MIN_HITS_TO_CONFIRM,
            max_age=MAX_AGE_FRAMES,
            iou_threshold=IOU_TRACKING_THRESHOLD,
            enable_soft_admission=ENABLE_SOFT_ADMISSION,
            soft_iou=SOFT_ADMISSION_IOU,
            soft_conf=SOFT_ADMISSION_CONF
        ) if ENABLE_KALMAN_TRACKING else None
        
        self.distance_estimator = DistanceEstimator(
            CAMERA_FOCAL_LENGTH,
            AVERAGE_POTHOLE_WIDTH,
            CAMERA_HEIGHT,
            CAMERA_ANGLE
        )
        
        self.frame_skipper = AdaptiveFrameSkipper(
            MIN_SKIP_FRAMES,
            MAX_SKIP_FRAMES,
            SKIP_DECAY
        ) if ENABLE_ADAPTIVE_SKIP else None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'pothole_detections': 0,
            'sign_detections': 0,
            'zone_counts': defaultdict(int),
            'sign_counts': defaultdict(int),
            'processing_times': [],
            'multiscale_triggers': 0
        }
    
    def process_video(self, video_path, output_path=None, show_display=True):
        """Process video with fusion detection"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {video_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n🎥 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"💾 Output: {output_path}")
        
        print("\n🚀 Starting fusion detection... Press 'q' to quit\n")
        start_time = time.time()
        
        frame_num = 0
        last_pothole_detections = []
        last_sign_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            self.stats['total_frames'] += 1
            
            # Adaptive frame skipping
            should_process = True
            if self.frame_skipper:
                should_process = self.frame_skipper.should_process(frame_num)
                if not should_process:
                    self.stats['skipped_frames'] += 1
                    display_frame = self._draw_cached_detections(
                        frame, last_pothole_detections, last_sign_detections, height
                    )
                    
                    if show_display:
                        cv2.imshow('Fusion Detection', display_frame)
                    if writer:
                        writer.write(display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
            
            self.stats['processed_frames'] += 1
            frame_start = time.time()
            
            # Detect potholes
            pothole_detections = self._detect_potholes(frame)
            
            # Detect signs
            sign_detections = self._detect_signs(frame)
            
            # Update frame skipper
            if self.frame_skipper:
                has_detections = len(pothole_detections) > 0 or len(sign_detections) > 0
                self.frame_skipper.update(has_detections)
            
            # Cache detections
            last_pothole_detections = pothole_detections
            last_sign_detections = sign_detections
            
            # Visualize
            display_frame = self._visualize_detections(
                frame, pothole_detections, sign_detections, 
                frame_num, total_frames, time.time() - frame_start, height
            )
            
            self.stats['processing_times'].append(time.time() - frame_start)
            
            # Display and save
            if show_display:
                cv2.imshow('Fusion Detection', display_frame)
            if writer:
                writer.write(display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Stopped by user")
                break
            
            # Progress update
            if frame_num % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_num / elapsed
                print(f"   Processed {frame_num}/{total_frames} frames ({fps_current:.1f} fps)")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        self._print_statistics(time.time() - start_time)
    
    def _detect_potholes(self, frame):
        """Detect potholes with tracking"""
        if ENABLE_SMART_MULTISCALE:
            detections = smart_multiscale_detect(
                self.pothole_model, frame, BASE_CONF_THRESHOLD_POTHOLE,
                SCALES, MULTISCALE_CONF_TRIGGER
            )
            if any(d['scale'] > SCALES[0] for d in detections):
                self.stats['multiscale_triggers'] += 1
        else:
            results = self.pothole_model(frame, conf=BASE_CONF_THRESHOLD_POTHOLE, 
                                        iou=IOU_THRESHOLD, verbose=False)[0]
            detections = [
                {
                    'bbox': np.array(box.xyxy[0].cpu().numpy(), dtype=int),
                    'confidence': float(box.conf[0])
                }
                for box in results.boxes
            ]
        
        # Apply Kalman tracking
        if self.pothole_tracker and detections:
            confirmed_detections = self.pothole_tracker.update(detections)
        else:
            confirmed_detections = detections
        
        return confirmed_detections
    
    def _detect_signs(self, frame):
        """Detect road signs with ground-level filtering"""
        results = self.sign_model(frame, conf=GENERIC_SIGN_CONF, verbose=False, device=self.device)[0]
        
        frame_height, frame_width = frame.shape[:2]
        sign_detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # === GROUND-LEVEL ROAD SIGN FILTERING ===
                
                # 1. Vertical position filter (signs should be in middle-to-lower part of frame)
                # Top portion of frame is usually sky/background - filter out
                y_center = (y1 + y2) / 2
                y_ratio = y_center / frame_height
                if y_ratio < SIGN_MIN_Y_RATIO:  # Too high in frame (likely aerial/billboard)
                    continue
                
                # 2. Size filter (ground-level signs have reasonable pixel size)
                sign_width = x2 - x1
                sign_height = y2 - y1
                sign_area = sign_width * sign_height
                
                # Filter out extremely large signs (billboards, aerial signs)
                if sign_area > (frame_width * frame_height * SIGN_MAX_AREA_RATIO):
                    continue
                
                # Filter out tiny signs (noise, distant irrelevant objects)
                if sign_width < SIGN_MIN_SIZE or sign_height < SIGN_MIN_SIZE:
                    continue
                
                # 3. Aspect ratio filter (road signs are usually squarish or rectangular)
                aspect_ratio = sign_width / sign_height if sign_height > 0 else 0
                # Most road signs have reasonable aspect ratios
                if aspect_ratio < SIGN_MIN_ASPECT_RATIO or aspect_ratio > SIGN_MAX_ASPECT_RATIO:
                    continue
                
                # 4. Position validation (signs should be within reasonable horizontal bounds)
                x_center = (x1 + x2) / 2
                x_ratio = x_center / frame_width
                # Allow signs across full width but filter extreme edges
                if x_ratio < SIGN_MIN_X_RATIO or x_ratio > SIGN_MAX_X_RATIO:
                    continue
                
                # 5. Distance-based size validation
                # Signs closer to bottom should be larger (perspective)
                if y_ratio > 0.7:  # Lower part of frame (closer)
                    if sign_area < SIGN_CLOSE_MIN_AREA:  # Close signs should be reasonably sized
                        continue
                elif y_ratio < 0.35:  # Upper-middle part (farther)
                    if sign_area > frame_width * frame_height * SIGN_FAR_MAX_AREA_RATIO:  # Far signs shouldn't be huge
                        continue
                
                # === END FILTERING ===
                
                # Determine label
                if conf >= BASE_CONF_THRESHOLD_SIGN:
                    label = self.sign_class_names.get(cls_id, f"Class {cls_id}")
                    is_generic = False
                else:
                    label = "Road Sign"
                    is_generic = True
                
                sign_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'label': label,
                    'is_generic': is_generic
                })
        
        return sign_detections
    
    def _visualize_detections(self, frame, pothole_dets, sign_dets, 
                              frame_num, total_frames, frame_time, height):
        """Draw all detections on frame"""
        display_frame = frame.copy()
        
        max_alert_level = 0
        closest_distance = 999
        
        # Draw potholes
        for det in pothole_dets:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            conf = det['confidence']
            
            # Filter tiny boxes
            box_area = (x2 - x1) * (y2 - y1)
            y_ratio = y2 / height
            min_area = 50 if y_ratio < 0.5 else 150
            if box_area < min_area:
                continue
            
            # Distance estimation
            distance = self.distance_estimator.estimate_distance(bbox, height)
            zone = self.distance_estimator.get_zone(distance)
            
            # Zone-aware confidence filtering
            zone_threshold = ZONE_CONF_THRESHOLDS.get(zone, BASE_CONF_THRESHOLD_POTHOLE)
            if conf < zone_threshold:
                continue
            
            color = ZONE_COLORS.get(zone, (0, 255, 0))
            
            # Update stats
            self.stats['pothole_detections'] += 1
            self.stats['zone_counts'][zone] += 1
            closest_distance = min(closest_distance, distance)
            
            alert_levels = {"CRITICAL": 4, "NEAR": 3, "MEDIUM": 2, "FAR": 1}
            max_alert_level = max(max_alert_level, alert_levels.get(zone, 0))
            
            # Draw detection
            thickness = 3 if zone in ["CRITICAL", "NEAR"] else 2
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"POTHOLE {zone} {distance:.1f}m {conf:.2f}"
            if 'hits' in det:
                label += f" H:{det['hits']}"
            
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1-lh-8), (x1+lw+4, y1), color, -1)
            cv2.putText(display_frame, label, (x1+2, y1-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw signs
        for det in sign_dets:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['label']
            is_generic = det['is_generic']
            
            # Color coding
            color = (0, 165, 255) if is_generic else (0, 255, 0)  # Orange for generic, green for known
            
            # Draw detection
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label_text = f"{label} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(display_frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update stats
            self.stats['sign_detections'] += 1
            self.stats['sign_counts'][label] += 1
        
        # Warning banner for potholes
        if max_alert_level >= 3:  # CRITICAL or NEAR
            warning_text = f"!!! POTHOLE {closest_distance:.1f}m AHEAD - SLOW DOWN !!!"
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(display_frame, warning_text, (20, 28),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        elif max_alert_level >= 2:  # MEDIUM
            warning_text = f"Pothole {closest_distance:.1f}m ahead - Prepare"
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 35), (0, 140, 255), -1)
            cv2.putText(display_frame, warning_text, (20, 23),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Info panel
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        info = [
            f"Frame: {frame_num}/{total_frames}",
            f"Potholes: {len(pothole_dets)}",
            f"Signs: {len(sign_dets)}",
            f"FPS: {current_fps:.1f}",
        ]
        
        y_offset = 50 if max_alert_level >= 2 else 15
        for i, text in enumerate(info):
            cv2.putText(display_frame, text, (10, y_offset + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(display_frame, text, (10, y_offset + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def _draw_cached_detections(self, frame, pothole_dets, sign_dets, height):
        """Draw cached detections for skipped frames"""
        display_frame = frame.copy()
        
        # Draw potholes (grayed out)
        for det in pothole_dets:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (100, 100, 100), 2)
            cv2.putText(display_frame, "TRACKING", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Draw signs (grayed out)
        for det in sign_dets:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (100, 100, 100), 2)
            cv2.putText(display_frame, "TRACKING", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return display_frame
    
    def _print_statistics(self, elapsed_time):
        """Print comprehensive statistics"""
        avg_fps = self.stats['processed_frames'] / elapsed_time if elapsed_time > 0 else 0
        avg_proc_time = np.mean(self.stats['processing_times']) * 1000 if self.stats['processing_times'] else 0
        speedup = self.stats['total_frames'] / self.stats['processed_frames'] if self.stats['processed_frames'] > 0 else 1
        
        print("\n" + "="*80)
        print("FUSION DETECTION COMPLETE - PERFORMANCE REPORT")
        print("="*80)
        
        print(f"\n⏱️  Processing Summary:")
        print(f"   Total Frames: {self.stats['total_frames']}")
        print(f"   Processed Frames: {self.stats['processed_frames']}")
        print(f"   Skipped Frames: {self.stats['skipped_frames']} ({self.stats['skipped_frames']/self.stats['total_frames']*100:.1f}%)")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Total Time: {elapsed_time:.2f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Avg Processing Time: {avg_proc_time:.1f}ms/frame")
        
        print(f"\n🕳️  Pothole Detection Statistics:")
        print(f"   Total Pothole Detections: {self.stats['pothole_detections']}")
        print(f"\n   Distance Distribution:")
        for zone in ["CRITICAL", "NEAR", "MEDIUM", "FAR"]:
            count = self.stats['zone_counts'][zone]
            pct = count / max(self.stats['pothole_detections'], 1) * 100
            print(f"      {zone:10s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n🚸 Road Sign Detection Statistics:")
        print(f"   Total Sign Detections: {self.stats['sign_detections']}")
        if self.stats['sign_counts']:
            print(f"\n   Top Detected Signs:")
            sorted_signs = sorted(self.stats['sign_counts'].items(), key=lambda x: x[1], reverse=True)
            for sign_name, count in sorted_signs[:10]:
                print(f"      {sign_name}: {count}")
        
        if ENABLE_SMART_MULTISCALE:
            print(f"\n🔬 Multi-scale Statistics:")
            print(f"   Multi-scale Triggered: {self.stats['multiscale_triggers']} times")
            print(f"   Multi-scale Rate: {self.stats['multiscale_triggers']/self.stats['processed_frames']*100:.1f}%")
        
        print("\n" + "="*80)

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Fusion Model - Pothole & Road Sign Detection")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video")
    parser.add_argument("--output", type=str, default="fusion_output.mp4",
                       help="Path to output video")
    parser.add_argument("--pothole-model", type=str, default=POTHOLE_MODEL_PATH,
                       help="Path to pothole detection model")
    parser.add_argument("--sign-model", type=str, default=SIGN_MODEL_PATH,
                       help="Path to sign detection model")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't display video while processing")
    
    args = parser.parse_args()
    
    # Check if files exist
    pothole_model_path = Path(args.pothole_model)
    sign_model_path = Path(args.sign_model)
    video_path = Path(args.video)
    
    if not pothole_model_path.exists():
        print(f"❌ Error: Pothole model not found at {pothole_model_path}")
        return
    
    if not sign_model_path.exists():
        print(f"❌ Error: Sign model not found at {sign_model_path}")
        return
    
    if not video_path.exists():
        print(f"❌ Error: Video not found at {video_path}")
        return
    
    print("="*80)
    print("FUSION MODEL - INTEGRATED POTHOLE & ROAD SIGN DETECTION")
    print("="*80)
    
    # Create detector
    detector = FusionDetector(
        pothole_model_path=pothole_model_path,
        sign_model_path=sign_model_path
    )
    
    # Process video
    detector.process_video(
        video_path=video_path,
        output_path=args.output if not args.no_display else None,
        show_display=not args.no_display
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
