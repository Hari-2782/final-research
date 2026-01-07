# Fusion Detection System - Pothole & Road Sign Detection

## Overview
This fusion model combines two powerful detection systems:
1. **Pothole Detection** with Kalman filtering and distance estimation
2. **Road Sign Detection** with 21 trained classes + generic fallback

## Features
✅ **Dual Model Inference** - Simultaneous pothole and road sign detection  
✅ **Recurrent Validation** - Kalman filtering reduces false positives  
✅ **Distance Estimation** - Accurate metric distance to potholes  
✅ **Adaptive Frame Skipping** - 3-5x speed improvement  
✅ **Smart Multi-Scale Detection** - Better small object detection  
✅ **Zone-Aware Thresholding** - Distance-adaptive confidence filtering  
✅ **Comprehensive Visualization** - Color-coded alerts and warnings  

## Quick Start

### Basic Usage
```bash
python fusion_detection.py --video "your_video.mp4"
```

### With Custom Output Path
```bash
python fusion_detection.py --video "your_video.mp4" --output "result.mp4"
```

### Without Display (Faster Processing)
```bash
python fusion_detection.py --video "your_video.mp4" --no-display
```

### With Custom Model Paths
```bash
python fusion_detection.py --video "your_video.mp4" --pothole-model "pothole_weights/best.pt" --sign-model "sign_weights/best.pt"
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | Yes | - | Path to input video file |
| `--output` | No | `fusion_output.mp4` | Path to output video |
| `--pothole-model` | No | `pothole_weights/best.pt` | Path to pothole model |
| `--sign-model` | No | `sign_weights/best.pt` | Path to sign model |
| `--no-display` | No | False | Disable video display |

## Requirements
```bash
pip install ultralytics opencv-python numpy torch
```

## Model Weights
The system uses two pre-trained models:
- **Pothole Model**: `pothole_weights/best.pt`
- **Sign Model**: `sign_weights/best.pt`

Both models are included in the project directory.

## Detection Zones (Potholes)
- 🔴 **CRITICAL** (0-10m): Magenta - Immediate danger
- 🟡 **NEAR** (10-25m): Yellow - Prepare to avoid
- 🟠 **MEDIUM** (25-50m): Orange - Advance warning
- 🔴 **FAR** (50-100m): Red - Early detection

## Road Sign Classes
The system detects 21 specific road sign types:
- Speed limits (20-80 km/h)
- Stop Sign, No Entry, Yield
- Priority Road, No Overtaking, No Parking
- Pedestrian Crossing, School Zone
- Roundabout, One Way
- Keep Right/Left, and more

**Plus** a generic "Road Sign" fallback for unrecognized signs.

## Output
The fusion detection creates:
1. **Video Output**: Annotated video with bounding boxes, labels, and warnings
2. **Console Statistics**: Comprehensive performance metrics
3. **Real-time Display**: Live video feed with detections (if not disabled)

## Performance
- **Processing Speed**: 15-30 FPS (depending on hardware)
- **Speedup**: 2-3x with adaptive frame skipping
- **GPU Acceleration**: Automatic CUDA support if available

## Controls
- Press **'q'** to quit during processing

## Example Output Statistics
```
FUSION DETECTION COMPLETE - PERFORMANCE REPORT
================================================================================

⏱️  Processing Summary:
   Total Frames: 3000
   Processed Frames: 1500
   Skipped Frames: 1500 (50.0%)
   Speedup: 2.00x
   Total Time: 120.50s
   Average FPS: 24.9
   Avg Processing Time: 40.2ms/frame

🕳️  Pothole Detection Statistics:
   Total Pothole Detections: 156

   Distance Distribution:
      CRITICAL  :   12 ( 7.7%)
      NEAR      :   34 (21.8%)
      MEDIUM    :   67 (42.9%)
      FAR       :   43 (27.6%)

🚸 Road Sign Detection Statistics:
   Total Sign Detections: 89

   Top Detected Signs:
      Speed Limit 50: 23
      Stop Sign: 15
      Road Sign: 18
      ...
```

## Tips
1. **For Better Performance**: Use `--no-display` flag
2. **For Higher Accuracy**: Ensure good video quality and lighting
3. **GPU Required**: For real-time processing on high-res videos
4. **Video Format**: Supports MP4, AVI, MOV, and most common formats

## Technical Details

### Pothole Detection
- **Model**: YOLOv8 with best.pt weights (84.6% mAP@50)
- **Tracking**: Kalman filter with soft admission
- **Distance**: Perspective-based + width-based estimation

### Road Sign Detection
- **Model**: YOLOv8 with best.pt weights
- **Classes**: 21 trained road sign types
- **Fallback**: Generic detection for unknown signs

## Troubleshooting

### Model Not Found
- Ensure `pothole_weights/best.pt` and `sign_weights/best.pt` exist
- Use `--pothole-model` and `--sign-model` to specify custom paths

### Video Not Opening
- Check video file path and format
- Ensure OpenCV can read the video codec

### Low FPS
- Enable GPU acceleration (install CUDA + PyTorch GPU)
- Use `--no-display` flag
- Reduce video resolution

### No Detections
- Check model confidence thresholds in code
- Ensure video quality is sufficient
- Verify models are correctly loaded

## Contact & Support
For issues, improvements, or questions about the fusion detection system, refer to the project documentation.

---
**Version**: 1.0  
**Last Updated**: January 2025
