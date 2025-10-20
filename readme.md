# Clear Path Detection

A real-time computer vision system for detecting clear navigation paths using object detection, depth estimation, and image segmentation. This project combines YOLOv8n models with MiDaS depth estimation to identify safe navigation routes in real-time video streams.

## 🚀 Features

- **Real-time Object Detection**: Uses YOLOv8n for detecting obstacles and objects
- **Depth Estimation**: Integrates MiDaS DPT_Hybrid for depth perception
- **Image Segmentation**: Employs YOLOv8n-seg for precise object segmentation
- **Path Extraction**: Intelligent decision-making algorithms to identify clear paths
- **IP Camera Support**: Real-time processing from network webcams
- **Optimized Performance**: Time-based throttling for efficient processing

## 📁 Project Structure

```
clear-path-detection/
├── src/
│   └── camera.py              # Main real-time processing script
├── notebook/
│   ├── full_model.ipynb       # Complete model demonstration
│   ├── clear_path_extraction_notebook.ipynb  # Path extraction algorithms
│   └── object_detection-segmentation-depth-path.ipynb  # Core CV pipeline
├── model/
│   ├── yolov8n.pt            # YOLOv8n object detection model
│   └── yolov8n-seg.pt        # YOLOv8n segmentation model
├── images/                   # Sample images and annotations
├── videos/                   # Sample videos and outputs
└── requirements.txt          # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd clear-path-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models** (if not already present):
   - YOLOv8n models will be automatically downloaded on first run
   - MiDaS models are loaded from torch hub

## 🚀 Quick Start

### Real-time Processing with IP Camera

```python
from src.camera import ThrottledIPWebcamProcessor
import torch

# Initialize processor
processor = ThrottledIPWebcamProcessor(
    camera_url="http://YOUR_IP:9191/video",  # Replace with your camera IP
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    process_interval=0.9  # Process every 0.9 seconds
)

# Start processing
processor.run()
```

### Running the Notebooks

1. **Object Detection & Segmentation**: 
   ```bash
   jupyter notebook notebook/object_detection-segmentation-depth-path.ipynb
   ```

2. **Path Extraction**: 
   ```bash
   jupyter notebook notebook/clear_path_extraction_notebook.ipynb
   ```

3. **Full Model Demo**: 
   ```bash
   jupyter notebook notebook/full_model.ipynb
   ```

## 🔧 Configuration

### Camera Settings
- **IP Camera URL**: Modify the `IP` variable in `camera.py`
- **Processing Interval**: Adjust `process_interval` for performance tuning
- **Resolution**: Default processing resolution is 640x640

### Detection Parameters
- **Confidence Threshold**: 0.25 (adjustable in model calls)
- **IoU Threshold**: 0.45
- **Group Width**: 64 pixels for path analysis
- **Focus Band**: Lower 30% of image (0.7-1.0)

### Path Detection Thresholds
- **Occupancy Max**: 0.02 (max object coverage)
- **Near Threshold**: 0.6 (depth threshold)
- **Near Max**: 0.2 (max near objects)

## 🎯 Key Algorithms

### Clear Path Detection Pipeline

1. **Object Detection**: YOLOv8n identifies obstacles
2. **Segmentation**: YOLOv8n-seg creates precise masks
3. **Depth Estimation**: MiDaS provides depth information
4. **Path Analysis**: Grid-based evaluation of clear areas
5. **Decision Making**: Threshold-based path selection

### Performance Optimization

- **Time-based Throttling**: Process frames at specified intervals
- **Buffer Management**: Minimal buffering for real-time performance
- **GPU Acceleration**: Automatic CUDA detection and usage

## 📊 Output

The system provides:
- **Visual Annotations**: Bounding boxes, masks, and depth visualization
- **Clear Path Indicators**: Green rectangles marking safe navigation areas
- **Real-time Statistics**: FPS, processing ratio, and clear path groups
- **Console Logging**: Detailed processing information

## 🔍 Use Cases

- **Autonomous Navigation**: Robot path planning
- **Surveillance Systems**: Security monitoring with path analysis
- **Smart Transportation**: Vehicle navigation assistance
- **Industrial Automation**: Warehouse robot navigation

## 🛡️ Requirements

- **Python**: 3.8+
- **CUDA**: Optional but recommended for GPU acceleration
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Camera**: IP camera or webcam with network access

## 📝 Notes

- Models are automatically downloaded on first run
- Processing performance depends on hardware capabilities
- Adjust parameters based on your specific use case
- The system works best with forward-facing cameras

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source. 

## 🆘 Support

For issues and questions:
1. Check the notebook examples
2. Review the configuration parameters
3. Ensure all dependencies are installed
4. Verify camera connectivity

---

**Note**: This system is designed for research and development purposes. Always test thoroughly before deploying in production environments.
```

## requirements.txt

```txt
# Core Computer Vision Libraries
opencv-python>=4.6.0
pillow>=7.1.2
numpy>=1.23.0

# Deep Learning Frameworks
torch>=1.8.0
torchvision>=0.9.0

# YOLO and Object Detection
ultralytics>=8.0.0
supervision>=0.26.0

# Image Processing and Visualization
matplotlib>=3.3.0
imageio[ffmpeg]>=2.25.0
seaborn>=0.13.0

# Data Processing
scipy>=1.4.1
pandas>=1.3.0

# Network and HTTP
requests>=2.23.0

# Configuration
pyyaml>=5.3.1

# System Utilities
psutil>=5.8.0

# Jupyter Notebook Support
jupyter>=1.0.0
ipykernel>=6.0.0

```

