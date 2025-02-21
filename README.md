# Real-Time Object Detection Using YOLO

## Overview
This project implements real-time object detection using YOLO (You Only Look Once) with both a standard OpenCV interface and a Streamlit web interface.

## Requirements
- Python 3.7+
- OpenCV
- Streamlit
- NumPy
- YOLO weights and configuration files

## Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Real-Time-Object-Detection-using-YOLO-main
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download YOLO files**
- Download YOLOv3 weights from: https://pjreddie.com/media/files/yolov3.weights
- Download YOLOv3 config from: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
- Download COCO names from: https://github.com/pjreddie/darknet/blob/master/data/coco.names

Place these files in the project root directory.

## Usage

### Streamlit Interface
```bash
streamlit run app.py
```
This will open a web interface with:
- Adjustable confidence threshold
- Camera source selection
- Start/Stop controls
- Real-time video feed with object detection

### Standard OpenCV Interface
```bash
python main.py
```
- Press 'q' to quit the application

## Features
- Real-time object detection
- Multiple interface options (Streamlit/OpenCV)
- Adjustable detection confidence
- Support for multiple camera sources
- Class probability display
- Non-maximum suppression for better detection

## Project Structure
```
├── app.py              # Streamlit interface
├── main.py            # OpenCV interface
├── requirements.txt   # Project dependencies
├── yolov3.weights    # YOLO model weights
├── yolov3.cfg        # YOLO model configuration
└── coco.names        # Class names file
```
