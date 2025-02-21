# Object Detection Using YOLO

## What is YOLO?

**YOLO (You Only Look Once)** is a widely used object detection algorithm that can detect multiple objects in an image or video in real time. Unlike traditional detection systems that look at different parts of an image multiple times, YOLO processes the entire image in one pass, making it faster and more efficient.

YOLO divides the input image into grids and predicts bounding boxes and class probabilities for each grid, allowing for the detection of multiple objects simultaneously. This is useful for applications such as:
- **Self-driving cars**: For detecting pedestrians, vehicles, and other obstacles.
- **Surveillance**: For monitoring objects or people in real-time.
- **Robotics**: For object recognition and interaction in dynamic environments.

## How YOLO Works

1. **Input Image**: YOLO takes an image and splits it into a grid.
2. **Grid System**: Each grid cell predicts multiple bounding boxes and their associated confidence scores.
3. **Class Predictions**: YOLO assigns class probabilities (e.g., car, dog, person) to each bounding box.
4. **Non-Max Suppression**: Reduces overlapping bounding boxes to ensure the most confident predictions are kept.

## Using YOLO

### Step 1: Download YOLO Weights

To use YOLO, you need the pre-trained weights file for the model. In this project, we use **YOLOv3-320**, a compact version of YOLOv3 that is smaller in size and faster in detection, though slightly less accurate than larger models.

You can download the YOLOv3-320 weights file from the official YOLO website:
- [Download YOLOv3-320 Weights](https://pjreddie.com/media/files/yolov3-tiny.weights)

After downloading, place the weights file in a `weights/` directory within your project folder.

### Step 2: Run Object Detection

Once you have the weights downloaded and placed correctly, you can run object detection on images or videos.

**Detect objects in an image**:
```bash
python main.py --source path_to_image.jpg --weights weights/yolov3.weights
