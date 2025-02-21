import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image

def load_yolo_model():
    # Load YOLO model
    yolo = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # Load class names
    classes = []
    with open("./coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return yolo, classes

def process_frame(frame, yolo, classes, conf_threshold):
    frame_height, frame_width = frame.shape[:2]
    
    # Create a blob from the frame
    blob = cv.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo.setInput(blob)
    
    # Get predictions
    layer_names = yolo.getUnconnectedOutLayersNames()
    layer_output = yolo.forward(layer_names)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Process detections
    for output in layer_output:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply NMS
    indexes = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
    
    # Draw boxes
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} ({confidences[i]:.2f})"
            
            # Draw rectangle and label
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    st.title("Real-Time Object Detection with YOLO")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    camera_source = st.sidebar.selectbox("Camera Source", [0, 1, 2], index=0)
    
    # Initialize YOLO
    yolo, classes = load_yolo_model()
    
    # Start button
    if st.sidebar.button("Start Detection"):
        cap = cv.VideoCapture(camera_source)
        
        # Create a placeholder for the video feed
        video_placeholder = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read from camera")
                break
                
            # Process frame
            processed_frame = process_frame(frame, yolo, classes, conf_threshold)
            
            # Convert BGR to RGB
            processed_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
            
            # Display the frame
            video_placeholder.image(processed_frame)
            
            # Stop button
            if st.sidebar.button("Stop"):
                break
                
        cap.release()

if __name__ == "__main__":
    main()
