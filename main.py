import cv2 as cv
import numpy as np

# Load YOLO model
yolo = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
classes = []
with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the video capture object (0 is for webcam, you can replace it with a video file path)
cap = cv.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Process the video stream in real-time
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Create a blob from the current frame
    blob = cv.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo.setInput(blob)

    # Get the output layer names
    layer_names = yolo.getUnconnectedOutLayersNames()
    layer_output = yolo.forward(layer_names)

    # Initialize lists for detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Iterate over each detection
    for output in layer_output:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Scale the bounding box back to the size of the frame
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append the box, confidence, and class ID to the lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Define font and colors for the bounding boxes and labels
    font = cv.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # Draw bounding boxes and labels on the frame
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i], 2))
            color = colors[i]
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(frame, label + " " + confi, (x, y + 20), font, 2, (255, 255, 255), 2)

    # Show the frame with the detections
    cv.imshow("YOLO Object Detection", frame)

    # Press 'q' to exit the real-time detection
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv.destroyAllWindows()