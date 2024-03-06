# Tests the YOLO (You Only Look Once) model via ultralytics library.
# Performs real-time object detection, detects and classifies objects.
# Displays their class names, confidence scores, and fps.

from ultralytics import YOLO
import cvzone
import cv2
import math
import time

cap = cv2.VideoCapture(0)  # Default webcam
cap.set(3, 640)  # capture width to 640 pixels
cap.set(4, 480)  # capture height to 480 pixels

confidence = 0.8 # Confidence threshold

# Load the YOLOv8-nano model
model = YOLO("../models/nano_version_14_100.pt")

# Define class names YOLO can detect
classNames = ["fake", "real"]

prev_frame_time = 0  # Store the previous frame time
new_frame_time = 0  # Store the current frame time

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)  # Detect objects in frame, stream=True for video input
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Bounding box dimensions
            w, h = x2 - x1, y2 - y1

            # Confidence score of detection
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Get class name of detected object
            cls = int(box.cls[0])
            if conf > confidence:
                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

            cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
            # Display class name & confidence score
            cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                               (max(0, x1), max(35, y1)), scale=2, thickness=3, colorR=color, colorB=color)


    # Calculate the FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
