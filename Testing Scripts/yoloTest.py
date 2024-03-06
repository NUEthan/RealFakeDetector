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

# Load the YOLOv8-nano model
model = YOLO("../models/yolov8n.pt")

# Define class names YOLO can detect
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair dryer", "toothbrush"
              ]

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
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence score of detection
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Get class name of detected object
            cls = int(box.cls[0])

            # Display class name & confidence score
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)

    # Calculate the FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
