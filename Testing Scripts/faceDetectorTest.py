# Test the real-time face detection using OpenCV and cvzone library.
# OpenCV for image processing and display.
# cvzone for the face detection.

from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import cv2


# Initialize the webcam
# 0 = Default camera, 1 = first external camera connected
cap = cv2.VideoCapture(0)

# minDetectionCon: Minimum detection confidence threshold
# modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        for bbox in bboxs:
            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

            # ---- Draw Data  ---- #
            # Draw a circle at the center of the face
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            # Display the confidence score
            cvzone.putTextRect(img, f'{score}%', (x, y - 10))
            # Draw a rectangle around the face
            cvzone.cornerRect(img, (x, y, w, h))

    cv2.imshow("Image", img)
    cv2.waitKey(1)

