from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time

###################################################
classID = 1  # 0 is fake, 1 is real
outputFolderPath = 'Dataset/CollectData'
blurThreshold = 35  # Larger value --> more focused
confidence = 0.8
save = True

offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6  # decimal points for normalization
debug = False
###################################################


# Initialize the webcam[0 = Def, 1 = first cam connected]
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  # True/False if face is blurred or not
    listInfo = []  # Normalized values & class name to send to txt file

    if bboxs:
        for bbox in bboxs:
            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

            # ---- Confidence Level ---- #
            if score > confidence:
                # ---- Offset Width ---- #
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)  # x is left side, so minus
                w = int(w + offsetW * 2)

                # ---- Offset Height ---- #
                offsetH = (offsetPercentageW / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                # ---- Avoid below 0 ---- #   // avoid corrupt data error
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # ---- Find Blurriness ---- #
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F) .var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # ---- Normalize Values ---- #
                ih, iw, _ = img.shape
                xc, yc = x + w/2, y + h/2  # center points
                xcn = round(xc / iw, floatingPoint)
                ycn = round(yc / ih, floatingPoint)
                # print(xcn, ycn)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                # print(xcn, ycn, wn, hn)

                # ---- Avoid above 1 ---- #   // avoid corrupt data error
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ---- Draw Data ---- #
                cvzone.cornerRect(imgOut, (x, y, w, h))
                cvzone.putTextRect(imgOut, f'Score: {score}% Blur: {blurValue}', (x, y - 20),
                                   scale=2, thickness=3)
                if debug:
                    cvzone.cornerRect(img, (x, y, w, h))
                    cvzone.putTextRect(img, f'Score: {score}% Blur: {blurValue}', (x, y - 20),
                                       scale=2, thickness=3)

                # ---- Save Data ---- #
                if save:
                    if all(listBlur) and listBlur!=[]:
                        # ---- Save Image ---- #
                        timeNow = time()
                        timeNow = str(timeNow).split(".")
                        timeNow = timeNow[0] + timeNow[1]
                        print(time())
                        cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                        # ---- Save Label Text File ---- #
                        for info in listInfo:
                            f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                            f.write(info)
                            f.close()

    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)

