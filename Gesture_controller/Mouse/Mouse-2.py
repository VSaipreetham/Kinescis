import cv2
import time
import autopy
import numpy as np
import HandTracking as ref

#Setting the frame and size of the camera
camWidth, camHeight = 640, 480
redFrame = 100  # Frame Reduction
fastness = 7
#To capture the video frame
picture = cv2.VideoCapture(0)
picture.set(3, camWidth)
picture.set(4, camHeight)
recognizer = ref.HandRecognizer(maxHands=1,detectionCon=0.7)
Screenwidth, Screenheight = autopy.screen.size()
# print(Screenwidth,Screenheight)
TimePresent = 0
locxPrev, locyPrev = 0, 0
locxcurr, locyCurr = 0, 0

status=True

while status:
    # 1. Finding the position Landmarks of Hand
    success, image = picture.read()
    image = recognizer.HandsFinding(image)
    FingersList, recBox = recognizer.GetPosition(image)
    # 2. Get the tip of the thumb and index fingers
    if len(FingersList) != 0:
        x1, y1 = FingersList[4][1:]
        x2, y2 = FingersList[8][1:]
        print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        raisefingers = recognizer.RaiseFingers()
        # print(fingers)
        cv2.rectangle(image, (redFrame, redFrame), (camWidth - redFrame, camHeight - redFrame),(0, 255, 204), 2)
        # 4. Only Index Finger : Moving Mode
        if raisefingers[0] == 1 and raisefingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (redFrame, camWidth - redFrame), (0, Screenwidth))
            y3 = np.interp(y1, (redFrame, camHeight - redFrame), (0, Screenheight))
            # 6. Smoothen Values
            locxcurr = locxPrev + (x3-locxPrev) / fastness
            locyCurr = locyPrev + (y3-locyPrev) / fastness

            # 7. Move Mouse
            autopy.mouse.move(Screenwidth - locxcurr, locyCurr)
            cv2.circle(image, (x1, y1), 15, (252, 74, 3), cv2.FILLED)
            locxPrev, locyPrev = locxcurr, locyCurr

        # 8. If both the index and thumb fingers are up then it is in the Clicking Mode
        if raisefingers[1] == 1 and raisefingers[2] == 1:
            # 9. Find distance between fingers
            length, image, lineInfo = recognizer.GapFinding(4, 8, image)
            print(length)
            # 10. Click mouse if distance short
            if length < 40:
                cv2.circle(image, (lineInfo[4], lineInfo[5]), 15, (255, 233, 11), cv2.FILLED)
                autopy.mouse.click()
    # 11. Caluclating the Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - TimePresent)
    TimePresent = cTime
    cv2.putText(image, str(int(fps)), (20, 50), cv2.FONT_ITALIC, 3, (252, 3, 186), 3)
    # 12. Display
    cv2.imshow("InpImage", image)
    cv2.waitKey(1)