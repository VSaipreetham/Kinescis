import cv2
import mediapipe as mp
import time
import math
import numpy as np


class HandRecognizer():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.detectionCon, self.trackCon)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.tipIds = [4, 8, 12, 16, 20]

    def HandsFinding(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.solutions = self.hands.process(imgRGB)
        print(self.solutions.multi_hand_landmarks)

        if self.solutions.multi_hand_landmarks:
            for handLms in self.solutions.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(image, handLms, self.mp_holistic.HAND_CONNECTIONS,
                                                   self.mp_drawing.DrawingSpec(color=(244,3, 252), thickness=2,circle_radius=2),
                                                   self.mp_drawing.DrawingSpec(color=(244, 3, 252), thickness=2,circle_radius=2)
                                                   )

        return image

    def GetPosition(self, image, handNo=0, draw=True):
        ListX = []
        ListY = []
        rectBox = []
        self.FingersList = []
        if self.solutions.multi_hand_landmarks:
            myHand = self.solutions.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                ListX.append(cx)
                ListY.append(cy)
                # print(id, cx, cy)
                self.FingersList.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            xmin, xmax = min(ListX), max(ListX)
            ymin, ymax = min(ListY), max(ListY)
            rectBox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(image, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (219, 252, 3), 2)

        return self.FingersList, rectBox

    def RaiseFingers(self):
        raisefingers = []
        # Thumb
        if self.FingersList[self.tipIds[0]][1] < self.FingersList[self.tipIds[0] - 1][1]:
            raisefingers.append(1)
        else:
            raisefingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.FingersList[self.tipIds[id]][2] < self.FingersList[self.tipIds[id] - 2][2]:
                raisefingers.append(1)
            else:
                raisefingers.append(0)

        #totalFingers = raisefingers.count(1)

        return raisefingers

    def GapFinding(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.FingersList[p1][1:]
        x2, y2 = self.FingersList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (123, 123, 186), t)
            cv2.circle(img, (x1, y1), r, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 255, 0), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    picture = cv2.VideoCapture(0)
    recogniser = HandRecognizer()
    while True:
        success, image = picture.read()
        image = recogniser.HandsFinding(image)
        FingersList, bbox = recogniser.GetPosition(image)
        if len(FingersList) != 0:
            print(FingersList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (252, 3, 186), 3)

        cv2.imshow("InpImage", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()