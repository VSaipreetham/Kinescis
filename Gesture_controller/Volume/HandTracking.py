import cv2
import mediapipe as mp
import time
import math


class HandRecognizer():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.maxHands = maxHands


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils



    def HandsFinding(self, img, draw=True):
        imgChange = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.solution = self.hands.process(imgChange)
        # print(results.multi_hand_landmarks)

        if self.solution.multi_hand_landmarks:
            for handLms in self.solution.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


    def GapFinding(self, img, handNo=0, draw=True):
        Fingersvaluelist = []
        if self.solution.multi_hand_landmarks:
            Handp = self.solution.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Handp.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, midx, midy)
                Fingersvaluelist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

        return Fingersvaluelist



    def GetDistances(self, p1, p2, img, draw=True):
        x1, y1 = self.FingersvalueList[p1][1], self.FingersvalueList[p1][2]
        x2, y2 = self.FingersvalueList[p2][1], self.FingersvalueList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (123, 123, 186), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0,0,255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    capture = cv2.VideoCapture(0)
    detector = HandRecognizer()
    while True:
        success, img = capture.read()
        img = detector.HandsFinding(img)
        FingersvalueList = detector.GapFinding(img)
        if len(FingersvalueList)!= 0:
            print(FingersvalueList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (252, 3, 186), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()