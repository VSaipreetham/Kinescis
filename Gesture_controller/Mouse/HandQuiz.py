import cv2
import csv
from cvzone.HandTrackingModule import HandDetector
import cvzone
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.5)


class Quiz():

    def __init__(self, data):
        self.question = data[0]
        self.option1 = data[1]
        self.option2 = data[2]
        self.option3 = data[3]
        self.option4 = data[4]
        self.result = int(data[5])

        self.userAns = None

    def update(self, cursor, bboxs):

        for x, bbox in enumerate(bboxs):
            x1, y1, x2, y2 = bbox
            if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                self.userAns = x + 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), cv2.FILLED)


# Import csv file Info
pathDir = "Quiz.csv"
with open(pathDir, newline='\n') as f:
    reader = csv.reader(f)
    dataAll = list(reader)[1:]

# Create Object for each Quiz
QnsList = []
for q in dataAll:
    QnsList.append(Quiz(q))

print("Total Quiz Objects Created:", len(QnsList))

Qno = 0
Totalqns = len(dataAll)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if Qno < Totalqns:
        mcq = QnsList[Qno]

        img, bbox = cvzone.putTextRect(img, mcq.question, [100, 100], 2, 2,(255,255,255),(255,0,0),offset=40, border=5)
        img, bbox1 = cvzone.putTextRect(img, mcq.option1, [100, 250], 2, 2,(2,0,0),(255,0,0), offset=40, border=5)
        img, bbox2 = cvzone.putTextRect(img, mcq.option2, [400, 250], 2, 2,(2,0,0),(255,0,0), offset=40, border=5)
        img, bbox3 = cvzone.putTextRect(img, mcq.option3, [100, 400], 2, 2,(2,0,0),(255,0,0), offset=40, border=5)
        img, bbox4 = cvzone.putTextRect(img, mcq.option4, [400, 400], 2, 2,(2,0,0),(255,0,0), offset=40, border=5)


        if hands:
            lmList = hands[0]['lmList']
            cursor = lmList[8]
            length, info = detector.findDistance(lmList[8], lmList[12])
            print(length)
            if length < 35:
                mcq.update(cursor, [bbox1, bbox2, bbox3, bbox4])
                if mcq.userAns is not None:
                    time.sleep(0.3)
                    Qno += 1
    else:
        score = 0
        for mcq in QnsList:
            if mcq.result == mcq.userAns:
                score += 1
        score = round((score / Totalqns) * 100, 2)
        img, _ = cvzone.putTextRect(img, "Quiz Completed", [250, 150], 2, 2,(255,255,255),(242,134,0), offset=50, border=5)
        img, _ = cvzone.putTextRect(img, f'Your Score: {score}%', [320, 300], 2, 2,(255,255,255),(212,0,123), offset=50, border=5)

    # Draw Progress Bar
    barValue = 150 + (950 // Totalqns) * Qno
    cv2.rectangle(img, (150, 600), (barValue, 650), (123, 123, 5), cv2.FILLED)
    cv2.rectangle(img, (150, 600), (1100, 650), (123, 123, 5), 5)
    img, _ = cvzone.putTextRect(img, f'{round((Qno / Totalqns) * 100)}%', [10,20], 2,2,(255,255,255),(255,0,0),offset=16)

    cv2.imshow("Img", img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break