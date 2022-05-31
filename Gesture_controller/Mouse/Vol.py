import cv2
import time
import numpy as np

import HandTracking as ref
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
#volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-20.0, None)



wCam, hCam = 1280, 720
picture = cv2.VideoCapture(0)
picture.set(3, wCam)
picture.set(4, hCam)
pTime = 0

detector = ref.HandRecognizer(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume.__iid__, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange() #[-65,0.03]
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
while True:
    success, img = picture.read()
    img = detector.HandsFinding(img)
    FingersvalueList = detector.GapFinding(img, draw=False)
    if len(FingersvalueList) != 0:
        print(FingersvalueList[4], FingersvalueList[8])

        x1, y1 = FingersvalueList[4][1], FingersvalueList[4][2]
        x2, y2 = FingersvalueList[8][1], FingersvalueList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255,234,0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(gapLength)

        # Hand range 50 - 300
        # Volume Range -65 - 0

        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(f'Length of line is: {int(length)} ,',f'Volume Value is : {vol}')
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (255,0,0), cv2.FILLED)
        if length > 290:
            cv2.circle(img, (cx, cy), 15, (255,0,0), cv2.FILLED)


    cv2.rectangle(img, (50, 150), (80, 400), (0,0,245), 3)
    cv2.rectangle(img, (50, int(volBar)), (80, 400), (0,210,245), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (45, 450), cv2.FONT_ITALIC,1, (242,132,143), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'Tps: {int(fps)}', (40, 50), cv2.FONT_ITALIC, 1, (134,265,178), 3)

    cv2.imshow("Finger Detection", img)
    cv2.waitKey(1)