import os
from cvzone.HandTrackingModule import HandDetector
import cv2


width=1200
height=700
foldPath="AI-Review-3"

#Cam config
capture=cv2.VideoCapture(0)
capture.set(3,width)
capture.set(4,height)

#Extract Presentation images
Imagespath=sorted(os.listdir(foldPath),key=len)
#print(Imagespath)


#Var
imgid=0
hs,ws=int(120*1),int(213*1)
btnclicked=False
btncntr=0
btndelay=25

recognizer=HandDetector(detectionCon=0.8,maxHands=1)


while True:
    success,pic=capture.read()
    pic=cv2.flip(pic,1)
    FullImgPath=os.path.join(foldPath,Imagespath[imgid])

    currImg=cv2.imread(FullImgPath)

    #Adding Image of webcam on PPT
    imgSmall=cv2.resize(pic,(ws,hs))
    h, w,_=currImg.shape

    #print(h,w)
    currImg[0:hs,w-ws:w] = imgSmall
    hands,pic=recognizer.findHands(pic)
    #300 is max gesture size
    cv2.line(pic,(0,300),(width,300),(225,225,19),5)

    if hands and btnclicked is False:
        hand=hands[0]
        fingers=recognizer.fingersUp(hand)
        #print(fingers)
        cx,cy=hand['center']


        if cy<300:        #Height of the face is hand handThreshold
            #Gest 1:-left slide
            if fingers == [1,0,0,0,0]:
                print("Left Side")
                if imgid>0:
                    btnclicked = True
                    imgid-=1

            if fingers == [0,1,0,0,0]:
                print("Right Side")
                if imgid<len(Imagespath)-1:
                    btnclicked = True
                    imgid+=1

    #Itr of Btn clicked
    if btnclicked:
        btncntr+=1
        if btncntr>btndelay:
            btncntr=0
            btnclicked=False


    cv2.imshow("Image",pic)
    cv2.imshow("Slides", currImg)

    close=cv2.waitKey(1)
    if close==ord('q'):
        break
