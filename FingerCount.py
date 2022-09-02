from calendar import c
import cv2 as cv
import time
import HandTrack as hm

cap = cv.VideoCapture(0)
ptime = 0
h = hm.DetectHand()
mylist=[8,12,16,20]
while True:
        success, img = cap.read()
        img = h.findHands(img,True)
        pos=h.getPositions(img)
        if len(pos)!=0:
                countlist=[]
                if pos[4][1]>pos[4-1][1]:
                        countlist.append(1)
                else:
                        countlist.append(0)

                for fig in mylist:
                        if(pos[fig][2]<pos[fig-2][2]):
                                countlist.append(1)
                        else:
                                countlist.append(0)
                # print(countlist)
                cv.putText(img,str("Count:"+str(countlist.count(1))),(10,400),cv.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
                
                        

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv.putText(img, str("FPS:"+str(int(fps))), (10, 70),
                   cv.FONT_HERSHEY_PLAIN, 3, (0, 255,255), 3)
        cv.imshow("HandDetection", img)
        cv.waitKey(1)