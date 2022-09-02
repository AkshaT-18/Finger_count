import cv2 as cv
import mediapipe as mp
import time

nphand=mp.solutions.hands
hand=nphand.Hands()
mpdraw=mp.solutions.drawing_utils
cap = cv.VideoCapture(0)
ptime=0

while True:
    success,img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    result=hand.process(imgRGB)
    
    if result.multi_hand_landmarks:
        # print(result.multi_hand_landmarks)
        for handlms in result.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                if id==4 or id==8 or id==12:
                    cv.circle(img,(cx,cy),10,(255,255,0),3)
                mpdraw.draw_landmarks(img,handlms,nphand.HAND_CONNECTIONS)
        
    
            

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
    cv.imshow("HandDetection",img)
    cv.waitKey(1)