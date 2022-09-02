import cv2 as cv
import mediapipe as mp
import time


class DetectHand():
    def __init__(self, mode=False, maxhands=2, complexity=1, detconfidence=0.5, tracconfidence=0.5) -> None:
        self.mode = mode
        self.maxhands = maxhands
        self.complexity = complexity
        self.detconfidence = detconfidence
        self.tracconfidence = tracconfidence
        self.nphand = mp.solutions.hands
        self.hand = self.nphand.Hands(
            self.mode, self.maxhands, self.complexity, self.detconfidence, self.tracconfidence)
        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hand.process(imgRGB)
        if self.result.multi_hand_landmarks:
            # print(result.multi_hand_landmarks)
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(
                            img, handlms, self.nphand.HAND_CONNECTIONS)
                
        return img
    
    def getPositions(self,img,handNo=0):

        lmlist=[]
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            print(myHand)
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])

                #print(id,lm)
                # if id==4 or id==8 or id==12:
                    #     cv.circle(img,(cx,cy),10,(255,255,0),3)
        return lmlist

            

def main():
    cap = cv.VideoCapture(0)
    ptime = 0
    h = DetectHand()
    while True:
        success, img = cap.read()
        img = h.findHands(img)
        h.getPositions(img)
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv.putText(img, str("FPS:"+str(int(fps))), (10, 70),
                   cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv.imshow("HandDetection", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
