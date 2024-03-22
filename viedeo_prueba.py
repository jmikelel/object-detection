import cv2
import numpy as np


dir=r"C:\Users\Josem\Documents\programacion\code\football-field-cropped-video.mp4"
cap=cv2.VideoCapture(dir)

backgroundObject=cv2.createBackgroundSubtractorMOG2(history=2)
kernel=np.ones((3,3),np.uint8)
kernel2=None

while True:
    ret, frame=cap.read()
    if not ret:
        break

    fgmask=backgroundObject.apply(frame)
    _, fgmask=cv2.threshold(fgmask,20,255,cv2.THRESH_BINARY)
    fgmask=cv2.erode(fgmask,kernel,iterations=1)
    fgmask=cv2.dilate(fgmask,kernel2,iterations=6)


    countors,_=cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    frameCopy=frame.copy()

    for cnt in countors:
        if cv2.contourArea(cnt)>20000:
            x,y,width,height=cv2.boundingRect(cnt)

        cv2.rectangle(frameCopy,(x,y),(x+width,y+height),(0,0,255),2)

        cv2.putText(frameCopy,"Dr Andres",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),1,cv2.LINE_AA)
    
    forground=cv2.bitwise_and(frame,frame,mask=fgmask)

    stacked=np.hstack((frame,forground,frameCopy))
    #cv2.imshow("stacked",cv2.resize(stacked,None,fx=0.1,fy=0.1))


    #cv2.imshow("forground",forground)
    cv2.imshow("framecopy",cv2.resize(frameCopy,None,fx=0.3,fy=0.3))
    #cv2.imshow("fgmask",fgmask)
    #cv2.imshow("img",frame)


    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()