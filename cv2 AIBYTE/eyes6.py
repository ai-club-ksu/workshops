import mediapipe as mp
import cv2
import time
from math import hypot
import numpy as np

old_time = 0
new_time = time.time()
model_path = '/absolute/path/to/face_landmarker.task'
mp_face = mp.solutions.face_mesh
face =mp_face.FaceMesh(refine_landmarks =True)
mpDraw =mp.solutions.drawing_utils
drawSpac = mpDraw.DrawingSpec(color=(255,255,255),thickness=1,circle_radius=1)
state = 1
cap = cv2.VideoCapture(0)
_,f = cap.read()
h,w,_ = f.shape
dT =0
backgraund = np.zeros((h,w,3),np.uint8)


while cap.isOpened():
    backgraund = np.zeros((h,w,3),np.uint8)
    eyes =[]
    _,frame = cap.read()
    rgbFrame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    process =face.process(rgbFrame)
    if process.multi_face_landmarks:
        for faceLM in process.multi_face_landmarks:
            for id,lm in enumerate(faceLM.landmark):
                h,w,_ = frame.shape
                x,y = int(lm.x*w),int(lm.y*h)
                eyes.append([id,x,y])
            mpDraw.draw_landmarks(backgraund,faceLM,mp_face.FACEMESH_CONTOURS,drawSpac,drawSpac) 

    if eyes !=[]:
        x158,y158 = eyes[158][1],eyes[158][2]
        x160,y160 = eyes[160][1],eyes[160][2]
        x144,y144 = eyes[144][1],eyes[144][2]
        x153,y153 = eyes[153][1],eyes[153][2]
        
        x133,y133 = eyes[133][1],eyes[133][2]
        x33,y33 = eyes[33][1],eyes[33][2]
        
        x387,y387 = eyes[387][1],eyes[387][2]
        x385,y385 = eyes[385][1],eyes[385][2]
        x380,y380 = eyes[380][1],eyes[380][2]
        x373,y373 = eyes[373][1],eyes[373][2]

        x363,y363 = eyes[263][1],eyes[263][2]
        x362,y362 = eyes[362][1],eyes[362][2]
        
        dL1 = hypot(x153-x160,y153-y160)
        dL2 = hypot(x158-x144,y158-y144)
        dL3 = hypot(x133-x33,y133-y33)
        dLF = (dL1 + dL2)/ dL3
        

        
        dR1 = hypot(x385-x373,y385-y373)
        dR2 = hypot(x387-x380,y387-y380)
        dR3 = hypot(x363-x362,y363-y362)
        dRF = (dR1 + dR2)/ dR3

        if dLF < 0.80 or dRF <0.80:
            state = 0
            if old_time ==0:
                old_time = time.time()

        else:
            state =1
            old_time = 0
            dT=0
        if old_time!= 0:   
            dT = time.time() - old_time
        
            if dT >=2:
                cv2.putText(backgraund,"sos",(300,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=6)

        
    
      
    cv2.imshow("frame",backgraund)      
    if cv2.waitKey(1) == ord('q'):
        break
    