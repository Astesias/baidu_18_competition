import cv2
import sys
import numpy as np
import time
from utils import SysInput,screenxy



def videocapture(n):

    if len(n)==1:
      hi,wi=screenxy()  
    elif len(n)==2:
      hi,wi=1000,750 
    elif len(n)==3: 
       hi,wi=310,250
    
    ho,wo=list(map(lambda x:int((x)/2),screenxy(60,0)))
    n=list(map(lambda x:'/dev/video'+x,n))
    print(n)
    capl=[]
    for i in n:
        c=cv2.VideoCapture(i,cv2.CAP_V4L)
        #c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) 
        c.set(cv2.CAP_PROP_FRAME_WIDTH, hi)                         
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, wi)
        capl.append(c)
    

    while 1:
      count=0
      for cap in capl:
        if cap.isOpened():
          count+=1
      if count==len(capl):
        break

    frames=[]
    rate=0.8
    while 1:
        key = cv2.waitKey(24)
        if key == ord('q'):
            break
        for c in capl:
            _, frame = c.read() 
            
            reshape=(ho * (2 if len(n)==1 else 1),wo)
            if len(n)==3:
              reshape=tuple(map(lambda x:int(x*rate),reshape))
              
            frame=cv2.resize(frame,reshape)
            frames.append(frame)
        #print(len(frames),frames[0].shape,frames[1].shape)
        
        if len(frames)==3:
          frames.append(np.zeros_like(frame))
        
        try:
          frame=np.concatenate(frames[:2],axis=1)
          frame=np.concatenate([frame,np.concatenate(frames[2:],axis=1)],axis=0)
        except Exception as e:
          pass
        
        
        cv2.imshow('test', frame)
        frames=[]
               
    for cap in capl:
        cap.release()        
    cv2.destroyAllWindows()  


if __name__ == '__main__' :
    SysInput(videocapture,'0')

      
      
      
      
      
