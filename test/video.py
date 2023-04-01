import cv2
import sys
import numpy
from utils import SysInput
  
def videocapture(n):
    if len(n)==1:
        n='/dev/video'+n[0]
    else:
        print( 'MutiCameraError')
        sys.exit(0)
    cap=cv2.VideoCapture(n,cv2.CAP_V4L)     
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = cap.get(cv2.CAP_PROP_FPS) 
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  
    #print( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) , cv2.CAP_PROP_FPS, int(cap.get(cv2.CAP_PROP_FOURCC)))  
    #1280 720 5 0 
    #writer = cv2.VideoWriter("video_result.mp4", fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read() 
        frame=cv2.resize(frame,(640,480))
        cv2.imshow('test', frame)
        #print(frame.shape)
        key = cv2.waitKey(24)
        
        #writer.write(frame)  
   
        if key == ord('q'):
            break
    cap.release()        
    cv2.destroyAllWindows() 


if __name__ == '__main__' :
    SysInput(videocapture,'0')
