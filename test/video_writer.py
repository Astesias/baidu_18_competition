import cv2
import sys
import numpy
import time
from utils import range_percent as R
  
def videocapture(n,path,vt=10):
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
    
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    #print( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) , cv2.CAP_PROP_FPS, int(cap.get(cv2.CAP_PROP_FOURCC)))  
    #1280 720 5 0 
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    print('Write on Camera {} '.format(n))
    
    t=time.time()
    f=1
    lefts=[]
    r=R(vt,process_name='Video Writing',ef=0)
    while cap.isOpened():
        ret, frame = cap.read() 
        if f:
          f-=1
          print('Video Shape:',frame.shape)
          
        #frame=cv2.resize(frame,(640,480))
        #cv2.imshow('test', frame)
        
        #left=vt-round(time.time()-t)
        #print( ('Video Writing Left: '+str(left)+'s\n') if left not in lefts else '',end='')
        #if left not in lefts:
         # lefts.append(left)
         
        r.update(round(time.time()-t),new=' s')
        
        writer.write(frame)  
        if time.time()-t>vt:
          break
    
    cap.release()      
    writer.release()  
    cv2.destroyAllWindows() 
    print('\nWrite in '+path)

if __name__ == '__main__' :
    videocapture(['0'] if len(sys.argv)==1 else sys.argv[1:],"/root/workspace/test/output/video/video_save.mp4")
    
