import cv2
import sys
import os
import time
from utils import range_percent as R
  
def imagecapture(n,path,vt=10):
    if len(n)==1:
        n='/dev/video'+n[0]
    else:
        print( 'MutiCameraError')
        sys.exit(0)
           
    cap=cv2.VideoCapture(n,cv2.CAP_V4L)  
    print('Write on Camera {} '.format(n))
    
    t=time.time()
    f=vt
    r=R(vt,process_name='Image Writing',ef=0)
    while cap.isOpened():
        ret, frame = cap.read() 
        
        if f==vt:
          print('Image Shape:',frame.shape)
          f-=1
        elif f:
          f-=1
        else:
            break
        cv2.imwrite(os.path.join(path,'{}.jpg'.format('_'.join(time.ctime(time.time()).split(' ')[3:5]),frame)
        r.update(vt-f,new=' p')
        
    cap.release()      
    cv2.destroyAllWindows() 
    print('\nWrite in '+path)

if __name__ == '__main__' :
    os.system('rm /root/workspace/test/output/image/*.jpg')
    imagecapture(['0'],"/root/workspace/test/output/image/",vt=10 if len(sys.argv)==1 else int(sys.argv[1]))
    
    
