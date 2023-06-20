from multiprocessing import Process as pcs
from multiprocessing import Queue 
from pysl import Config,os_enter,easy_request,cmd
import os,config_make,time,cv2,sys
import numpy as np

Q_Order=Queue(maxsize=5)
config_make.make_cfg()

cfg=Config('./configs.json')
server,port=cfg.server,cfg.port

def post_data(data,server=server):
    easy_request(server+'/data/',method='POST',data={'msg':data},
                     header={"Content-type": "application/json"})

def server_tasker(server,port):

    print(f'Django server on {server}')

    with os_enter('./') as oe:
        oe.cd('test/dj/mysite')
        if os.name=='nt':
            oe.cmd(f'python manage.py runserver 0.0.0.0:{port}')
        else:
            oe.cmd(f'python3 manage.py runserver 0.0.0.0:{port}')

def linear_regression(x,y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
    A = np.mat([[N,sumx],[sumx,sumx2]])
    b = np.array([sumy,sumxy])

    return np.linalg.solve(A,b)

def core(img,resize=None,mask=None):
    # img = cv2.imread(file)
    img=img[10:-10,::,:]
    # b, g, r = cv2.split(img)
    # img = cv2.merge([b,g,r])
    
    if resize:
        if not isinstance(resize,(list,tuple)):
            img=cv2.resize(img,(resize,resize))
        else:
            img=cv2.resize(img,tuple(resize))
            
    if not mask:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([60, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.medianBlur(mask, 5) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) 
    
    mymask=np.zeros_like(mask)
    areas=np.array([])
    lrs=np.array([])

    s= cv2.findContours(mask, cv2.RETR_TREE|cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if os.name!='nt':
      contours=s[1]
    else:
      contours=s[0]
    
    
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > 50:
            
            cnt=np.squeeze(cnt,axis=1)
            mymask[cnt[:,1],cnt[:,0]]=255
            lr=linear_regression(cnt[:,1],cnt[:,0])[1]
            lr=-np.arctan(lr)*180/np.pi
            
            lrs=np.append(lrs,lr)
            areas=np.append(areas,area)
         
        
    k=areas/sum(areas)
    return sum(lrs*k),mymask




def get_order(Q_Order,server):
    time.sleep(2)
    while 1:
        time.sleep(1)
        order=easy_request(server+'/order/')
        if order!='NoData':
            Q_Order.put(order)
            if Q_Order.qsize()==5:
                Q_Order.get()
            print(order,'-----------------')

def main_tasker(Q_Order):
    import time 
    time.sleep(6)
    import cv2
    
    while 1:
        vdir='/dev/video0'
        cap=cv2.VideoCapture(vdir,cv2.CAP_V4L)     

        while cap.isOpened():
            _, frame = cap.read() 
            
            if Q_Order.qsize():
                order=Q_Order.get()
                if order=='shot':
                    n+=1
                    path=f'./output/{pathname(n)}'
                    
                    angle,mask=core(frame)
                    frame=mask
                    print('angle=',angle)
                    post_data(f'D{angle}')
                    
                    framewrite(frame,path)
                    print(f'frame saved in {path}')
                    framewrite(frame,'test/dj/mysite/static/img/tmp.jpg')
                elif order=='shot_off':
                    break
        cap.release()      

        
def framewrite(frame,path):
    cv2.imwrite(path,frame)

def pathname(n):
    return f'{n}.jpg'



if __name__=='__main__':
    pcs(target=server_tasker,args=[server,port]).start()
    pcs(target=get_order,args=[Q_Order,server]).start()
    pcs(target=main_tasker,args=[Q_Order]).start()
    
