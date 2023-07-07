from multiprocessing import Process as pcs
from multiprocessing import Queue 
from pysl import Config,os_enter,easy_request,cmd
import os,config_make,time,cv2,sys

Q_Order=Queue(maxsize=5)
config_make.make_cfg()

cfg=Config('./configs.json')
server,port=cfg.server,cfg.port

def server_tasker(server,port):

    print(f'Django server on {server}')

    with os_enter('./') as oe:
        oe.cd('test/dj/mysite')
        if os.name=='nt':
            oe.cmd(f'python manage.py runserver 0.0.0.0:{port}')
        else:
            oe.cmd(f'python3 manage.py runserver 0.0.0.0:{port}')

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

def main_tasker(Q_Order,class_name=''):
    import time 
    time.sleep(6)
    import cv2
    
    n = 0
    while 1:
      if f'{pathname(n,class_name)}' not in os.listdir('output'):
        break
      else:
        n+=1
    
    while 1:
        vdir='/dev/video'+ ('0' if len(sys.argv)<3 else sys.argv[2])
        cap=cv2.VideoCapture(vdir,cv2.CAP_V4L)     

        while cap.isOpened():
            _, frame = cap.read() 
            
            if Q_Order.qsize():
                order=Q_Order.get()
                if order=='shot':
                    n+=1
                    path=f'./output/{pathname(n,class_name)}'
                    framewrite(frame,path)
                    print(f'frame saved in {path}')
                    
                    framewrite(frame,'test/dj/mysite/static/img/tmp.jpg')
                elif order=='shot_del':
                    try:
                        os.remove(path)
                        n-=1
                        print(f'delete {path}')
                    except:
                        print('Nothing to delete')
                elif order=='shot_off':
                    break
        cap.release()      
        # if Q_Order.qsize():
        #     order=Q_Order.get()
        #     if order=='shot':
        #         path=f'./output/{cnt}.jpg'
        #         # framewrite(frame,path)
        #         print(f'frame saved in {path}')
        #         cnt+=1

        
def framewrite(frame,path):
    cv2.imwrite(path,frame)

def pathname(n,class_name):
  if class_name:
    return f'{class_name}_{n:0>5}.jpg'
  else:
    return f'{n:0>5}.jpg'


if __name__=='__main__':
    if len(sys.argv)>=2:
      class_name=sys.argv[1]
    else:
      class_name=''
    pcs(target=server_tasker,args=[server,port]).start()
    pcs(target=get_order,args=[Q_Order,server]).start()
    pcs(target=main_tasker,args=[Q_Order,class_name]).start()
    
