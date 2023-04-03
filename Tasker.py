from multiprocessing import Process as pcs
from multiprocessing import Queue 
from pysl import Config,os_enter,easy_request
import os,config_make,time

Q_Order=Queue(maxsize=5)
config_make.make_cfg()

cfg=Config('./configs.json')
server,port=cfg.server,cfg.port

def server_tasker(server,port):

    print(f'Django server on {server}')

    with os_enter() as oe:
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

def main_tasker(Q_Order):
    with os_enter() as oe:
        oe.cd('Badui_18_ysl')
        



if __name__=='__main__':
    pcs(target=server_tasker,args=[server,port]).start()
    pcs(target=get_order,args=[Q_Order,server]).start()
    pcs(target=main_tasker,args=[Q_Order]).start()
    
