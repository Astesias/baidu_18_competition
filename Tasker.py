from multiprocessing import Process as pcs
from multiprocessing import Queue 
from pysl import Config,os_enter,easy_request
import os,config_make,time,sys
from pprint import pprint

Q_Order=Queue(maxsize=5)
config_make.make_cfg()

cfg=Config('./configs.json')
pprint(cfg.data)
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
    time.sleep(6)
    while 1:
        time.sleep(1)
        order=easy_request(server+'/order/')
        if order!='NoData':
            print(order,'-----------------')
            Q_Order.put(order)
            #if order=='exit':
            #    return
            if Q_Order.qsize()==5:
                Q_Order.get()
                print('Warning: order num max')

def main_tasker(Q_Order,cfg):
    os.chdir('./Baidu_18_ysl')
    sys.path.insert(0,'./')
    from main import run
    run(Q_Order,cfg)



if __name__=='__main__':
    pcs(target=server_tasker,args=[server,port]).start()
    pcs(target=get_order,args=[Q_Order,server]).start()
    pcs(target=main_tasker,args=[Q_Order,cfg]).start()
    
