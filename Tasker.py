from multiprocessing import Process as pcs
from multiprocessing import Queue 
from pysl import Config,os_enter,easy_request
import os,config_make,time,sys
from pprint import pprint

Q_Order=Queue(maxsize=5)
try:
    config_make.make_cfg()
except:
    Warning('configs make failed')
cfg=Config('./configs.json')

pprint(cfg.data)
server,port=cfg.server,cfg.port

def server_tasker(server,port):

    print(f'\nDjango server on {server}\n')

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
    from main_pcs import run
    task=[]  # 12 label 10 trade
    print('Note: Start with Tasker.py , main project use main_pcs.py')
    print('Note: Tasker with task',(*task,) if task else 'None')
    run(Q_Order,cfg,True,*task)

def data_poster(server):
    n=1
    time.sleep(6)
    while 1:
        time.sleep(1)
        easy_request(server+'/data/',method='POST',data={'msg':('D' if n%2==0 else 'S')+str(n)},
                     header={"Content-type": "application/json"})
        n+=1

def fin_msg(Q_Order):
    while 1:
        time.sleep(1)
        with open('MSG.txt','r') as fp:
            msg=fp.read()
        if msg:
            open('MSG.txt','w')
            Q_Order.put(msg)
        


if __name__=='__main__':
    usb_flag=False
    for i in os.listdir('/dev'):
        if 'USB' in i:
            usb_flag=True
    assert usb_flag,'Error , TTYUSB not found'
    if len(sys.argv)==2 and int(sys.argv[1])==1:
        pcs(target=server_tasker,args=[server,port]).start()
        pcs(target=get_order,args=[Q_Order,server]).start()
        
    if len(sys.argv)==2 and int(sys.argv[1])==2:
        pcs(target=fin_msg,args=[Q_Order]).start()

    if os.name!='nt':
        pcs(target=main_tasker,args=[Q_Order,cfg]).start()
    else:
        pcs(target=data_poster,args=[server]).start()
        
        
        
        
        
        
        
