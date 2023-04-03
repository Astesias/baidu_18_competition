import time,os,sys
from pysl import easy_request,getip
from threading import Thread as thd
from multiprocessing import Process as pcs


def pcs_tasker(port):
    from pysl import cmd

    if os.name!='nt':
        cmd(f'python3 manage.py runserver 0.0.0.0:{port}')
    else:
        cmd(f'python manage.py runserver 0.0.0.0:{port}')


def write():
    time.sleep(2)
    i = 0
    while 1:
        time.sleep(1)
        msg = str(i)
        print(msg, '------------')
        easy_request(f'http://{getip()[0]}:1881/data', method='POST',
                     data={'msg': msg},
                     header={"Content-type": "application/json"}
                     )
        i += 1


if __name__ == '__main__':
    if len(sys.argv)==1:
        thd(target=write).start()
        pcs(target=pcs_tasker).start()
    else:
        pcs(target=pcs_tasker,args=sys.argv[1:]).start()
