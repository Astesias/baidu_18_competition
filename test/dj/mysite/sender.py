import time,os
from pysl import easy_request,getip
from threading import Thread as thd
from multiprocessing import Process as pcs


def pcs_tasker():
    from pysl import cmd
    
    print(f'runserver in: http://{getip()[0]}:1881')
    if os.name!='nt':
        cmd('python3 manage.py runserver 0.0.0.0:1881')
    else:
        cmd('python manage.py runserver 0.0.0.0:1881')


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
    thd(target=write).start()
    pcs(target=pcs_tasker).start()
