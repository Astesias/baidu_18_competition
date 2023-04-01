# from multiprocessing import Process as pcs
# from multiprocessing import Queue
# import time

# def pcs_tasker(Qin):

#     from threading import Thread as thd
#     from mysite.views import Q
#     from pysl import cmd
    
#     thd(target=cmd,args=['python manage.py runserver 0.0.0.0:8000']).start()

#     Qin.put(Q)
    


# if __name__=='__main__': 

#     Qin=Queue()
#     djserver=pcs(target=pcs_tasker,args=[Qin])

#     djserver.start()
#     djserver.join()
#     Qout=Qin.get()

#     while 1:
#          print(Qout.qsize())
#          Qout.put(time.time())
#          time.sleep(0.5)

import json,time,random
from manage import main
from threading import Thread as thd

def random_name(n):
    st='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    st=st+st.lower()+'_0123456789'
    r=''
    for i in range(n):
        if i==0:  
            r+=random.choice(st[:-9])
        else:
            r+=random.choice(st)
    return r

def cmd(command,log=False):
    import subprocess
    cmd=subprocess.getstatusoutput(command)
    if log:
        print(('Success' if not cmd[0] else 'Fail') + ' Command:\n   '+command)
        print(cmd[1].replace('Active code page: 65001',''))
    if cmd[0] and not log:
        raise Exception(f'cmd order {command} failed')


def write():
    while 1:
        time.sleep(1)
        with open('tmp.json','w') as fp:
            json.dump(
                {'msg':random_name(5)},
                fp
            )

if __name__=='__main__': 
    thd(target=write).start()
    main()
