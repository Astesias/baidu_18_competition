import time,os
from pysl import cmd


t=time.time()
while 1:
    assert os.path.exists('/dev/ttyUSB0')
    print('looping')
    time.sleep(5)
    cmd('echo 1 > /dev/ttyUSB0')
    if time.time()-t>180:
        print('timeout')
        break