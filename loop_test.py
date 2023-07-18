import time,os

t=time.time()
while 1:
    assert os.path.exists('/dev/ttyUSB0')
    print('looping')
    time.sleep(1)
    if time.time()-t>60:
        print('timeout')
        break