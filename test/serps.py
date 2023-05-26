
# uartdemo.py
from serial import Serial
import time

try:
  while 1:
    uartport = "/dev/ttyPS0"
    bps = 115200
    timeout = 0.5

    serial = Serial(uartport,bps,timeout=timeout)

    result = serial.write("HelloWorld".encode('utf8'))
    while serial.inWaiting():
        cmd_temp = serial.read()
        print(cmd_temp)
        time.sleep(1)
    time.sleep(1)

except Exception as e:
  serial.close()
  print("----error-----",e)
  