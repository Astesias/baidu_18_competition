
# uartdemo.py
from serial import Serial
import time

try:
  while 1:
    uartport = "/dev/ttyPS1"
    bps = 9600
    timeout = 0.5

    serial = Serial(uartport,bps,timeout=timeout)

    result = serial.write("HelloWorld".encode('utf8'))
    print(result)
    time.sleep(1)


except Exception as e:
  serial.close()
  print("----error-----",e)
  