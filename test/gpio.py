import os
import sys
import time
import argparse
from utils import SysInput
from pathlib import Path

def set_gpio(pn,level=True):
    # param init
    port = pn+8+329+78+4
    level=1 if level else 0
    port_path = "/sys/class/gpio/gpio{}".format(port)
    port_direction_path = "{}/direction".format(port_path)
    port_value_path = "{}/value".format(port_path)

    # enable gpio port
    if Path(port_path).is_dir() is False:
        os.system("echo {} > /sys/class/gpio/export".format(port))

    # check enable success
    if Path(port_path).is_dir() is False:
        print("Error: Enable Port {} Faield.".format(port))
        sys.exit()

    # gpio set out mode
    os.system("echo out > {}".format(port_direction_path))
    os.system("echo {} > {}".format(level,port_value_path))  # set 

    print("{} Port is {} ...".format(pn,'high' if level else 'low'))

def set_all(string):
    if isinstance(string,list):
        string=string[0]
    assert len(string)==6
    for n,i in enumerate(string):
        if i!='x':
          set_gpio(n+1,int(i))


if __name__=='__main__':
    SysInput(set_all,'111111')    # 1:high 0:low x:pass