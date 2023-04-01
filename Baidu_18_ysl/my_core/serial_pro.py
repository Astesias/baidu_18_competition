"""
Created on Tue Apr  5 16:33:51 2022

@author: 鞠秉宸
"""

import serial
import numpy as np
import time


class MySerial(object):
    def __init__(self,dev,bound,_timeout):
        self._dev = dev
        self._bound = bound
        self._timeout = _timeout
        self.Serial = serial.Serial(self._dev,self._bound,timeout=self._timeout)
        
    def read(self):
        #print(str(self.Serial.readline(),'ascii'))
        return str(self.Serial.readline(),'utf-8').replace(" ","").replace("\n","").replace("\r","")
        
    def write(self,data):
        self.Serial.write(data.encode('utf-8'))
    def flushInput(self):
        self.Serial.flushInput() 
    def MWrite(self,data):
        #start = time.time()
        while(True):
            self.write(data)
            ##print("waiting...")
            if(self.read()=="DONE"):
                break
            #print(time.time()-start)
            #if(time.time()-start>1):
             #   break
 
 
def init_serial_port():
    count = 0
    while(True):
        try:
            ser = MySerial('/dev/ttyUSB'+str(count),115200,0.02)
            #print("MySerial open succussed")
            break
        except:
            count = count+1
            if(count==10):
                #print("MySerial open failed")
                return 
                
    #print("Start to Warmup...")
    time.sleep(2)
    count=0
    while(True):
        ser.write("start")
        time.sleep(0.02)
        count = count+1
        if(count==6):
            break
    #print("Warmup finished...")
    return ser
    
    
if(__name__=="__main__"):
    count=0
    while(True):
        try:
            ser = MySerial('/dev/ttyUSB'+str(count),115200,0.2)
            #print("MySerial USB"+str(count)+" open succussed")
            break
        except:
            count = count+1
            if(count==10):
                #print("MySerial open failed")
                pass
    time.sleep(1)
    
    while(True):
        str1="start:start/"
        ser.write(str1)
        time.sleep(0.05)
        strr = ser.read()
        
        #print(strr+":"+str(strr=="DONE"))
        if(strr=="DONE"):
            #print("OK")
            break
'''
    while True:
        str1 = input("input:")
        print(len(str1))
        ser.write(str1)
        strr = ser.read()
        if(strr=="DONE"):
            print("OK")
        print("get:",len(strr))
        #print("get:",str(ser.Serial.readline(),'utf-8'))
        time.sleep(0.05)'''