from pysl import cmd,Config
import serial
import sys
import serial.tools.list_ports
import time,os
from threading import Thread

class Communication():

  def __init__(self,com,bps,timeout):
    self.port = com
    self.bps = bps
    self.timeout =timeout
    global Ret
    try:
       self.main_engine= serial.Serial(self.port,self.bps,timeout=self.timeout)

       if (self.main_engine.is_open):
        Ret = True
    except Exception as e:
      print( e)
    self.Print_Name()
  def Print_Name(self):
    print('Serial Configs:')
    print('    name:',self.main_engine.name)
    print('    port:',self.main_engine.port)
    print('    baudrate:',self.main_engine.baudrate)
    print('    bytesize:',self.main_engine.bytesize)
    print('    parity:',self.main_engine.parity)
    print('    stopbits:',self.main_engine.stopbits)
    print('    timeout:',self.main_engine.timeout)
    print('    writeTimeout:',self.main_engine.writeTimeout)
    print('    xonxoff:',self.main_engine.xonxoff)
    print('    rtscts:',self.main_engine.rtscts)
    print('    dsrdtr:',self.main_engine.dsrdtr)
    print('    interCharTimeout:',self.main_engine.interCharTimeout)

  
  def Open_Engine(self):
    self.main_engine.open()

  
  def Close_Engine(self):
    self.main_engine.close()
    print(self.main_engine.is_open)


  @staticmethod
  def Print_Used_Com():
    port_list = list(serial.tools.list_ports.comports())
    print(port_list)

  def Read_Size(self,size):
    return self.main_engine.read(size=size)

  def Read_Line(self):
    return self.main_engine.readline()


  def Send_data(self,data):
    self.main_engine.write(data)



  def Recive_data(self,way):
   
    print("Rcving")
    while True:
      try:
      
        if self.main_engine.in_waiting:
          if(way == 0):
            for i in range(self.main_engine.in_waiting):
              print("rcv:"+str(self.Read_Size(1)))
              data1 = self.Read_Size(1).hex()
              data2 = int(data1,16)
              if (data2 == "exit"): 
                break
              else:
                 print("rcv"+data1+" rcv"+str(data2))
          if(way == 1):
       
            data = self.main_engine.read_all()
            if (data == "exit"): 
              break
            else:
               print("rcv", data)
      except Exception as e:
        print(e)
        
        
        
cmd('python3 config_make.py')
cfg=Config('./configs.json')
serial_host, serial_bps = cfg['serial_host'], cfg['serial_bps']

t=time.time()
ser = Communication(serial_host, serial_bps,5)
while 1:
    if ser.main_engine.inWaiting():
        msg=str(ser.main_engine.read_all(),'utf8').replace('\r','').replace('\n','\\n').strip(' ')
        print('recv',msg)
        if 'run' in msg:
            Thread(target=cmd,args=['python3 Tasker.py 1'])
        if 'exit' in msg:
            cmd('bash kill_looper.sh')
            
    else:
      time.sleep(1)



    if time.time()-t>480:
        print('timeout')
        break
        
        
        
        
        
        
        
        
        