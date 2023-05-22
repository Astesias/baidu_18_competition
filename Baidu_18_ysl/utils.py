import os
import cv2
import sys
import time
import subprocess
from functools import wraps
from numpy import sin,cos,pi

class Fplog():
    def __init__(self,filename,ser=None):
        self.fp=open(filename,'w')
        self.filename=filename
        self.ser=ser
    def add(self,message):
        self.fp.write('{} : {}\n'.format(getime(),message))
        if self.ser:
            self.ser.Send_data( (message+'\n').encode('utf8'))
    def close(self):
        self.fp.close()

class Timeit():
    def __init__(self,linesname=''):
        self.info=linesname+':'
        self.line=sys._getframe().f_back.f_lineno+1
        self.t=time.time()

    def out(self,newinfo=None,**kw):
        line=sys._getframe().f_back.f_lineno
        sprint('In file {}| {} line {}-{} takes {:.2f} s'
              .format(get_fname(),self.info,self.line,line-1,time.time()-self.t),**kw)
        if newinfo:
            self.info=newinfo+':'
        self.t=time.time()
        self.line=line+1

class Timer():
    def __init__(self,sep=1,once=False):
        self.start=0 if not once else time.time()
        self.sep=sep
    def T(self):
        if time.time()-self.start>self.sep:
            self.start=time.time()  
            return True
        else:
            return False
        
def SysInput(func,default,**kw):
    sprint(sys.argv,**kw)
    if len(sys.argv)>=1:
        func(sys.argv[1:])
    else:
        func([default])

def get_fname(ori=False):
    if ori:
        return path2filename(__file__)
    return path2filename(sys.argv[0])

def getime():
    t=time.localtime()
    return time.strftime('%Y_%m_%d|%H_%M_%S',t)

def path2filename(path):
    if type(path)!=type('str'):
        raise TypeError('path is a str,not {}'.format(type(path)))
    if path.rfind('\\')>path.rfind('/'):
        return path[path.rfind('\\')+1:]
    else:
        return path[path.rfind('/')+1:]

def Timety(timer=None,**kwo):
    def _timety(func):
        @wraps(func)
        def wrapper(*arg,**kw):
            t=time.time()
            r=func(*arg,**kw)
            if (timer and timer.T()) or not timer:
                #print('\033[3A' if func.__name__=='PredictFrame' else '',end='')
                sprint('function {} take {:.2f}s'.format(func.__name__,(time.time()-t)),**kwo)
            return r
        return wrapper
    return _timety

def sprint(message='',ser=None,logger=None,normal=True,T=0,end='\n',sep=' '):
    if ser:
        ser.Send_data( (message+end).encode('utf8'))
    if logger:
        logger.add('[{:.2f}s] : {}'.format(time.time()-T,message))
    if normal:
        print('[{:.2f}s] : {}'.format(time.time()-T,message),end=end,sep=sep)


def display_angle(frame,angle):
    midx,midy=int(frame.shape[1]/2),int(frame.shape[0]/2)
    p1=int(midx-50*sin(angle/180*pi)),int(midy+50*cos(angle/180*pi))
    p2=int(midx+50*sin(angle/180*pi)),int(midy-50*cos(angle/180*pi))
    
    cv2.line(frame,p2,p1,(255,255,0),4,3)
    cv2.putText(frame,'{:^5.1f}'.format(abs(angle)),(midx+20,midy),
                cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255) if angle>0 else (255,0,255))
    cv2.imshow('SEG',frame)


##############################################  hardware tools

# gpio
from pathlib import Path

def set_gpio(pn,level=True,**kw):
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
        sprint("Error: Enable Port {} Faield.".format(port),**kw)
        sys.exit()

    # gpio set out mode
    os.system("echo out > {}".format(port_direction_path))
    os.system("echo {} > {}".format(level,port_value_path))  # set 

    sprint("{} Port is {} ...".format(pn,'high' if level else 'low'),**kw)

def set_all_gpio(string,**kw):
    assert len(string)==6
    for n,i in enumerate(string):
        if i!='x':
          set_gpio(n+1,int(i),**kw)
    sprint(**kw)

# serial
import serial
import serial.tools.list_ports

class Communication():
    def __init__(self,com,bps,timeout,**kw):
        self.port = com
        self.bps = bps
        self.timeout =timeout

        global Ret
        try:
            self.main_engine= serial.Serial(self.port,self.bps,timeout=self.timeout)
            if (self.main_engine.is_open):
                Ret = True
        except Exception as e:
            print(e)
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
   
        print("Recving Thread Start")
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

def Serial_init(com="/dev/ttyPS1",bps=115200,timeout=0.5,**kw):
    return Communication(com,bps,timeout,**kw)

# useless
class range_percent():
    def __init__(self,total,process_name='Process',obj="#",nonobj=' ',ef=1):
        self.total = total
        self.process_name=process_name
        self.obj=obj
        self.nonobj=nonobj
        self.ef=ef
    def update(self,now,new=''):
        precent=now/self.total
        num=int(100*precent)
        sys.stdout.flush()
        print("\r\r\r", end="")
        print("{} {:>3}% |".format(self.process_name,num),self.obj*(num//3),self.nonobj*(33-num//3),
                '|{}/{}'.format(now,self.total),sep='', end=new)
        if now==self.total and self.ef:
          print()
          self.ef=0
        sys.stdout.flush()

def screenxy(wl=0,hl=0):
    cmd = ['xrandr']
    cmd2 = ['grep', '*']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
    p.stdout.close()
    resolution_string, junk = p2.communicate()
    resolution = resolution_string.split()[0]
    width, height = resolution.split(b'x')
    return int(width)-wl,int(height)-hl

def mmap(func_or_method,ite,arg=[],kw={}):
    r=[]
    if type(arg)!=type([]):
        arg=[arg]
    if type(func_or_method)==type(''):
        for i in ite:
            # print(('i.{}(*{},**{})'.format(func_or_method,arg,kw)))
            r.append(eval('i.{}(*{},**{})'.format(func_or_method,arg,kw)))
        return r
    else:
        for i in ite:
            r.append( func_or_method(i,*arg,**kw) )
        return r

def check_cap(capl,**kw):
    t=time.time()
    while 1:
      count=0
      for cap in capl:
        if cap.isOpened():
          count+=1
      if count==len(capl):
        sprint('Camera open Success',**kw)
        break
      if time.time()-t>10:
          raise IOError('Camera open failed')