import logging
import sys
import subprocess
import traceback
class ezlog():
    def __init__(self,filename):
        logging.basicConfig(
                            filename=filename,
                            format='%(asctime)s %(module)s.py line %(lineno)d: %(message)s',
                            datefmt='%Y/%m/%d|%H:%M:%S',
                            level=logging.INFO
                            )
        self.filename=filename
    def add(self,message):
        logging.critical(message)  
    def err(self):
        logging.error(traceback.format_exc())
    def flush(self):
        f=open(self.filename,'w')
        f.close()

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
        print("{} {:>3}% |".format(self.process_name,num),self.obj*(num//3),self.nonobj*(33-num//3),'|{}/{}'.format(now,self.total),sep='', end=new)
        if now==self.total and self.ef:
          print()
          self.ef=0
        sys.stdout.flush()
        
     
class timeit():
    def __init__(self,linesname=''):
        self.info=linesname+':'
        self.line=sys._getframe().f_back.f_lineno+1
        self.t=time.time()
    def out(self,newinfo=None):
        line=sys._getframe().f_back.f_lineno
        print('In file {}| {} line {}-{} takes {} s'
              .format(get_fname(),self.info,self.line,line-1,time.time()-self.t))
        if newinfo:
            self.info=newinfo+':'
        self.t=time.time()
        self.line=line+1
        
        
def SysInput(func,default):
    import sys
    print(sys.argv)
    if len(sys.argv)>=2:
        func(sys.argv[1:])
    else:
        func([default])
        

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