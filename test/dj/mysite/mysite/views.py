from django.http import HttpResponse
from django.shortcuts import render
from queue import Queue
import time,json,traceback,os

class Config():
    def __init__(self,path):
        self.path = path
        if os.path.exists(path):
            with open(path) as fp:
                self.data=json.load(fp)
        else:
            self.data={}

    def add(self,item_k,item_v):
        self.data[item_k]=item_v
    
    def save(self,path=None):
        if not path:
            path=self.path
        with open(path,'w') as fp:
            json.dump(self.data,fp,ensure_ascii=False,indent=2)
        
    def __enter__(self):
        return self
    def __exit__(self, type, value, trace):
        self.save()
    def __call__(self,key):
        return self.data[key]
    def __del__(self):
        try:
            self.save()
        except:
            pass

T=None
Q=Queue(maxsize=5)
Q2=Queue(maxsize=2)

def wifi(request):
    return render(request, 'log.html',{})

def data(request):

    if request.method=='POST':
        d=json.loads(request.body)
        Q.put(d['msg'])
        return HttpResponse('200')
    else:
        t=request.path.strip('data/')

        global T 
        if not T:
            T=int(t)
        last=(int(t)-T)/1000

        if Q.qsize():
            msg=Q.get()
    
            res=' [{:.2f}] '.format(last).ljust(10,'x')+f'| {msg: <6}'
            res=res.replace('x','&#8194')

            return HttpResponse(res)
        else:
            return HttpResponse('NoData')

        # try:
        #     with open('tmp.json','r') as fp:
        #         msg=json.load(fp)['msg']
        #     open('tmp.json','w').close()
        #     res=f'[{last:.2f}s]|{msg}'
        #     return HttpResponse(res)
        # except:
        #     with open('log.txt','w') as fp:
        #         fp.write(traceback.format_exc())
        #     return HttpResponse('NoData')


def order(request):

    if request.method=='POST':
        
        d=str(request.body)
        Q2.put(d.strip('\'').split('=')[1])
        return HttpResponse('200')
    else:
        if Q2.qsize():
            msg=Q2.get()
            return HttpResponse(msg)
        else:
            return HttpResponse('NoData')
