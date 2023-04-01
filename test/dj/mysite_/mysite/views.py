from django.http import HttpResponse
from django.shortcuts import render
from multiprocessing import Queue
import time,json,traceback

T=None
Q=Queue(maxsize=5)

def wifi(request):
    context          = {}
    context['hello'] = 'WIFI!'
    return render(request, 'log.html', context)

def data(request):

    t=request.path.strip('data/')

    global T 
    if not T:
        T=int(t)

    last=(int(t)-T)/1000

    # global Q
    
    # if Q.qsize():
    #     msg=Q.get()
    #     res=f'[{last:.2f}s]|{msg}'
    #     return HttpResponse(res)
    # else:
    #     return HttpResponse('NoData')

    try:
        with open('tmp.json','r') as fp:
            msg=json.load(fp)['msg']
        open('tmp.json','w').close()
        res=f'[{last:.2f}s]|{msg}'
        return HttpResponse(res)
    except:
        with open('log.txt','w') as fp:
            fp.write(traceback.format_exc())
        return HttpResponse('NoData')



    

