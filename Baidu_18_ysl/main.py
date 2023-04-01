import os
import cv2
import time
import traceback
from utils import Serial_init,Fplog
from utils import Timeit,Timety,Timer
from utils import getime,sprint,set_all_gpio,mmap,check_cap
from detection import detection_init,predict,drawResults

h,w=128,128 # read json and assrt 32
ser=Serial_init("/dev/ttyPS0",115200,0.5)
log_dir='log/'+getime()
os.mkdir(log_dir)
logger_gpio=    Fplog(os.path.join(log_dir,'gpio.txt'),ser=None)
logger_results= Fplog(os.path.join(log_dir,'results.txt'),ser=ser)
logger_looptime=Fplog(os.path.join(log_dir,'looptime.txt'),ser=None)
logger_modelrun=Fplog(os.path.join(log_dir,'modelrun.txt'),ser=None)

try:
    T=time.time()
    timeit=Timeit('Initialization')
    
    cap1 = cv2.VideoCapture('/dev/video0',cv2.CAP_V4L) # front camera
    cap2 = cv2.VideoCapture('/dev/video1',cv2.CAP_V4L) # left  camera
    cap3 = cv2.VideoCapture('/dev/video2',cv2.CAP_V4L) # right camera
    caplist=[cap1,cap2,cap3]
    mmap('set',caplist,arg=[cv2.CAP_PROP_FRAME_WIDTH, w])
    mmap('set',caplist,arg=[cv2.CAP_PROP_FRAME_HEIGHT,h])

    global MODEL_CONFIG,PREDICTOR,DISPLAYER
    DISPLAYER,MODEL_CONFIG,PREDICTOR=detection_init("../test/face_model/usb_yolov3.json")
    classes=MODEL_CONFIG.labels

    @Timety(timer=None,ser=None,logger=logger_modelrun,T=T)
    def PredictFrame(cap,display=True):
        _,frame=cap.read()
        if not _:
            raise IOError('Device bandwidth beyond')
        result = predict(frame,MODEL_CONFIG,PREDICTOR)
        if display:
            drawResults(frame,result,MODEL_CONFIG)
            DISPLAYER.putFrame(frame)
        return mmap('unpack',result,arg=[MODEL_CONFIG.labels])

    @Timety(timer=None,ser=None,logger=logger_modelrun,T=T)
    def SegmentationRoad(cap):
        # TODO
        return 'line_info'

    timeit.out('Mainloop',logger=logger_modelrun,T=T)

    timer_predict=Timer(0.05)# 0.12
    timer_loop=Timer(0.01)# 0.12

    _=1
    switch=True
    loop_times=0 
    result_l=result_r=[]
    check_cap(caplist,T=T)
    while True:
    
        t=time.time()
        loop_times+=1
        if not _:
          line_info=SegmentationRoad(cap1)
          sprint(line_info,T=T,ser=ser,logger=logger_results)

        if timer_predict.T():   

            if switch:
                result_l=PredictFrame(cap2)
            else:
                result_r=PredictFrame(cap3)

            if _:
                sprint('start',T=T,ser=ser,logger=logger_results)
                _=0
            else:
                sprint('result_l: {} ==== result_r: {}'.format(result_l,result_r),
                        T=T,ser=ser,logger=logger_results)

            #switch=not switch
            #set_all_gpio('111111',normal=False,ser=None,logger=logger_gpio) # low spped
            
        if T-time.time()>(8*60):
            raise TimeoutError
        
        if timer_loop.T():
            sprint('loop_times: {} takes {:.2f} s'.format(loop_times,time.time()-t),
                    T=T,ser=None,logger=logger_looptime)

except:
    timeit.out('end',logger=logger_modelrun,T=T)  
    logger_results.add('{:.2f} : Error raised'.format(time.time()-T))
    logger_results.add(traceback.format_exc())

finally:
    mmap('release',caplist)
    mmap('close',[logger_gpio,logger_looptime,logger_modelrun,logger_results])
