# -*- coding:utf-8 编码测试
import os
import cv2
import time
import traceback
from pysl import Config,truepath
from mask2angle import core

if os.name!='nt':
    from utils import Serial_init,Fplog
    from utils import Timeit,Timety,Timer
    from utils import getime,sprint,set_all_gpio,mmap,check_cap,display_angle
    from detection import detection_init,predict,drawResults

def run(Q_order,cfg):

    h,w=cfg['input_frame_size']
    assert not (h%32+w%32) , 'input_frame_size must be multiple of 32'
  
    serial_host,serial_bps=cfg['serial_host'],cfg['serial_bps']
    ser=Serial_init(serial_host,serial_bps,0.5)

    log_dir='log/'+getime()
    assert 'main.py' in os.listdir() , 'work directory is not correct'
    os.mkdir(log_dir)

    logger_gpio    =Fplog(os.path.join(log_dir,'gpio.txt'),ser=None)        # gpio输出
    logger_results =Fplog(os.path.join(log_dir,'results.txt'),ser=None)      # 预测结果输出
    logger_looptime=Fplog(os.path.join(log_dir,'looptime.txt'),ser=None)    # 循环计时
    logger_modelrun=Fplog(os.path.join(log_dir,'modelrun.txt'),ser=None)    # 模型运行

    try:
        T=time.time() # 总计时
        timeit=Timeit('Initialization') # 开始初始化
        
        cap1 = cv2.VideoCapture('/dev/video0',cv2.CAP_V4L) # 前摄像头
        #cap2 = cv2.VideoCapture('/dev/video1',cv2.CAP_V4L) # 左摄像头
        #cap3 = cv2.VideoCapture('/dev/video2',cv2.CAP_V4L) # 右摄像头
        caplist=[cap1,]#cap2,cap3]
        mmap('set',caplist,arg=[cv2.CAP_PROP_FRAME_WIDTH, w]) # 设置视频流大小
        mmap('set',caplist,arg=[cv2.CAP_PROP_FRAME_HEIGHT,h])

        global MODEL_CONFIG,PREDICTOR,DISPLAYER # 初始化检测器
        DISPLAYER,MODEL_CONFIG,PREDICTOR=[None,None,None]#detection_init(cfg['model_json'])
        classes=None#MODEL_CONFIG.labels

        @Timety(timer=None,ser=None,logger=logger_modelrun,T=T) # 目标检测
        def PredictFrame(cap,display=True):
            _,frame=cap.read()
            if not _:
                raise IOError('Device bandwidth beyond')
            result = predict(frame,MODEL_CONFIG,PREDICTOR)
            if display: # 可视化调试
                drawResults(frame,result,MODEL_CONFIG)
                DISPLAYER.putFrame(frame)
            return mmap('unpack',result,arg=[MODEL_CONFIG.labels])

        #@Timety(timer=None,ser=None,logger=logger_modelrun,T=T) # 图像分割
        def SegmentationRoad(cap,display=True):
            _,frame=cap.read()
            angle=core(frame)
            if display:
                display_angle(frame,angle)
                cv2.waitKey(10)
                
            return '{:.2f}'.format(angle)

        timeit.out('Mainloop',logger=logger_modelrun,T=T) # 开始主循环

        timer_predict=Timer(0.05) # 最多0.05s识别一次
        timer_loop=Timer(5) # 循环log最多10输出一次

        Start=True # 首次运行标志
        switch=True # 左右摄像头切换
        loop_times=0 
        result_l=result_r=[]
        check_cap(caplist,T=T,logger=logger_modelrun) # 检测摄像头状态
        while True:
        
            t=time.time() # 循环开始计时
            loop_times+=1 

            if Start:
                line_info=SegmentationRoad(cap1)
                sprint(line_info,T=T,ser=None,logger=logger_results)
                sprint(line_info,T=T,ser=ser,logger=None,end='\n\r')

            # if timer_predict.T():   

            #     if switch:
            #         result_l=PredictFrame(cap2)
            #     else:
            #         result_r=PredictFrame(cap3)

            #     if Start:
            #         sprint('start',T=T,ser=ser,logger=logger_results)
            #         Start=False
            #     else:
            #         sprint('result_l: {} ==== result_r: {}'.format(result_l,result_r),
            #                 T=T,ser=ser,logger=logger_results)

                #switch=not switch
                #set_all_gpio('111111',normal=False,ser=None,logger=logger_gpio) # low speed
                
            if time.time()-T>(60*5): # 防死机
                raise TimeoutError
            
            if timer_loop.T():
                sprint('loop_times: {} takes {:.2f} s'.format(loop_times,time.time()-t),
                        T=T,ser=None,logger=logger_looptime)

    except:
        timeit.out('end',logger=logger_modelrun,T=T)
        logger_results.add('{:.2f} : Error raised'.format(time.time()-T))
        logger_results.add(traceback.format_exc())

    finally:
        mmap('release',caplist) # 释放资源
        mmap('close',[logger_gpio,logger_looptime,logger_modelrun,logger_results])


if __name__=='__main__':

    print(Config(truepath(__file__,'../configs.json')).data)

    if os.name!='nt':
        #run(Q(),Config(truepath(__file__,'../configs.json')).data)
        run(None,Config(truepath(__file__,'../configs.json')))
