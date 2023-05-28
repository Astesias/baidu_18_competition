# -*- coding:utf-8 编码测试
import os
import sys
import cv2
import time
import traceback
from pprint import pprint
from pysl import Config,truepath

from mask2angle import core

if os.name!='nt':
    from utils import Serial_init,Fplog
    from utils import Timeit,Timety,Timer
    from utils import ser_read,quene_get,order_respone
    from utils import getime,sprint,set_all_gpio,mmap,check_cap,display_angle
    from detection import detection_init,predict,drawResults

def run(Q_order,cfg):

    h,w=cfg['input_frame_size']
    assert not (h%32+w%32) , 'input_frame_size must be multiple of 32'
  
    serial_host,serial_bps=cfg['serial_host'],cfg['serial_bps']
    ser=Serial_init(serial_host,serial_bps,0.5)

    log_dir='log/'+getime()
    assert 'main.py' in os.listdir() , 'work directory is not correct :{os.getcwd()}'
    os.mkdir(log_dir)

    #logger_gpio    =Fplog(os.path.join(log_dir,'gpio.txt'))         # gpio输出
    logger_results =Fplog(os.path.join(log_dir,'results.txt'))      # 预测结果输出
    logger_looptime=Fplog(os.path.join(log_dir,'looptime.txt'))     # 循环计时
    logger_modelrun=Fplog(os.path.join(log_dir,'modelrun.txt'))     # 模型运行
    logger_list=[logger_results,logger_looptime,logger_modelrun]

    try:
        T=time.time() # 总计时
        timeit=Timeit('Initialization') # 开始初始化
        
        cap1 = cv2.VideoCapture('/dev/'+cfg['videos'][0],cv2.CAP_V4L) # 前摄像头
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
        def SegmentationRoad(cap,display=False):
            _,frame=cap.read()
            nonlocal order
            order_respone(order,frame=frame)
            angle=core(frame)
            if display:
                display_angle(frame,angle)
                cv2.waitKey(10)
                
            return '{:.0f}'.format(angle)

        timeit.out('Mainloop',logger=logger_modelrun,T=T) # 开始主循环

        timer_predict=Timer(0.05) # 最多0.05s识别一次
        timer_loop=Timer(5) # 循环log最多5s输出一次

        Start=False # 运行标志
        switch=True # 左右摄像头切换
        loop_times=0 
        result_l=result_r=[]
        check_cap(caplist,T=T,logger=logger_modelrun) # 检测摄像头状态
        while True:
            
            order=quene_get(Q_order)
            
            if not (Start or ser_read(ser) or order=='run'):
                continue
            else:
                Start=True
            if ser_read(ser) or order=='exit':
                Start=False

            t=time.time() # 循环开始计时
            loop_times+=1 

            if Start:
                line_info=SegmentationRoad(cap1)
                sprint(line_info,T=T,ser=None,logger=logger_results,end='\n\r')
                try:
                  ser.main_engine.flushInput() 
                  ser.main_engine.flushOutput() 
                  sprint(f'[!0:{line_info}/]\n\r',T=T,ser=ser,logger=None,normal=False)
                except:
                  ser.main_engine.close()
                  print('wait serial ')
                  time.sleep(3)
                  ser=Serial_init(serial_host,serial_bps,0.5)

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
                
            if T-time.time()>(10*60): # 防死机
                raise TimeoutError
            
            if timer_loop.T():
                sprint('loop_times: {} takes {:.2f} s'.format(loop_times,time.time()-t),
                        T=T,ser=None,logger=logger_looptime)

    except:
        timeit.out('end',logger=logger_modelrun,T=T)
        logger_results.add('{:.2f} : Error raised'.format(time.time()-T))
        logger_results.add(traceback.format_exc())
        print('Error main.py return ')

    finally:
        mmap('release',caplist) # 释放资源
        mmap('close',logger_list)


if __name__=='__main__':

    pprint(Config(truepath(__file__,'../configs.json')).data)

    if os.name!='nt':
        #run(Q(),Config(truepath(__file__,'../configs.json')).data)
        run(None,Config(truepath(__file__,'../configs.json')))
