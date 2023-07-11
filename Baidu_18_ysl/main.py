# -*- coding:utf-8 编码测试
import os
import sys
import cv2
import time
import traceback
import numpy as np
from pprint import pprint
from threading import Thread
from pysl import Config,truepath,mute_all

from mask2angle4 import core

if os.name!='nt':
    from utils import Serial_init,Fplog
    from utils import Timeit,Timety,Timer
    from utils import ser_read,quene_get,order_respone,post_data
    from utils import getime,sprint,set_all_gpio,mmap,check_cap,display_angle
    from detection import detection_init,predict,drawResults

def run(Q_order,cfg,open=False,switch_init=None):

    h,w=map(int,cfg['input_frame_size'])
    assert not (h%32+w%32) , 'input_frame_size must be multiple of 32'
    
    item_index=cfg['item_index']
    
  
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
        
        cap1 = cv2.VideoCapture(cfg['videos'][0],cv2.CAP_V4L) # 前摄像头
        cap2 = cv2.VideoCapture(cfg['videos'][1],cv2.CAP_V4L) # 边摄像头
        #cap3 = cv2.VideoCapture('/dev/video2',cv2.CAP_V4L) # 右摄像头
        caplist=[cap1,cap2]#cap2,cap3]
        mmap('set',[cap1],arg=[cv2.CAP_PROP_FRAME_WIDTH, 160]) # 设置视频流大小
        mmap('set',[cap1],arg=[cv2.CAP_PROP_FRAME_HEIGHT,160])
        mmap('set',[cap2],arg=[cv2.CAP_PROP_FRAME_WIDTH, 160]) # 设置视频流大小
        mmap('set',[cap2],arg=[cv2.CAP_PROP_FRAME_HEIGHT,160])
        #cap1.set(int(cap1.get(cv2.CAP_PROP_FOURCC)), cv2.VideoWriter_fourcc(*'MJPG'))

        global MODEL_CONFIG,PREDICTOR,DISPLAYER # 初始化检测器
        DISPLAYER,MODEL_CONFIG,PREDICTOR=detection_init(cfg['model_json'])

        classes=MODEL_CONFIG.labels
        #print(classes)
        # 0  bonfire            _
        # 1  building           _
        # 2  building_blue 
        # 3  building_cyan 
        # 4  building_green 
        # 5  endorbegin         _
        # 6  flyup              _
        # 7  item1 
        # 8  item2 
        # 9  item3 
        # 10 trade              _
        # 11 spray 
        # 12 spray_label        _
        # 13 vortex             _
        # [0 1 5 6 11 12]
        
        group_down=     [0,1,5   ,6,10,12,13]
        down_label_left=[1,3,9999,1,1, 1,  3]
        building_left=[1,1,1]
        # group_up=[2,3,4,7,8,9,11]
        group_map={
                    0:None,
                    1:[2,3,4],
                    5:None,
                    6:None,
                    10:[7,8,9],
                    12:[11],
                    13:None
                  }
        communicate_down_map={
                            1:'[@1/]',10:'[@2/]',0:'[@3/]',6:'[@4/]',12:'[@5/]',13:'[@6/]',5:''
                             }
        building_map={
                        2:'[$1/]',3:'[$2/]',4:'[$3/]'
                     }

        #@Timety(timer=Timer(0.5),ser=None,logger=logger_modelrun,T=T) # 目标检测
        def PredictFrame(cap=None,display=False,frame=None):
            if not isinstance(frame,np.ndarray):
                _,frame=cap.read()
            else:
                _ = True
            frame=cv2.resize(frame,(160,160))
            if not _:
                raise IOError('Device bandwidth beyond')
            result = predict(frame,MODEL_CONFIG,PREDICTOR)
            if display: # 可视化调试
                drawResults(frame,result,MODEL_CONFIG)
                DISPLAYER.putFrame(frame)
            return result #mmap('unpack',result,arg=[MODEL_CONFIG.labels])

        #@Timety(timer=None,ser=None,logger=logger_modelrun,T=T) # 图像分割
        def SegmentationRoad(cap,display=False):
            _,frame=cap.read()
            frame=cv2.resize(frame,(640,480))
            nonlocal predict_frame
            predict_frame=frame.copy()
            nonlocal order
            order_respone(order,frame=frame)
            with mute_all():
                angle=core(frame)
            if display:
                frame=display_angle(frame,angle)
                DISPLAYER.putFrame(frame)
            return round(-angle),predict_frame
        

        timeit.out('Mainloop',logger=logger_modelrun,T=T) # 开始主循环

        timer_predict=Timer(1) # 最多1s识别一次
        timer_vortex=Timer(3) # 环岛最多3s一次

        Start=False # 运行标志
        switch=False # 摄像头切换
        put_flag=False
        if switch_init:
            switch=True
            switch_content=switch_init
        vortex_seek=0
        vortex_time=0
        green_cnt=0  
            
        check_cap(caplist,T=T,logger=logger_modelrun) # 检测摄像头状态
        while True:
            
            #############
            order=quene_get(Q_order)
            
            if not (Start or 'start' in ser_read(ser) or order=='run' or len(sys.argv)==2 or open):
                continue
            else:
                #Thread(target=SegmentationRoad,args=[cap1]).start()
                Start=True
            # if order=='exit':
            #     Start=False
            #############

            # t=time.time() # 循环开始计时
            # loop_times+=1 

            if Start:

                ################################## Segmentation
                if not vortex_time:
                    line_err,predict_frame=SegmentationRoad(cap1)
                    sprint(f'[:{line_err:.0f}/]',T=T,ser=ser,logger=None,normal=False)
                
                #sprint(str(line_err) + ('->' if line_err<0 else '<-'),
                #       T=T,ser=None,logger=logger_results,end='\n\r')
                
                # post_data(cfg.server,f'S{line_info}')
                #ser.Send_data( '[:30\]'.encode('utf8'))
                #continue
             
            if timer_predict.T():   
                
                ################################## Detection
                if not switch:
                    results=PredictFrame(frame=predict_frame)
                    #continue
                    result_from='cap1'
                    if results:
                      kind,_=results[0]
                      if kind in group_down and down_label_left[group_down.index(kind)] and kind!=5:
                          sprint(communicate_down_map[kind],T=T,ser=ser,logger=None)
                          down_label_left[group_down.index(kind)]-=1
                          print(f'Detetion view {classes[kind]}')
                          if group_map[kind] and kind!=13:
                              h_,w_=h,w
                              h,w=160,160
                              switch=not switch
                              switch_content=kind
                              print(f'Detetion switch {classes[kind]}')
                              ser_read(ser) # flush input
                              tmp_T=timer_predict.sep
                              if kind==1:
                                  time.sleep(2)
                              
                          elif kind==13 or vortex_time:
                              
                              if timer_vortex.T() and down_label_left[group_down.index(kind)]:
                                  vortex_seek+=1
                                  print(f'Seek vortex at {vortex_seek} time')
                                  if vortex_seek==1:
                                      vortex_time==10000
                                      print('vortex_time start')
                                  elif vortex_seek==2:
                                      pass
                                  elif vortex_seek==3:
                                      vortex_time==20
                              else:
                                  vortex_time-=1
                                  if vortex_seek<=2:
                                      if vortex_seek>=9900:
                                          print('line',end=' ')
                                          sprint(f'[:0/]',T=T,ser=ser,logger=None,normal=True)
                                      else:
                                          print('round',end=' ')
                                          sprint(f'[:40/]',T=T,ser=ser,logger=None,normal=True)
                                  elif vortex_seek==3:
                                      vortex_time=0
                                      print('vortex_time end')
                                      sprint(f'[:40/]',T=T,ser=ser,logger=None,normal=True)
                              
                          
                else:
                    timer_predict.sep=0
                    
                    results=PredictFrame(cap2)
                    result_from='cap2'
                    results=[row for row in results if row[0] in group_map[switch_content]]
                    print(results)
                    
                    if results or put_flag:
                      if switch_content==1: # building
                            if kind==4:
                                green_cnt+=1
                            if kind!=4 or green_cnt>7:
                                kind,_=results[0]
                                if building_left[kind-2]:
                                    building_left[kind-2]-=1
                                    print(f'Building {classes[kind]}')
                                    sprint(building_map[kind],T=T,ser=ser,logger=None)
                                else:
                                    print(classes[kind],'was found')
  
                      elif switch_content==10: # items
                            target_kind=classes.index(cfg['items_kind'])

                            results.sort(key=lambda x:x[-1])
                            results=results[:3]

                            items=[_ for _ in group_map[switch_content]]
                            for index,result in enumerate(results):
                                kind,_=result
                                if kind in items:
                                    
                                    items.remove(kind)
                                else:
                                    okind,rkind=results[index][0],items.pop(0)
                                    results[index][0]=rkind

                                    print(f'auto replace: {classes[okind]} -> {classes[rkind]}')
                            print(results)
                            
                            sum_cx=0
                            target_flag=False
                            for index,result in enumerate(results):
                                kind,cx=result
                                sum_cx+=cx
                                if target_kind==kind:
                                    err=(cx-3)-w/2
                                    target_flag=True
                            
                            
                            avg_cx=sum_cx/len(results) if len(results) else w/2
                            center_err=avg_cx-w/2
                            
                            if not target_flag:
                                err=3
                            else:
                                err=err if abs(err)>3 else 0
                                
                            if not put_flag:
                                center_err=center_err if abs(center_err)>20 else 0
                            else:
                                if item_index==2:
                                    if len(results)==0:
                                        center_err=-3
                                    else:
                                        center_err=results[-1][1]-w/2
                                if item_index==1:
                                    center_err=0
                                    
                                if item_index==0:
                                    if len(results)==0:
                                        center_err=3
                                    else:
                                        center_err=results[0][1]-w/2
                            if err==0 and not put_flag:
                                print('| | | | | | | | | | | |\n| | Reach the item  | |\n| | | | | | | | | | | |')
                            if put_flag and center_err==0:
                                print('| | | | | | | | | | | |\n| | Put the item    | |\n| | | | | | | | | | | |')
                            sprint(f'[&{err:.0f}:{center_err:.0f}/]',T=T,ser=ser,logger=None)
  
                      elif switch_content==12: # spray
                          target_index=cfg['spray_index']
                          spray_cx=[]
                          results.sort(key=lambda x:x[1])
                          for result in results:
                              _,cx=result
                              spray_cx.append(cx)
                          spray_cx.sort()
                          print(f'Spray {spray_cx}')
                          if len(spray_cx)!=3:
                              print(f'Warning spray num = {len(spray_cx)} instead of 3')
                          if len(spray_cx)<2:
                              print(f'Warning spray num = {len(spray_cx)} contiune')
                          
                          target_index=target_index if target_index<=1 else -1 # 0 1 -1
                          target_cx=spray_cx[target_index]
                          err=w/2-target_cx
                          err=err if abs(err)>20 else 0
                          print(f'Spray [*{err:.0f}/]')
                          sprint(f'[*{err:.0f}/]',T=T,ser=ser,logger=None)
                          pass

                    msg=ser_read(ser)
                    if 'done' in msg:
                        switch=not switch
                        print(f'Detetion switch out from {classes[kind]}')
                        timer_predict.sep=tmp_T
                        h,w=h_,w_
                        put_flag=False
                    elif 'ok' in msg:
                        put_flag=True
                        

            if T-time.time()>(10*60): # 防死机
                raise TimeoutError

            # if timer_loop.T():
            #     sprint('loop_times: {} takes {:.2f} s'.format(loop_times,time.time()-t),
            #             T=T,ser=None,logger=logger_looptime)
            # set_all_gpio('111111',normal=False,ser=None,logger=logger_gpio) # low speed

    except:
        timeit.out('end',logger=logger_modelrun,T=T)
        logger_results.add('{:.2f} : Error raised'.format(time.time()-T))
        logger_results.add(traceback.format_exc())
        print('Error main.py returned ')
        print(traceback.format_exc())

    finally:
        mmap('release',caplist) # 释放资源
        mmap('close',logger_list)


if __name__=='__main__':

    pprint(Config(truepath(__file__,'../configs.json')).data)

    if os.name!='nt':
        from multiprocessing import Queue 
        Q_Order=Queue(maxsize=5)
        #run(Q(),Config(truepath(__file__,'../configs.json')).data)
        run(Q_Order,Config(truepath(__file__,'../configs.json')),1)
