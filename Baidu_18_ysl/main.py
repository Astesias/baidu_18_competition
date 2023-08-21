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

from votex_angle import core_votex
from mask2angle4 import core 

if os.name!='nt':
    from utils import Serial_init,Fplog
    from utils import Timety,Timer
    from utils import ser_read,quene_get,order_respone,post_data
    from utils import getime,sprint,set_all_gpio,mmap,check_cap,display_angle
    from detection import detection_init,predict,drawResults

def run(Q_order,cfg,open=False,switch_init=None):

    log_dir='log/'+getime()
    assert 'main.py' in os.listdir() , 'work directory is not correct :{os.getcwd()}'
    os.mkdir(log_dir)
    
    #logger_gpio    =Fplog(os.path.join(log_dir,'gpio.txt'))         # gpio输出
    logger_results =Fplog(os.path.join(log_dir,'results.txt'))      # 预测结果输出
    logger_looptime=Fplog(os.path.join(log_dir,'looptime.txt'))     # 循环计时
    logger_modelrun=Fplog(os.path.join(log_dir,'modelrun.txt'))     # 模型运行
    logger_list=[logger_results,logger_looptime,logger_modelrun]
    
    # configs
    h,w=map(int,cfg['input_frame_size'])
    assert not (h%32+w%32) , 'input_frame_size must be multiple of 32'
    
    
    serial_host,serial_bps=cfg['serial_host'],cfg['serial_bps']
    ser=Serial_init(serial_host,serial_bps,0.5)
    
    item_index=cfg['item_index']
    item_offet=-15
    target_index=cfg['spray_index']
    target_offset=0


    try:
        T=time.time() # 总计时
        
        cap1 = cv2.VideoCapture(cfg['videos'][0],cv2.CAP_V4L) # 前摄像头
        cap2 = cv2.VideoCapture(cfg['videos'][1],cv2.CAP_V4L) # 边摄像头
        #cap3 = cv2.VideoCapture(cfg['videos'][2],cv2.CAP_V4L) # 右摄像头
        caplist=[cap1,cap2]
        mmap('set',caplist,arg=[cv2.CAP_PROP_FRAME_WIDTH, w]) # 设置视频流大小
        mmap('set',caplist,arg=[cv2.CAP_PROP_FRAME_HEIGHT,h])
        #cap1.set(int(cap1.get(cv2.CAP_PROP_FOURCC)), cv2.VideoWriter_fourcc(*'MJPG'))
        check_cap(caplist,T=T,logger=logger_modelrun) # 检测摄像头状态

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
        down_label_left=[1,3,9999,1,1, 1, 9999]
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
        def SegmentationRoad(cap,display=False,mode_vside=False):
            _,frame=cap.read()
            frame=cv2.resize(frame,(640,480))
            nonlocal predict_frame
            predict_frame=frame.copy()
            nonlocal order
            order_respone(order,frame=frame)
            with mute_all():
                if mode_vside:
                    angle=core_votex(frame)
                else:
                    angle=core(frame)
            if display:
                frame=display_angle(frame,angle)
                DISPLAYER.putFrame(frame)
            return round(-angle),predict_frame

        timer_predict=Timer(0.8) # 最多1s识别一次
        timer_vortex=Timer(5) # 环岛最多5s一次

        Start=False # 运行标志
        switch=False # 摄像头切换
        put_flag=False
        if switch_init:
            switch=True
            switch_content=switch_init
            
        vortex_seek=0
        vortex_side_mode=False
        green_cnt=0  
        vortex_time=0
        fixed_angle=None  # - ->  + <-
        
    
        st_time=50
        rt_time_low=40
        rt_time_high=15
        
        rt_rota_low=-20
        rt_rota_high=-40
        
        v1_time=st_time+rt_time_low
        v2_time=rt_time_high+rt_time_low
        v3_time=rt_time_high
        
        
        while True:
            #print(ser.main_engine.out_waiting)
            #############
            order=quene_get(Q_order)
            ser_read(ser)
            
            if not (Start or order=='run' or len(sys.argv)==2 or open):
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
                if fixed_angle==None:
                    line_err,predict_frame=SegmentationRoad(cap1,mode_vside=vortex_side_mode)
                    sprint(f'[:{line_err:.0f}/]',T=T,ser=ser,logger=None,normal=False) 
                else:
                    line_err,predict_frame=SegmentationRoad(cap1,mode_vside=vortex_side_mode)
                    sprint(f'[:{fixed_angle:.0f}/]',T=T,ser=ser,logger=None,normal=False) 
                    fixed_angle=None

                if vortex_time:
                    print(vortex_time)
                    vortex_time-=1
                    if vortex_seek==1:
                        if vortex_time>st_time:
                            pass
                        else:
                            fixed_angle=rt_rota_low
                            if vortex_time==0:
                                vortex_side_mode=False
                                
                    elif vortex_seek==2:
                        if vortex_time>rt_time_high:
                            fixed_angle=rt_rota_high
                        else:
                            fixed_angle=rt_rota_low
                                
                    elif vortex_seek==3:
                        fixed_angle=rt_rota_high
                        vortex_side_mode=True    
                
            if timer_predict.T():   
                
                ################################## Detection
                if not switch:
                    results=PredictFrame(frame=predict_frame)
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
                              if not down_label_left[1]:
                                  vortex_side_mode=True
                              
                          elif kind==13 or vortex_time:
                              if timer_vortex.T():
                                  vortex_seek+=1
                                  print('Note:Seek vortex')
                                  if vortex_seek==1:
                                      print('vortex_time start 1')
                                      vortex_side_mode=True
                                      vortex_time=v1_time
                                  elif vortex_seek==2:
                                      print('vortex_time start 2')
                                      # vortex_side_mode=True
                                      vortex_time=v2_time
                                  elif vortex_seek==3:
                                      print('vortex_time start 3')
                                      # vortex_side_mode=True
                                      vortex_time=v3_time

        

                          
                            
                else:
                    timer_predict.sep=0
                    
                    results=PredictFrame(cap2)
                    result_from='cap2'
                    results=[row for row in results if row[0] in group_map[switch_content]]
                    results.sort(key=lambda x:x[1])
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
                            
                            sum_cx=0
                            target_item_flag=False
                            for index,result in enumerate(results):
                                kind,cx=result
                                sum_cx+=cx
                                if target_kind==kind:
                                    err=(cx+item_offet)-w/2
                                    target_item_flag=True
                            
                            
                            avg_cx=sum_cx/len(results) if len(results) else w/2
                            center_err=avg_cx-w/2
                            
                            if not target_item_flag:
                                err=3
                            else:
                                err=err/abs(err)*3 if abs(err)>=2 else 0
                                
                            if not put_flag:
                                center_err=666
                            else:
                                results=[result for result in results if result[0]!=target_kind]
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
                          
                          results=results[:3]
                          
                          spray_cx=[]
                          
                          for result in results:
                              _,cx=result
                              spray_cx.append(cx)
                          spray_cx.sort(reverse=True)
                          print(f'Spray {spray_cx}')

                          if len(spray_cx)<=2:
                              print(f'Warning spray num = {len(spray_cx)} contiune')
                              
                          #  target 0   100 50 10
                              
                          if len(spray_cx)==3:
                              err=w/2-spray_cx[target_index]+target_offset
                          elif 0<len(spray_cx)<=2:
                              if target_index==0:
                                  err=w/2-spray_cx[0]+target_offset
                              elif target_index==2:
                                  err=w/2-spray_cx[-1]+target_offset
                              elif target_index==1:
                                  err=w/2-sum(spray_cx)/len(spray_cx)+target_offset
                          elif len(spray_cx)==0:
                              print('Spray not found')
                                  
                          err=err if abs(err)>3 else 0
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

            # set_all_gpio('111111',normal=False,ser=None,logger=logger_gpio) # low speed

    except:
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
