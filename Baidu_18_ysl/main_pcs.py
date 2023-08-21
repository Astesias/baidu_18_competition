# -*- coding:utf-8 编码测试
import os
import sys
import cv2
import time
import traceback
import numpy as np
from pprint import pprint
from threading import Thread
from pysl import Config, truepath, mute_all, cmd
from multiprocessing import Process, Queue

from vortex_mode import core as vor_core
from mask2angle5 import core

if os.name != 'nt':
    from utils import Serial_init, Fplog
    from utils import Timety, Timer, tmp_timer
    from utils import ser_read, quene_get, order_respone, post_data
    from utils import getime, sprint, set_all_gpio, mmap, check_cap, display_angle
    from detection import detection_init, predict, drawResults


def run(Q_order, cfg, open=False, switch_init=None):

    T = time.time()  # 总计时
    
    log_dir = 'log/'+getime()
    assert 'main.py' in os.listdir(
    ), 'work directory is not correct :{os.getcwd()}'
    os.mkdir(log_dir)

    # logger_gpio    =Fplog(os.path.join(log_dir,'gpio.txt'))         # gpio输出
    logger_results = Fplog(os.path.join(log_dir, 'results.txt'))      # 预测结果输出
    logger_looptime = Fplog(os.path.join(log_dir, 'looptime.txt'))     # 循环计时
    logger_modelrun = Fplog(os.path.join(log_dir, 'modelrun.txt'))     # 模型运行
    logger_list = [logger_results, logger_looptime, logger_modelrun]

    # configs
    h, w = map(int, cfg['input_frame_size'])
    assert not (h % 32+w % 32), 'input_frame_size must be multiple of 32'
    #print('ttyUSB',cmd("ls /dev/ttyU*"))
    
    item_index = cfg['item_index']
    item_offet = -15
    target_index = cfg['spray_index']
    target_offset = 30

    global MODEL_CONFIG, PREDICTOR  # 初始化检测器
    DISPLAYER, MODEL_CONFIG, PREDICTOR = detection_init(cfg['model_json'])
    classes = MODEL_CONFIG.labels

    cmd('cp ../tmp.jpg ./')
    sprint('\nNote: copy tmp.jpg',T=T)
    
    ## try
    sprint('Note: Serial init',T=T)
    serial_host, serial_bps = cfg['serial_host'], cfg['serial_bps']
    ser = Serial_init(serial_host, serial_bps, 0.5)
    sprint('Note: Serial init done',T=T,ser = ser)
    
    cap1 = cv2.VideoCapture(cfg['videos'][0], cv2.CAP_V4L)  # 前摄像头
    cap2 = cv2.VideoCapture(cfg['videos'][1], cv2.CAP_V4L)  # 边摄像头
    if 'SP2812: SP2812' in cmd('v4l2-ctl -d '+cfg['videos'][0]+ ' --all'):
        pass
    else:
        cap1,cap2=cap2,cap1
    caplist = [cap1, cap2]
    mmap('set', [cap1], arg=[cv2.CAP_PROP_FRAME_WIDTH, w])  # 设置视频流大小
    mmap('set', [cap1], arg=[cv2.CAP_PROP_FRAME_HEIGHT, h])
    
    mmap('set', [cap2], arg=[cv2.CAP_PROP_FRAME_WIDTH, 640])  # 设置视频流大小
    mmap('set', [cap2], arg=[cv2.CAP_PROP_FRAME_HEIGHT, 640])
    
    
    #cap1.set(int(cap1.get(cv2.CAP_PROP_FOURCC)), cv2.VideoWriter_fourcc(*'MJPG'))
    check_cap(caplist, T=T, logger=logger_modelrun)  # 检测摄像头状态
    
    w=640

    # cleass defind
    bonfire = 0          # 0  bonfire            _
    building = 1          # 1  building           _
    building_blue = 2          # 2  building_blue
    building_cyan = 3          # 3  building_cyan
    building_green = 4          # 4  building_green
    endorbegin = 5          # 5  endorbegin         _
    flyup = 6          # 6  flyup              _
    item1 = 7          # 7  item1
    item2 = 8          # 8  item2
    item3 = 9          # 9  item3
    trade = 10         # 10 trade              _
    spray = 11         # 11 spray
    spray_label = 12         # 12 spray_label        _
    vortex = 13         # 13 vortex             _
    
    class_name = {0:'bonfire',1:'building',6:'flyup',10:'trade',12:'spray_l'}

    group_down =      [bonfire, building, endorbegin,flyup, trade, spray_label, vortex]
    down_label_left = [1,       3,        9999,      1,     1,     1,           9999]
    #down_label_left = [0,       0,        9999,      1,     0,     0,           0]
    #                   ？       x         x          x      ？     x             x                       
    building_left = [1, 1, 1]
    view_list = [spray_label,building,building,bonfire,trade,flyup,building]
    view_list = [spray_label,building,building,bonfire,trade,building]
    #view_list = [spray_label,building,building,bonfire,trade,flyup,building]
    view_list = [flyup,building]
    #view_list = [trade]
    #view_list = [building]

    for _i,_ in enumerate(down_label_left):
        if _==0:
            while group_down[_i] in view_list:
                view_list.remove(group_down[_i])
    view_list_index = 0
    sprint(f'view list: {view_list}',T=T)
    sprint([class_name[_] for _ in view_list],T=T)
    
    
    group_map = {
        bonfire:    None,
        building:   [building_blue, building_cyan, building_green],
        endorbegin: None,
        flyup:      None,
        trade:      [item1, item2, item3],
        spray_label: [spray],
        vortex:     None
    }
    communicate_down_map = {
        building: '[@1/]', trade: '[@2/]', bonfire: '[@3/]', flyup: '', #'[@x/]',
        spray_label: '[@5/]', vortex: '[@6/]', endorbegin: ''
    }
    beep_msg=communicate_down_map[vortex]
    building_map = {
        building_blue: '[$1/]', building_cyan: '[$2/]', building_green: '[$3/]'
    }

    class Result(): # wrapper
        def __init__(self,result):
            self.data=list(result)
            self.__str__=self.__repr__
            self.data[3]=round(self.data[3],2)
        def cx(self):
            return self.data[1]
        def kind(self):
            return self.data[0]
        def cy(self):
            return self.data[2]
        def score(self):
            return round(self.data[3],2)
        def __iter__(self):
            yield self.kind()
            yield self.cx()
        def __getitem__(self,i):
            return self.data[:2][i]
        def __repr__(self):
            return str(self.data)
            
    # @Timety(timer=Timer(0.5),ser=None,logger=logger_modelrun,T=T) # 目标检测
    def PredictFrame(cap=None, display=False, frame=None,resize=None):
        if not isinstance(frame, np.ndarray):
            _, frame = cap.read()
            # print(frame.shape)
            assert _,'ERROR PRE CAP DISCONNECT'
        else:
            _ = True
        #print('Pre before resize',frame.shape)
        if not resize:
          frame = cv2.resize(frame, (160,160))
        else:
          frame = cv2.resize(frame, resize)
        #print('Pre after resize',frame.shape)
        
        #cv2.imwrite('__tmp.jpg', frame)
        if not _:
            raise IOError('Device bandwidth beyond')
        result = predict(frame, MODEL_CONFIG, PREDICTOR)

        if display:  # 可视化调试
            drawResults(frame, result, MODEL_CONFIG)
            DISPLAYER.putFrame(frame)
        return result  # mmap('unpack',result,arg=[MODEL_CONFIG.labels])

    def predict_process(Qo, Qi):
        pause = False
        once  = False
        while 1:
            try:
                while 1:
                    if Qi.qsize():
                        order=Qi.get()
                        if order=='pause':
                            pause = True
                        elif order=='resume':
                            pause = False
                        elif order=='once':
                            once = True
                        
                        while Qo.qsize():
                            Qo.get()
                        
                    if not pause or once:
                        try:
                            frame = cv2.imread('tmp.jpg')
                        except:
                            continue
                        r=PredictFrame(frame=frame,resize=( (640,480) if once else None))
                        once = False
                        Qo.put(r)
                        # if r:
                        #     print(('' if r[0][0]!=vortex else 'Warning add vor\n'),end='')
                    else:
                        while Qo.qsize():
                            Qo.get()
            except:
                print((traceback.format_exc()))

    # @Timety(timer=None,ser=None,logger=logger_modelrun,T=T) # 图像分割
    def SegmentationRoad(cap, display=False, mode_vside=None):
        _, frame = cap.read()
        assert _,'ERROR SEG CAP DISCONNECT'
        
        #print('Seg before resize',frame.shape)
        cv2.imwrite('_tmp.jpg', frame)
        os.rename('_tmp.jpg', 'tmp.jpg')
        frame = cv2.resize(frame, (640//2, 480//2))
        #print('Seg after resize',frame.shape)
        nonlocal predict_frame
        predict_frame = frame.copy()
        nonlocal order
        order_respone(order, frame=frame)
        # with mute_all():
        if mode_vside!=None:
            angle = 2*vor_core(frame,close_left=mode_vside)
        else:
            angle = core(frame)
        if display:
            frame = display_angle(frame, angle)
            DISPLAYER.putFrame(frame)
        return round(-angle), predict_frame

    def fixed_angle(x):
        sprint(f'[:{x:.0f}/]', T=T, ser=ser, logger=None, normal=False)   # - ->  + <-

    Qo = Queue() # 模型结果输出
    Qi = Queue() # 进程控制
    
    
    process=True
    if process:
        Process(target=predict_process, args=[Qo, Qi]).start()
        print('Note: mutiprocess using')

    timer_vortex = Timer(5)  # 环岛最多5s一次
    timer_build = Timer(2)  # 识别最多5s一次

    Start = False  # 运行标志
    switch = False  # 摄像头切换
    sprint(f'Note: Wait start {not (Start or open)}',T=T)
    
    mid_kind = None
    put_flag = False
    item_search_dir=None # 1 forword
    item_find_center_dir=1
    
    spray_find_mid = False
    spray_find_mid_dir = -1
    spray_last=None
    
    if switch_init:
        switch = True
        switch_content = switch_init
        sprint(f'Note: switch content init into {classes[switch_content]}',T=T)

    green_cnt = blue_cnt = cyan_cnt = 0
    switch_delay = 0

    vortex_seek = 0
    vortex_LEFT=1
    vortex_RIGHT=0
    vortex_content =[vortex_LEFT,vortex_RIGHT,None][2]
    vortex_time=0
    vortex_time_start=0
    vortex_time_content=None

    for _ in range(3):
        PredictFrame(cap2)  # warmup model
    sprint('Note: Success Start!!!',T=T)
    for _ in range(3):
        sprint(beep_msg,T=T, ser=ser,normal=False)
        time.sleep(0.1)
    sprint('Note: beep for opening!!',T=T)
    
    while True:

        #print(ser.main_engine.out_waiting,ser.main_engine.in_waiting)
        order = quene_get(Q_order)
        if order:
            sprint('MSG TAKE:',T=T)
            sprint(order,ser=ser,T=T)
        
        if not (Start or order == 'run' or len(sys.argv) == 2 or open):
                continue
        else:
            ser_read(ser)
            Start = True
            
        if Start:
            # Segmentation
            if vortex_content==None:
                line_err, predict_frame = SegmentationRoad(cap1)
            elif time.time()-vortex_time_start<vortex_time:
                line_err, predict_frame = SegmentationRoad(cap1,mode_vside=vortex_time_content)
            else:
                line_err, predict_frame = SegmentationRoad(cap1,mode_vside=vortex_content)
            fixed_angle(line_err)
        

        # Detection
        if not switch:
            # continue
            if not process:
                results=PredictFrame(frame=predict_frame)
                results=[Result(result) for result in results]
            else:
                #print(Qo.qsize())
                if Qo.qsize():
                    results = Qo.get()
                    results=[Result(result) for result in results]
                else:
                    results=[]
                
            if switch_delay:
                switch_delay-=1
                continue
                
            if results: #and timer_predict.T():
                #continue
                kind, _ = results[0]
                if kind in group_down and down_label_left[group_down.index(kind)] and kind != endorbegin:
                    
                    sprint(f'Detetion view {classes[kind]} {results[0].score():.2f}',T=T)
                    if kind!=vortex:
                        if view_list_index==len(view_list):
                            sprint('Error next should be None',T=T)
                            continue
                        elif kind!=view_list[view_list_index]:
                            sprint(f'Error next should be {class_name[view_list[view_list_index]]}',T=T)
                            continue
                        else:
                            view_list_index+=1
                    else:
                        if view_list_index!=len(view_list):
                            sprint('Error vortex should be final',T=T)
                            continue

                    sprint(communicate_down_map[kind],T=T, ser=ser,logger=None,mute=(kind!=vortex)) # 地标发多次
                    if kind!=vortex:
                        down_label_left[group_down.index(kind)] -= 1
                    
                    if group_map[kind] and kind != vortex:
                        switch = not switch
                        switch_content = kind
                        sprint(f'Detetion switch {classes[kind]}',T=T)
                        ser_read(ser)  # flush input
                        Qi.put('pause')
                        sprint('Note: predictor pause',T=T)
                        # if not down_label_left[building]:
                        #     vortex_side_mode = True
        
                    elif kind == vortex and timer_vortex.T():
                        vortex_seek += 1
                        print('v mode ',vortex_seek)
                        if vortex_seek==1:
                            vortex_time=1.5
                            vortex_time_start=time.time()
                            vortex_time_content=vortex_LEFT
                            vortex_content=vortex_RIGHT
                        elif vortex_seek==2:
                            vortex_content=vortex_RIGHT
                        elif vortex_seek==3:
                            vortex_time=1
                            vortex_time_start=time.time()
                            vortex_time_content=vortex_RIGHT
                            vortex_content=vortex_LEFT

        else:
            results = PredictFrame(cap2,resize=(640,640))
            results = [row for row in results if row[0]
                       in group_map[switch_content]]
            results=[Result(result) for result in results]
            results = results[:3]
            results.sort(key=lambda x: x[1])
            # if not results and switch_content==building:
            #     sprint('[*30/]',ser=ser,T=T)
            

            if results or put_flag:
                if switch_content == building:  # building
                    if building not in view_list[view_list_index:]:
                        vortex_content=vortex_LEFT
                    else:
                        print(view_list,view_list_index)
                    if results:
                        kind,cx=results[0]
                        sprint(f'Seek building {classes[kind]}',T=T)
                        if kind == building_green:
                            green_cnt += 1
                        elif kind == building_cyan:
                            cyan_cnt+=1
                        elif kind==building_blue:
                            blue_cnt+=1
                        ################
                        # if abs(cx-w/2)>70:
                        #     building_err = 70*(cx-w/2)/abs(cx-w/2)
                        #     sprint(f'[*{building_err:.0f}/]',T=T)
                        #     continue
                        ################
                    #print(green_cnt,cyan_cnt,blue_cnt)
                    if ((kind == building_green and green_cnt > 5) or \
                        (kind == building_cyan and cyan_cnt > 5) or \
                        (kind == building_blue and blue_cnt > 5) ) and timer_build.T():
                        kind, _ = results[0]
                        if building_left[kind-2]:
                            green_cnt = blue_cnt = cyan_cnt = 0
                            building_left[kind-2] -= 1
                            sprint(f'Building {classes[kind]}',T=T)
                            sprint(building_map[kind],T=T, ser=ser, logger=None,mute=True)  
                        else:
                            sprint(f'{classes[kind]} was found',T=T)

                elif switch_content == trade:  # items
                    down_label_left[group_down.index(building)]=1  # fix
                    target_kind = classes.index(cfg['items_kind']) 
                    
                    err=center_err=-666
                    mid_line=w/2 + item_offet
                    
                    if put_flag:
                        results = [result for result in results if result[0] != target_kind]
                    results = results[:3] if not put_flag else results[:2]
                    if mid_kind and put_flag:
                        target_kind = mid_kind
                    if results:
                        cnt_cx=len(results)
                        
                        sum_cx=0
                        target_cx=None
                        items = [_ for _ in group_map[switch_content]]
                        break_flag = False
                        for index, result in enumerate(results):
                            kind, cx = result
                            sum_cx += cx
                            
                            if kind==target_kind:
                                target_cx=cx
                            if kind in items:
                                items.remove(kind)
                            else:
                                break_flag=True
                                sprint(results,T=T)
                                sprint('Search error',T=T)
                                break 
                        if break_flag:
                            continue
                        avg_cx=round(sum_cx/cnt_cx,2)
                        
                        if not put_flag:
                            if not mid_kind:
                                if abs(avg_cx-mid_line)>100 and cnt_cx==1:
                                    item_find_center_dir = 1 if (avg_cx-mid_line)>100 else -1
                                err = item_find_center_dir * 70
                                if cnt_cx==3:
                                    mid_kind = results[1][0]
                                    item_search_dir = 1 if (target_cx-mid_line)>0 else -1
                                    if mid_kind==target_kind:
                                        item_search_dir = 0
                                    continue
                            else:
                                err = item_search_dir * 70
                                if target_cx:
                                    err = target_cx - mid_line
                        else:
                            if item_search_dir==0:
                                center_err = 0
                            else:
                                center_err = -item_search_dir * 70
                                if target_cx:
                                    center_err = target_cx - mid_line
                        sprint(results,T=T)
                        print(f'mid_dir {item_find_center_dir},find_dir {item_search_dir},target {target_kind}\n'+\
                              f'avg_cx  {avg_cx},mid_kind {mid_kind},target_cx {target_cx}\n')
                                
                        err =err if abs(err) >= 6 else 0
                        center_err =center_err if abs(center_err) >= 6 else 0
                            
                        if err == 0 and not put_flag:
                            print('Reach the item')
                        if put_flag and center_err == 0:
                            print('Put the item')
                        sprint(f'[&{err:.0f}:{center_err:.0f}/]',T=T, ser=ser)
                    

                elif switch_content == spray_label:  # spray

                    # spray_find_mid = False
                    # spray_find_mid_dir = -1
                    
                    results = results[:3]
                    if not results:
                        continue
                    mid_line = w/2+target_offset
                    
                    spray_cx = []
                    for result in results:
                        _, cx = result
                        spray_cx.append(cx)
                    spray_cx.sort(reverse=True)
                    sprint(spray_cx,T=T)
                    
                    cnt_spray=len(spray_cx)
                    avg_spray=round(sum(spray_cx)/cnt_spray,2)

                    if not spray_find_mid:
                        if cnt_spray==3:
                            spray_find_mid=True
                            spray_last = spray_cx
                            continue
                        else:
                            if abs(avg_spray-mid_line)>100 and cnt_spray==1:
                                spray_find_mid_dir = -1 if (avg_spray-mid_line)>100 else 1
                              
                            err = spray_find_mid_dir * 70
                            
                    else:
                        if cnt_spray == 3:
                            err = mid_line-spray_cx[target_index]
                            spray_last = spray_cx
                        elif 0 < cnt_spray <= 2:
                            if target_index == 0:
                                err = mid_line-spray_cx[0]
                            elif target_index == 2:
                                err = mid_line-spray_cx[-1]
                            elif target_index == 1:
                                mid_last=spray_last[1]
                                compare_cx=[abs(_-mid_last) for _ in spray_cx]
                                err = mid_line-spray_cx[compare_cx.index(min(compare_cx))]
                        else:
                            sprint('Search error',T=T)
                    print('spy_findmid',spray_find_mid,'spy_finddir',spray_find_mid_dir,
                          'avg_cx',avg_spray)
                        
                    err = err if abs(err) > 6 else 0
                    sprint(f'[*{err:.0f}/]', T=T, ser=ser, logger=None)
          

            msg = ser_read(ser)
            if 'done' in msg:
                switch = not switch
                Qi.put('resume')
                sprint(f'Detetion switch out from {classes[kind]}',T=T)
                sprint('predictor resume',T=T)
                green_cnt = blue_cnt = cyan_cnt = 0
                put_flag = False
                switch_delay = 20
            elif 'ok' in msg:
                put_flag = True

        if T-time.time() > (10*60):  # 防死机
            raise TimeoutError

        # set_all_gpio('111111',normal=False,ser=None,logger=logger_gpio) # low speed
            
            
    # except:
    #     # logger_results.add('{:.2f} : Error raised'.format(time.time()-T))
    #     # logger_results.add(traceback.format_exc())
    #     print('Error main.py returned ')
    #     print(traceback.format_exc())

    # finally:
    #     mmap('release', caplist)  # 释放资源
    #     mmap('close', logger_list)


if __name__ == '__main__':

    pprint(Config(truepath(__file__, '../configs.json')).data)

    if os.name != 'nt':
        Q_Order = Queue(maxsize=5)
        # run(Q(),Config(truepath(__file__,'../configs.json')).data)
        run(Q_Order, Config(truepath(__file__, '../configs.json')), False)
  
