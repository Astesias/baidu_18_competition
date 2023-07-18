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

from votex_angle import core_votex
from mask2angle4 import core

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
    #print(h,w)
    
    serial_host, serial_bps = cfg['serial_host'], cfg['serial_bps']
    ser = Serial_init(serial_host, serial_bps, 0.5)
    
    item_index = cfg['item_index']
    item_offet = -18
    target_index = cfg['spray_index']
    target_offset = -6

    global MODEL_CONFIG, PREDICTOR  # 初始化检测器
    DISPLAYER, MODEL_CONFIG, PREDICTOR = detection_init(cfg['model_json'])
    classes = MODEL_CONFIG.labels

    cmd('cp ../tmp.jpg ./')
    sprint('\nNote: copy tmp.jpg',T=T)
    try:
        cap1 = cv2.VideoCapture(cfg['videos'][0], cv2.CAP_V4L)  # 前摄像头
        cap2 = cv2.VideoCapture(cfg['videos'][1], cv2.CAP_V4L)  # 边摄像头
        # cap3 = cv2.VideoCapture(cfg['videos'][2],cv2.CAP_V4L) # 右摄像头
        caplist = [cap1, cap2]
        mmap('set', caplist, arg=[cv2.CAP_PROP_FRAME_WIDTH, w])  # 设置视频流大小
        mmap('set', caplist, arg=[cv2.CAP_PROP_FRAME_HEIGHT, h])
        #cap1.set(int(cap1.get(cv2.CAP_PROP_FOURCC)), cv2.VideoWriter_fourcc(*'MJPG'))
        check_cap(caplist, T=T, logger=logger_modelrun)  # 检测摄像头状态

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

        group_down = [bonfire, building, endorbegin,
                      flyup, trade, spray_label, vortex]
        down_label_left = [1,      3,       9999,
                           1,    1,    1,          9999]
        building_left = [1, 1, 1]
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
            building: '[@1/]', trade: '[@2/]', bonfire: '[@3/]', flyup: '[@4/]',
            spray_label: '[@5/]', vortex: '[@6/]', endorbegin: ''
        }
        beep_msg=communicate_down_map[vortex]
        building_map = {
            building_blue: '[$1/]', building_cyan: '[$2/]', building_green: '[$3/]'
        }

        # @Timety(timer=Timer(0.5),ser=None,logger=logger_modelrun,T=T) # 目标检测
        def PredictFrame(cap=None, display=False, frame=None):
            if not isinstance(frame, np.ndarray):
                _, frame = cap.read()
            else:
                _ = True
            frame = cv2.resize(frame, (320, 320))
            if not _:
                raise IOError('Device bandwidth beyond')
            result = predict(frame, MODEL_CONFIG, PREDICTOR)
            if display:  # 可视化调试
                drawResults(frame, result, MODEL_CONFIG)
                DISPLAYER.putFrame(frame)
            return result  # mmap('unpack',result,arg=[MODEL_CONFIG.labels])

        def predict_process(Qo, Qi):
            pause = False
            while 1:
                try:
                    while 1:
                        if Qi.qsize():
                            msg=Qi.get()
                            if msg=='break':
                                sys.exit(0)
                                break
                            pause = not pause
                            
                            while Qo.qsize():
                                Qo.get()
                            
                        if not pause:
                            try:
                                frame = cv2.imread('tmp.jpg')
                            except:
                                continue
                            r=PredictFrame(frame=frame)
                            Qo.put(r)
                            # if r:
                            #     print(('' if r[0][0]!=vortex else 'Warning add vor\n'),end='')
                        else:
                            while Qo.qsize():
                                Qo.get()
                except:
                    import traceback
                    print((traceback.format_exc()))
                    sys.exit(0)
                    break
                
            

        # @Timety(timer=None,ser=None,logger=logger_modelrun,T=T) # 图像分割
        def SegmentationRoad(cap, display=False, mode_vside=False):
            _, frame = cap.read()
            cv2.imwrite('_tmp.jpg', frame)
            os.rename('_tmp.jpg', 'tmp.jpg')
            frame = cv2.resize(frame, (640, 480))
            nonlocal predict_frame
            predict_frame = frame.copy()
            nonlocal order
            order_respone(order, frame=frame)
            # with mute_all():
            if mode_vside:
                angle = core_votex(frame)
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
        #timer_predict = Timer(3)  # 识别最多5s一次

        Start = False  # 运行标志
        switch = False  # 摄像头切换
        put_flag = False
        sprint(f'Note: Wait start {not (Start or open)}',T=T)
        
        if switch_init:
            switch = True
            switch_content = switch_init
            sprint(f'Note: switch content init into {classes[switch_content]}',T=T)

        green_cnt = 0
        last_kind=[-1,-1]

        vortex_contrl = False
        vortex_seek = 0
        vortex_side_mode = False
        sprint(f'Note: vortex side mode {vortex_side_mode}',T=T)

        slow_vortex_para=True
        sprint('Note: vortex_param use '+('slow' if slow_vortex_para else 'fast'),T=T)
        if not slow_vortex_para:
            st_time = 2
            rt_time_low = 2
            rt_time_high = 1.2
    
            rt_rota_low = -20
            rt_rota_high = -40
            
        else:
            st_time = 3.5
            rt_time_low = 2.5
            rt_time_high = 1.6
            st_time_low = 1.5
    
            rt_rota_low = -20
            rt_rota_high = -40     
        

        for _ in range(3):
            PredictFrame(cap2)  # warmup model
        sprint('Note: Success Start!!!',T=T)
        sprint('Note: beep for opening!!',T=T)
        sprint(beep_msg,T=T, ser=ser,normal=False)
                 
        while True:
            
            #print(ser.main_engine.out_waiting,ser.main_engine.in_waiting)
            #############
            order = quene_get(Q_order)
            msg = ser_read(ser)
            
            if msg=='exit':
                Qi.put('break')
                sys.exit(0)
            
            if not (Start or order == 'run' or len(sys.argv) == 2 or open):
                    continue
            else:
                ser_read(ser)
                Start = True
                
                
            # if order=='exit':
            #     Start=False
            #############

            # t=time.time() # 循环开始计时
            # loop_times+=1

            if Start:
                #print(Qo.qsize())
                # Segmentation
                if not vortex_contrl:
                    line_err, predict_frame = SegmentationRoad(cap1, mode_vside=vortex_side_mode)
                    fixed_angle(line_err)
                    #sprint('[@]',T=T, ser=ser, logger=None)
                    #fixed_angle('[@]')

                else:
                    if vortex_seek == 1:
                        tmpt = tmp_timer(st_time)
                        while not tmpt():
                            line_err, predict_frame = SegmentationRoad(cap1, mode_vside=vortex_side_mode)
                            fixed_angle(line_err)
                        #     print('s 1 1')
                        # print('s 1 1 end')
                        tmpt = tmp_timer(rt_time_low)
                        while not tmpt():
                            fixed_angle(rt_rota_low)
                        #     print('s 1 2')
                        # print('s 1 end')
                        vortex_side_mode = False

                    elif vortex_seek == 2:
                        tmpt = tmp_timer(rt_time_high)
                        while not tmpt():
                            fixed_angle(rt_rota_high)
                            # print('s 2 1')
                            
                        tmpt = tmp_timer(st_time_low)
                        vortex_side_mode = True
                        while not tmpt():
                            line_err, predict_frame = SegmentationRoad(cap1, mode_vside=vortex_side_mode)
                            fixed_angle(line_err)                          
                            # print('s 2 2')
                        vortex_side_mode = False
                            
                        tmpt = tmp_timer(rt_time_low)
                        while not tmpt():
                            fixed_angle(rt_rota_low)
                            # print('s 2 3')

                    elif vortex_seek == 3:
                        tmpt = tmp_timer(rt_time_high)
                        while not tmpt():
                            fixed_angle(rt_rota_high)
                            # print('s 3 1')
                        vortex_side_mode = True
#                    while Qo.qsize():
#                        Qo.get()
                    Qi.put('resume')
                    sprint(f'Note: predictor resume {vortex_seek}',T=T)
                    vortex_contrl = False
            # Detection
            if not switch:

                if not process:
                    results=PredictFrame(frame=predict_frame)
                else:
                    #print(Qo.qsize())
                    if Qo.qsize():
                        results = Qo.get()
                    else:
                        results=[]
                    
                #print('res',results,)
                if results: #and timer_predict.T():
                    kind, _ = results[0]
                    if kind in group_down and down_label_left[group_down.index(kind)] and kind != endorbegin:
                        
#                        if kind!=last_kind:
#                            pass
#                        else:
#                            last_kind=kind
#                            continue
                            
                        kinds_flag=True
                        for k in last_kind:
                            if k!=kind:
                                kinds_flag=False
                                break
                        if not kinds_flag:
                            last_kind.pop(0)
                            last_kind.append(kind)
                            continue
                    
                        sprint(communicate_down_map[kind],T=T, ser=ser, logger=None)
                        down_label_left[group_down.index(kind)] -= 1
                        sprint(f'Detetion view {classes[kind]}',T=T)
                        if group_map[kind] and kind != vortex:
                            switch = not switch
                            switch_content = kind
                            sprint(f'Detetion switch {classes[kind]}',T=T)
                            ser_read(ser)  # flush input
                            Qi.put('pause')
        
                            sprint(f'Note: predictor pause {vortex_seek}',T=T)
                            if not down_label_left[building]:
                                vortex_side_mode = True

                        elif kind == vortex:
                            if timer_vortex.T():
                                vortex_seek += 1
                                sprint(f'Note: Seek vortex {vortex_seek}',T=T)
                                Qi.put('pause')
                                sprint(f'Note: predictor pause {vortex_seek}',T=T)
                                vortex_contrl = True
                                if vortex_seek == 1:
                                    vortex_side_mode = True
                                elif vortex_seek == 2:
                                    pass
                                elif vortex_seek == 3:
                                    pass

            else:
                results = PredictFrame(cap2)
                results = [row for row in results if row[0]
                           in group_map[switch_content]]
                results.sort(key=lambda x: x[1])
                # print(results)

                if results or put_flag:
                    if switch_content == building:  # building
                        if kind == building_green:
                            green_cnt += 1
                        if kind != building_green or green_cnt > 7:
                            kind, _ = results[0]
                            if building_left[kind-2]:
                                building_left[kind-2] -= 1
                                sprint(f'Building {classes[kind]}',T=T)
                                sprint(building_map[kind],T=T, ser=ser, logger=None)
                            else:
                                sprint(f'{classes[kind]} was found',T=T)

                    elif switch_content == trade:  # items
                        target_kind = classes.index(cfg['items_kind'])

                        results = results[:3]

                        items = [_ for _ in group_map[switch_content]]
                        for index, result in enumerate(results):
                            kind, _ = result
                            if kind in items:
                                items.remove(kind)
                            else:
                                okind, rkind = results[index][0], items.pop(0)
                                results[index][0] = rkind

                                sprint(f'Warning auto replace: {classes[okind]} -> {classes[rkind]}',T=T)

                        sum_cx = 0
                        target_item_flag = False
                        for index, result in enumerate(results):
                            kind, cx = result
                            sum_cx += cx
                            if target_kind == kind:
                                err = (cx+item_offet)-w/2
                                target_item_flag = True

                        avg_cx = sum_cx/len(results) if len(results) else w/2
                        center_err = avg_cx-w/2

                        if not target_item_flag:
                            err = 3
                        else:
                            err = err/abs(err)*3 if abs(err) >= 2 else 0

                        if not put_flag:
                            center_err = 666
                        else:
                            results = [
                                result for result in results if result[0] != target_kind]
                            if item_index == 2:
                                if len(results) == 0:
                                    center_err = -3
                                else:
                                    center_err = results[-1][1]-w/2
                            if item_index == 1:
                                center_err = 0

                            if item_index == 0:
                                if len(results) == 0:
                                    center_err = 3
                                else:
                                    center_err = results[0][1]-w/2
                        if err == 0 and not put_flag:
                            print(
                                '| | | | | | | | | |\n| | Reach the item  | |\n| | | | | | | | | |')
                        if put_flag and center_err == 0:
                            print(
                                '| | | | | | | | | |\n| | Put the item    | |\n| | | | | | | | | |')
                        sprint(f'[&{err:.0f}:{center_err:.0f}/]',
                               T=T, ser=ser, logger=None)

                    elif switch_content == spray_label:  # spray

                        results = results[:3]

                        spray_cx = []

                        for result in results:
                            _, cx = result
                            spray_cx.append(cx)
                        spray_cx.sort(reverse=True)
                        sprint(f'Spray {spray_cx}',T=T)

                        if len(spray_cx) <= 2:
                            sprint(f'Warning spray num = {len(spray_cx)} contiune',T=T)

                        #  target 0   100 50 10

                        if len(spray_cx) == 3:
                            err = w/2-spray_cx[target_index]+target_offset
                        elif 0 < len(spray_cx) <= 2:
                            if target_index == 0:
                                err = w/2-spray_cx[0]+target_offset
                            elif target_index == 2:
                                err = w/2-spray_cx[-1]+target_offset
                            elif target_index == 1:
                                err = w/2-sum(spray_cx) / \
                                    len(spray_cx)+target_offset
                        elif len(spray_cx) == 0:
                            sprint('Warining Spray not found',T=T)

                        err = err if abs(err) > 5 else 0
                        sprint(f'[*{err:.0f}/]', T=T, ser=ser, logger=None)
                        pass

                if 'done' in msg:
                    switch = not switch
                    Qi.put('resume')
                    sprint(f'Detetion switch out from {classes[kind]}',T=T)
                    put_flag = False
                elif 'ok' in msg:
                    put_flag = True

            if T-time.time() > (10*60):  # 防死机
                raise TimeoutError

            # set_all_gpio('111111',normal=False,ser=None,logger=logger_gpio) # low speed
    except:
        logger_results.add('{:.2f} : Error raised'.format(time.time()-T))
        logger_results.add(traceback.format_exc())
        sprint('Error main.py returned ',T=T)
        sprint(traceback.format_exc(),T=T)

    finally:
        mmap('release', caplist)  # 释放资源
        mmap('close', logger_list)


if __name__ == '__main__':

    pprint(Config(truepath(__file__, '../configs.json')).data)

    if os.name != 'nt':
        Q_Order = Queue(maxsize=5)
        # run(Q(),Config(truepath(__file__,'../configs.json')).data)
        run(Q_Order, Config(truepath(__file__, '../configs.json')), False)
  