from multiprocessing import Process as pcs
from multiprocessing import Queue 
from pysl import Config,os_enter,easy_request,mmap,mute_all,drewlinecross
import os,config_make,time,sys
from pprint import pprint
import numpy as np
import os
from edgeboard import PaddleLitePredictor
import sys

os.chdir('./Baidu_18_ysl')
sys.path.insert(0,'./')
from core import Display,SystemConfig,ModelConfig,Timer,createCapture,fpga_preprocess
os.chdir('..')
sys.path.insert(0,'./')

import cv2

Q_Order=Queue(maxsize=5)
try:
    config_make.make_cfg()
except:
    Warning('configs make failed')
cfg=Config('./configs.json')

pprint(cfg.data)
server,port=cfg.server,cfg.port


g_predictor = PaddleLitePredictor()
class PredictResult(object):
    """result wrapper"""
    def __init__(self, category, score, x, y, width, height):
        """init"""
        self.type = int(category)
        self.score = score
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

def predictresult(type_, score, x, y, width, height):
    return [int(type_),int(x)+int(width/2)] #int(x),int(y),int(width),int(height)

def predictorInit():
    """predictor config and initialization"""
    global g_model_config
    global g_system_config

    global g_predictor
    if g_model_config.is_combined_model:
        g_predictor.set_model_file(g_model_config.model_file)
        g_predictor.set_param_file(g_model_config.params_file)
    else:
        # TODO add not combined model load
        pass
    try:
        g_predictor.load()
        print("Predictor Init Success !!!")
        return 0
    except:
        print("Error: CreatePaddlePredictor Failed.")
        return -1


def predict(frame,g_model_config,g_predictor, timer=None):
    """predict with paddlelite and postprocess"""
    origin_frame  = frame.copy()
    origin_h, origin_w, _ = origin_frame.shape


    # if not g_system_config.use_fpga_preprocess:
    #     input_data = cpu_preprocess(frame, g_model_config)
    #     g_predictor.set_input(input_data, 0)
    # else:
    #     input_tensor = g_predictor.get_input(0)
    #     fpga_preprocess(frame, input_tensor, g_model_config)
    input_tensor = g_predictor.get_input(0)
    fpga_preprocess(frame, input_tensor, g_model_config)
 
    if g_model_config.is_yolo:
        feed_shape = np.zeros((1, 2), dtype=np.int32)
        feed_shape[0, 0] = origin_h
        feed_shape[0, 1] = origin_w

        shape_tensor = g_predictor.set_input(feed_shape, 1)

    # global IS_FIRST_RUN
    # if IS_FIRST_RUN:
    #     IS_FIRST_RUN = False
    #     g_predictor.run()
    # else:
    #     if timer:
    #         timer.Continue()
    #     g_predictor.run()
    #     if timer:
    #         timer.Pause()

    g_predictor.run()

    outputs = np.array(g_predictor.get_output(0))


    

    res = list()
    if outputs.shape[1] == 6:
        for data in outputs:
            score = data[1]
            type_ = data[0]
            if score < g_model_config.threshold:
                continue
            if g_model_config.is_yolo:
                data[4] = data[4] - data[2]
                data[5] = data[5] - data[3]
                res.append(PredictResult(*data))
            else:
                h, w, _ = origin_frame.shape
                x = data[2] * w
                y = data[3] * h
                width = data[4]* w - x
                height = data[5] * h - y
                res.append(PredictResult(type_, score, x, y, width, height))
    return res
    

def printResults(frame, predict_result):
    """print result"""
    for box_item in predict_result:
        if len(g_model_config.labels) > 0:
            print("label: {}".format(g_model_config.labels[box_item.type]))
            str_ = "index: {}".format(box_item.type) + ", score: {}".format(box_item.score) \
                  + ", loc: {}".format(box_item.x) + ", {}".format(box_item.y) + ", {}".format(box_item.width) \
                  + ", {}".format(box_item.height)
            print(str_)

def boundaryCorrection(predict_result, width_range, height_range):
    """clip bbox"""
    MARGIN_PIXELS = 2
    predict_result.width = width_range - predict_result.x - MARGIN_PIXELS  \
                           if predict_result.width > (width_range - predict_result.x - MARGIN_PIXELS) \
        else predict_result.width

    predict_result.height = height_range - predict_result.y - MARGIN_PIXELS \
          if predict_result.height > (height_range - predict_result.y - MARGIN_PIXELS) \
        else predict_result.height

    predict_result.x =  MARGIN_PIXELS if predict_result.x < MARGIN_PIXELS else predict_result.x
    predict_result.y = MARGIN_PIXELS if predict_result.y < MARGIN_PIXELS else predict_result.y
    return predict_result

def drawResults(frame, results):
    """draw result"""
    frame_shape = frame.shape
    for r in results:
        r = boundaryCorrection(r, frame_shape[1], frame_shape[0])
        if r.type >= 0 and r.type < len(g_model_config.labels):
            origin = (r.x, r.y+20)
            label_name = g_model_config.labels[r.type]
            cv2.putText(frame, label_name.replace('building_',''), 
                        origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 224) , 1)
            cv2.rectangle(frame, (r.x, r.y), (r.x + r.width, r.y + r.height), (0, 0, 224),1)

            print('name: {}, xs: {} , ys: {} w: {} ,h: {}'.format(
                    label_name,
                    r.x,
                    r.y,
                    r.width,
                    r.height))




def server_tasker(server,port):

    print(f'\nDjango server on {server}\n')

    with os_enter('./') as oe:
        oe.cd('test/dj/mysite')
        if os.name=='nt':
            oe.cmd(f'python manage.py runserver 0.0.0.0:{port}')
        else:
            oe.cmd(f'python3 manage.py runserver 0.0.0.0:{port}')

def get_order(Q_Order,server):
    time.sleep(6)
    while 1:
        time.sleep(1)
        order=easy_request(server+'/order/')
        if order!='NoData':
            print(order,'-----------------')
            Q_Order.put(order)
            #if order=='exit':
            #    return
            if Q_Order.qsize()==5:
                Q_Order.get()
                print('Warning: order num max')

def main_tasker(Q_Order,cfg):
    os.chdir('./Baidu_18_ysl')
    sys.path.insert(0,'./')
    from main import run
    run(Q_Order,cfg)


def data_poster(server):
    n=1
    time.sleep(6)
    while 1:
        time.sleep(1)
        easy_request(server+'/data/',method='POST',data={'msg':('D' if n%2==0 else 'S')+str(n)},
                     header={"Content-type": "application/json"})
        n+=1

if __name__ == "__main__":

    pcs(target=server_tasker,args=[server,port]).start()
    pcs(target=get_order,args=[Q_Order,server]).start()

    print(os.getcwd())
    if len(sys.argv) > 1:
        system_config_path = sys.argv[1]
    else:
        system_config_path = "./dete_model/usb.json" #"../configs/detection/yolov3/image.json"

    print("SystemConfig Path: {}".format(system_config_path))
    g_system_config = SystemConfig(system_config_path)
    model_config_path = g_system_config.model_config_path
    system_config_root = system_config_path[:system_config_path.rfind("/")]
    g_model_config = ModelConfig(os.path.join(system_config_root, model_config_path))

    
    display = Display("PaddleLiteDetectionDemo")
    timer = Timer("Predict", 100)

    cap1=cv2.VideoCapture(cfg.videos[0],cv2.CAP_V4L)
    cap2=cv2.VideoCapture(cfg.videos[1])
    caps=[cap1,cap2]
    mmap('set',caps,arg=[cv2.CAP_PROP_FRAME_WIDTH, 320]) 
    mmap('set',caps,arg=[cv2.CAP_PROP_FRAME_HEIGHT,320])
    
    

    ret = predictorInit()
    if ret != 0:
        print("Error!!! predictor init failed .")
        sys.exit(-1)

    read_cap=1
    update_frame=False

    try:
        while True:
            if Q_Order.qsize():
                order=Q_Order.get()
                if order=='video_sw':
                    read_cap=abs(read_cap-1)
                elif order=='update':
                    update_frame=True
                    
            _,frame=caps[read_cap].read()
            print('read')
            frame=cv2.resize(frame,(160,160))
            w=h=160
            origin_frame = frame.copy()

            predict_result = predict(origin_frame ,g_model_config,g_predictor)
            #predict_result.sort(key=lambda x:x[1])
            #print(predict_result)
#            for _result in predict_result:
#                k,xs=_result
#                print(k,int(xs-w/2),sep='- ',end=' :: ')

            # if g_system_config.predict_log_enable:
            #     printResults(origin_frame, predict_result)
            if update_frame:
                print('update')
                update_frame=False
                drawResults(origin_frame, predict_result)
                os.remove('test/dj/mysite/static/img/tmp.jpg')
                drewlinecross(origin_frame,80,lineWidth=1)
                cv2.imwrite('test/dj/mysite/static/img/tmp.jpg',origin_frame)
                print('write')
                # display.putFrame(origin_frame)
    except:
        import traceback
        print(traceback.print_exc())
        for c in caps:
            c.release()



