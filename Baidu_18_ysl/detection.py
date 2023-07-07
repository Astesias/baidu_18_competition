"""detection demo"""
import os
import cv2
import sys
import numpy as np
from edgeboard import PaddleLitePredictor
from core import Display,SystemConfig,ModelConfig,Timer
from core import fpga_preprocess,cpu_preprocess,createCapture

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
    def unpack(self,classes):
        return [classes[self.type],self.x,self.y,self.width,self.height]

def predict_result(type_, score, x, y, width, height):
    return int(type_),int(x)+int(width/2) #int(x),int(y),int(width),int(height)


def predictorInit(g_model_config,g_predictor):
    """predictor config and initialization"""
    # global g_model_config
    # global g_system_config
    # global g_predictor

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

    outputs=outputs[np.argsort(-outputs[:, 1])] ######## sort result

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
                res.append(predict_result(*data))
            else:
                h, w, _ = origin_frame.shape
                x = data[2] * w
                y = data[3] * h
                width = data[4]* w - x
                height = data[5] * h - y
                res.append(predict_result(type_, score, x, y, width, height))
    return res
    

def printResults(frame, predict_result,g_model_config):
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

def drawResults(frame, results,g_model_config):
    """draw result"""
    frame_shape = frame.shape
    for r in results:
        r = boundaryCorrection(r, frame_shape[1], frame_shape[0])
        if r.type >= 0 and r.type < len(g_model_config.labels):
            origin = (r.x, r.y)
            label_name = g_model_config.labels[r.type]
            cv2.putText(frame, label_name, origin, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 224), 2)
            cv2.rectangle(frame, (r.x, r.y), (r.x + r.width, r.y + r.height), (0, 0, 224), 2)


def detection_init(system_config_path):
    g_predictor = PaddleLitePredictor()
    g_system_config=SystemConfig(system_config_path)
    print("Config Path: {}".format(system_config_path))
    model_config_path=g_system_config.model_config_path
    system_config_root=system_config_path[:system_config_path.rfind("/")]
    g_model_config = ModelConfig(os.path.join(system_config_root, model_config_path))
        
    

    #timer = Timer("Predict", 100)
    #capture = createCapture(config.input_type,config.input_path)
    #if capture is None:
    #    exit(-1)
    
    display = Display("PaddleLiteDetectionDemo")
    ret = predictorInit(g_model_config,g_predictor)
    if ret != 0:
        print("Error!!! predictor init failed .")
        sys.exit(-1)

    return display,g_model_config,g_predictor



if __name__ == "__main__":
    g_predictor = PaddleLitePredictor()

    if len(sys.argv) > 1:
        system_config_path = sys.argv[1]
    else:
        system_config_path = "../test/face_model/usb_yolov3.json" #"../configs/detection/yolov3/image.json"

    print("SystemConfig Path: {}".format(system_config_path))

 
    g_system_config = SystemConfig(system_config_path)
    model_config_path = g_system_config.model_config_path
    system_config_root = system_config_path[:system_config_path.rfind("/")]
    g_model_config = ModelConfig(os.path.join(system_config_root, model_config_path))

  
    display = Display("PaddleLiteDetectionDemo")
    timer = Timer("Predict", 100)
    capture = createCapture(g_system_config.input_type, g_system_config.input_path)
    if capture is None:
        exit(-1)
    ret = predictorInit(g_model_config,g_predictor)
    if ret != 0:
        print("Error!!! predictor init failed .")
        sys.exit(-1)


    while True:

   
        frame = capture.getFrame()
        # print(frame)
        origin_frame = frame.copy()
        predict_result = predict(frame,g_model_config,g_predictor, timer)
        drawResults(origin_frame, predict_result,g_model_config)


        if g_system_config.predict_log_enable:
            printResults(origin_frame, predict_result,g_model_config)
            pass
        if g_system_config.predict_log_enable and capture.getType() != "image":
            display.putFrame(origin_frame)
        elif capture.getType() == "image":
            cv2.imwrite("DetectionResult.jpg", origin_frame)
            break

        if g_system_config.predict_time_log_enable:
    
            #timer.printAverageRunTime()
            pass

    if g_system_config.display_enable and g_system_config.input_type != "image":
        display.stop()

    capture.stop()
