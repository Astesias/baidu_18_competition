# -*- coding:utf-8 编码测试
import numpy as np
from edgeboard import *
import sys
from core import *
import cv2

g_predictor = PaddleLitePredictor()


class PredictResult(object):
    """result wrapper"""
    def __init__(self, score, index):
        """init"""
        self.score = score
        self.index = index

global cr
cr=None

def predictorInit():
    """predictor config and initialization"""
    global g_model_config
    global g_system_config
    global cr

    global g_predictor
    if g_model_config.is_combined_model:
        g_predictor.set_model_file(g_model_config.model_file)
        g_predictor.set_param_file(g_model_config.params_file)
    else:
        # TODO add not combined model load
        pass
    try:
        g_predictor.load()
        cr = Cruiser(g_predictor)
        print("Predictor Init Success !!!")
        return 0
    except:
        print("Error: CreatePaddlePredictor Failed.")
        return -1


IS_FIRST_RUN = True


cnn_args = {
    "shape": [1, 3, 128, 128],
    "ms": [125.5, 0.00392157]
}
class Cruiser:
    args = cnn_args

    def __init__(self,predictor):

        self.predictor=predictor
        hwc_shape = list(self.args["shape"])
        hwc_shape[3], hwc_shape[1] = hwc_shape[1], hwc_shape[3]
        self.buf = np.zeros(hwc_shape).astype('float32')
        self.size = self.args["shape"][2]
        self.means = self.args["ms"]

    # CNN网络的图片预处理
    def image_normalization(self, image):
        image = cv2.resize(image, (self.size, self.size))

        image = image.astype(np.float32)
        #print(image.shape)
        if 1: #platform.system() == "Windows":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))  # HWC to CHW
        # 转换成BGR
        img = (image - self.means[0]) * self.means[1]
        img = img.reshape([1, 3, 128, 128])

        return img

    # CNN网络预测
    def infer_cnn(self, image):
        data = self.image_normalization(image)
        self.predictor.set_input(data, 0)

        self.predictor.run()
        out = self.predictor.get_output(0)
        print(np.array(out))
        return np.array(out)[0][0]

    def cruise(self, image):
        res = self.infer_cnn(image)
        # print(res)
        return res


def predict(frame, timer):
    
    if not g_system_config.use_fpga_preprocess:
        input_data = cpu_preprocess(frame, g_model_config)
        g_predictor.set_input(input_data, 0)
    else:
        input_tensor = g_predictor.get_input(0)
        fpga_preprocess(frame, input_tensor, g_model_config)
    global IS_FIRST_RUN
    if IS_FIRST_RUN:
        IS_FIRST_RUN = False
        g_predictor.run()
    else:
        timer.Continue()
        g_predictor.run()
        timer.Pause()
    output = np.array(g_predictor.get_output(0))

    max_index = np.argmax(output, axis=1)[0]
    score = output[0, max_index]
    return PredictResult(score, max_index)


def printResults(frame, predict_result):
    """print result"""
    if len(g_model_config.labels) > 0:
        print("label: {}".format(g_model_config.labels[predict_result.index]))
        str = "index: {}".format(predict_result.index) + " " + "score: {}".format(predict_result.score)
        print(str)


def drawResults(frame, predict_result):
    """draw result"""
    if predict_result.score > g_model_config.threshold:
        text = g_model_config.labels[predict_result.index] + " " + str(predict_result.score)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 224), 2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        system_config_path = sys.argv[1]
    else:
        system_config_path = "../configs/classification/resnet50/image.json"            #配置路径

    print("SystemConfig Path: {}".format(system_config_path))

    g_system_config = SystemConfig(system_config_path)                                  #配置列表
    model_config_path = g_system_config.model_config_path
    system_config_root = system_config_path[:system_config_path.rfind("/")]
    g_model_config = ModelConfig(os.path.join(system_config_root, model_config_path))   #模型路径
    display = Display("PaddleLiteClassificationDemo")
    timer = Timer("Predict", 100)                                                       #定时器?
    capture = createCapture(g_system_config.input_type, g_system_config.input_path)     #usb视频流或图片
    if capture is None:
        exit(-1)

    ret = predictorInit()                                                               #识别器
    if ret != 0:
        print("Error!!! predictor init failed .")
        sys.exit(-1)
    while True:
        frame = capture.getFrame()            
        result = cr.cruise(frame)
        print(result)
        continue
        
                             
        predict_result = predict(frame, timer)                                          #定时预测
        print(predict_result.score,predict_result.index)
        if g_system_config.predict_log_enable:
            printResults(frame, predict_result)

        if g_system_config.display_enable and capture.getType() != "image":
            display.putFrame(frame)
        elif g_system_config.display_enable and capture.getType() == "image":
            drawResults(frame, predict_result)
            cv2.imwrite("clssificationResult.jpg", frame)
            break

        if g_system_config.predict_time_log_enable:                                     #定时器log
            timer.printAverageRunTime()

    if g_system_config.display_enable and g_system_config.input_type != "image":
        display.stop()                                                                  #检测文件停止?

    capture.stop()
