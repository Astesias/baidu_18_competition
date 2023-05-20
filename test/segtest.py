"""segmentation demo"""
import numpy as np
from edgeboard import *
import sys
from cor import *
import cv2,os
from fline import FollowLine

g_predictor = PaddleLitePredictor()


class PredictResult(object):
    """result wrapper"""
    def __init__(self, score, index):
        """init"""
        self.score = score
        self.index = index


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


IS_FIRST_RUN = True


def predict(frame, timer):
    """predict with paddlelite and postprocess"""
    if 0:#not g_system_config.use_fpga_preprocess:
        input_data = cpu_preprocess(frame, g_model_config)
        g_predictor.set_input(input_data, 0)
    else:
        input_tensor = g_predictor.get_input(0)
        print('input',frame.shape)
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
    print('output:',output.shape)
    
    _, channel, height, width = output.shape
    segMat = np.argmax(output[0, :, :, :], axis=0)
    
    segMat[segMat > 0] = 255
    
    
    channel=1
    
    
    output=output[0, channel, :, :]
    avg=sum(sum(output))/output.shape[0]/output.shape[1]
    thd=float(sys.argv[2]) if len(sys.argv)>=3 else 0.01
    print(output)
    print(output.shape)
    output[output > thd] = 255
    output[output < thd] = 0
    print(output)
    print(avg)
    
    #_, channel, height, width = output.shape
    #segMat = np.argmax(output[0, :, :, :], axis=0)
    
    #segMat[segMat > 0] = 255
    #print('result',segMat.shape)
    #print(sum(segMat))
    return output


def drawMask(frame, segMat):
    """print result"""
    mask_mat = cv2.copyMakeBorder(segMat, 0, 0, 0, 0, cv2.BORDER_CONSTANT, (127.5, 127.5, 127.5)).astype(np.float32)
    origin_h, origin_w, _ = frame.shape
    origin_croped_mat = cv2.resize(mask_mat, (origin_w, origin_h), cv2.INTER_NEAREST).astype(np.uint8)

    for h in range(origin_h):
        for w in range(origin_w):
            if origin_croped_mat[h, w] > 0:
                frame[h, w, :] = [128, 255, 255]
def drawMask2(frame, segMat):
    frame[segMat==255]=[128, 255, 255]

if __name__ == "__main__":
    f=FollowLine()
    
    #if len(sys.argv) > 1:
    #    system_config_path = sys.argv[1]
    #else:
    system_config_path = ".ignore/image.json"

    print("SystemConfig Path: {}".format(system_config_path))

    g_system_config = SystemConfig(system_config_path)
    model_config_path = g_system_config.model_config_path
    system_config_root = system_config_path[:system_config_path.rfind("/")]
    g_model_config = ModelConfig(os.path.join(system_config_root, model_config_path))
    display = Display("PaddleLiteSegmentationDemo")
    timer = Timer("Predict", 100)
    capture = createCapture(g_system_config.input_type, g_system_config.input_path)
    if capture is None:
        exit(-1)

    ret = predictorInit()
    if ret != 0:
        print("Error!!! predictor init failed .")
        sys.exit(-1)
    aa=1
    while aa:
        aa=0
        frame = cv2.imread(sys.argv[1])
        print(frame.shape,frame.dtype)
        predict_result = predict(frame, timer)
        print(f.GetValue(predict_result))
        

        if g_system_config.predict_log_enable and capture.getType() != "image":
            display.putFrame(frame)
        elif capture.getType() == "image":
            cv2.imwrite("segmentationResult.jpg", predict_result.transpose(0,1))
            drawMask2(frame, predict_result)
            cv2.imwrite("vis.jpg", frame)
            os.rename("vis.jpg","vis_os.jpg")
            os.rename("segmentationResult.jpg","segmentationResult_os.jpg")
            cv2.imshow('1',frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pass
            break

        if g_system_config.predict_time_log_enable:
            timer.printAverageRunTime()
