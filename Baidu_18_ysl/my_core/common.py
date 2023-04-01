# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:33:51 2022

@author: 鞠秉宸
"""


import os
import time
import json
import numpy as np
from edgeboard import *

def compercol(f,ff):
    thres=20
    a=int(f[0])
    b=int(ff[0])
    c=abs(a-b)
    if c>thres:
        return 1
    else:
        return 0

def get_cubenumber(centerlist,img):
    centerlist=sorted(centerlist,key=lambda s:s[0])
    reslutbb=[[1,1,0],[1,0,1],[0,1,1],[0,0,0]]
    b=len(centerlist)
    reslutll=[]
    if(b>=3):
        Labimg=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        one=Labimg[centerlist[0][1],centerlist[0][0]]
        
        two=Labimg[centerlist[1][1],centerlist[1][0]]
        three=Labimg[centerlist[2][1],centerlist[2][0]]
        reslutll.append(compercol(one,two))
        reslutll.append(compercol(one,three))
        reslutll.append(compercol(two,three))
        for i in range(0,4):
            if reslutll==reslutbb[i]:
                return i+1
            if i==4:
                return -1
    return -1

class ModelConfig(object):
    def __init__(self,dir):
        self.model_parrent_path = dir
        with open(dir+"/config.json") as f:
            value = json.load(f)
        self.width=value["input_width"]
        self.height=value["input_height"]
        self.format = value["format"]
        self.means = value["mean"]
        self.scales = value["scale"]
        if "threshold" in value:
            self.threshold = value["threshold"]
        else:
            self.threshold = 0.00000001
            
        self.is_combined_model = True
        
        if "network_type" in value:
            self.network_type = value["network_type"]
            self.is_yolo = self.network_type == "YOLOV3"
        self.model_file = os.path.join(self.model_parrent_path, value["model_file_name"])
        self.params_file = os.path.join(self.model_parrent_path, value["params_file_name"])
        
        self.labels = list()
        if "labels_file_name" in value:
            label_path = os.path.join(self.model_parrent_path, value["labels_file_name"])
            with open(label_path, "r") as f:
                while True:
                    line = f.readline().strip()
                    if line == "":
                        break
                    self.labels.append(line)        
                    
class my_result(object):
    def __init__(self,model_config,result):
        if(result.type==-1):
            self.label = "NoFind"
        else:
            self.label= model_config.labels[result.type]
        self.x = int(result.x+result.width*0.5)
        self.y = int(result.y+result.height*0.5)
        self.direct = ""
            
        
      

class PredictResult(object):

    def __init__(self, category, score, x, y, width, height):
        self.type = int(category)
        self.score = score
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        

class Predict(object):
    def __init__(self,model_config):
        self.predictor = PaddleLitePredictor()
        self.result = list()
        self.model_config = model_config
        
    def load(self):
        if self.model_config.is_combined_model:
            self.predictor.set_model_file(self.model_config.model_file)
            self.predictor.set_param_file(self.model_config.params_file)
        self.predictor.load()
        #print("Predictor Init Success !!!")
        
    def fpga_preprocess(self,frame,input_tensor):
        means = self.model_config.means
        scales = self.model_config.scales
        dst_shape = [self.model_config.height, self.model_config.width]
        dst_format = self.model_config.format
        image_transformer = ImageTransformer(frame, means, scales, dst_shape, dst_format)
        image_transformer.transforms(input_tensor)
    
    def predict_FengGe(self,frame):
        origin_frame  = frame.copy()
        origin_h, origin_w, _ = origin_frame.shape
    
        input_tensor = self.predictor.get_input(0)
        self.fpga_preprocess(frame, input_tensor)
    
        
        self.predictor.run()
    
        return np.array(self.predictor.get_output(0))

    
    def predict1(self,frame):
        origin_frame  = frame.copy()
        origin_h, origin_w, _ = origin_frame.shape
    
        input_tensor = self.predictor.get_input(0)
        self.fpga_preprocess(frame, input_tensor)
    
        if self.model_config.is_yolo:
            feed_shape = np.zeros((1, 2), dtype=np.int32)
            feed_shape[0, 0] = origin_h
            feed_shape[0, 1] = origin_w
            shape_tensor = self.predictor.set_input(feed_shape, 1)
        
        self.predictor.run()
    
        outputs = np.array(self.predictor.get_output(0))
        
        self.result = list()
        if(outputs.size>=6):
            for data in outputs:
                score = data[1]
                type_ = data[0]
                if score < self.model_config.threshold:
                    continue
                if self.model_config.is_yolo:
                    data[4] = data[4] - data[2]
                    data[5] = data[5] - data[3]
                    self.result.append(PredictResult(*data))
                else:
                    h, w, _ = origin_frame.shape
                    x = data[2] * w
                    y = data[3] * h
                    width = data[4]* w - x
                    height = data[5] * h - y
                    self.result.append(PredictResult(type_, score, x, y, width, height))
          
        return self.result
    
    def printResult(self,result):
        if(result.label!="NoFind"):
            print("label:",result.label,"  x:",result.x,"  y:",result.y)
            
    def printResults(self,result):
        for box_item in result:
            if len(self.model_config.labels) > 0:
                str1 = "label:{}/".format(self.model_config.labels[box_item.type])
                str2 = "x:{}/".format(box_item.x) + "y:{}/".format(box_item.y) + "width:{}/".format(box_item.width) \
                      + "height:{}/".format(box_item.height)
                print(str1+str2)
                

