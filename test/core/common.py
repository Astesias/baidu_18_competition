"""common utils"""
import os
import time
import json


class FrameWrapper(object):
    """frame wrapper"""
    def __init__(self, frame, channel):
        """init"""
        self.frame  = frame
        self.channel = channel

def assert_check_file_exist(dir_name, file_name):
    """file existence check"""
    assert os.path.exists(os.path.join(dir_name, file_name)), \
        "Error!!!! ModelConfig: File {} not exit, Please Check your model".format(file_name)


class ModelConfig(object):
    """model config class"""

    def __init__(self, model_path):
        """init"""
        self.model_parrent_path = model_path
        json_config_path = os.path.join(model_path, "config.json")
        assert os.path.exists(json_config_path), "Error: ModelConfig file path: {} not found".format(json_config_path)
        with open(json_config_path) as f:
            value = json.load(f)
        print("Config:".format(value))
        self.input_width = value["input_width"]
        self.input_height = value["input_height"]
        self.format = value["format"]
        self.means = value["mean"]
        self.scales = value["scale"]

        if "threshold" in value:
            self.threshold = value["threshold"]
        else:
            self.threshold = 0.5
            print("Warnning !!!!,json key: threshold not found, default : {}".format(self.threshold))
        self.is_yolo = False

        if "network_type" in value:
            self.network_type = value["network_type"]
            self.is_yolo = self.network_type == "YOLOV3"

        if "model_file_name" in value and "params_file_name" in value:
            self.is_combined_model = True
        elif "model_file_name" in value and "params_file_name" in value:
            self.is_combined_model = False
        else:
            raise ValueError(
                "json config Error !!!! "
                "combined_model: need params_file_name model_file_name, "
                "separate_model: need model_dir only.")

        if self.is_combined_model:
            assert_check_file_exist(self.model_parrent_path, value["model_file_name"])
            self.model_file = os.path.join(self.model_parrent_path, value["model_file_name"])
            assert_check_file_exist(self.model_parrent_path, value["params_file_name"])
            self.params_file = os.path.join(self.model_parrent_path, value["params_file_name"])
            self.model_params_dir = ""
        else:
            assert_check_file_exist(self.model_parrent_path, value["model_dir"])
            self.model_params_dir = os.path.join(self.model_parrent_path, value["model_dir"])
            self.params_file = ""
            self.model_file = ""

        self.labels = list()
        if "labels_file_name" in value:
            label_path = os.path.join(self.model_parrent_path, value["labels_file_name"])
            assert os.path.exists(label_path), "Open Label File failed, file path: {}".format(label_path)
            with open(label_path, "r") as f:
                while True:
                    line = f.readline().strip()
                    if line == "":
                        break
                    self.labels.append(line)


class SystemConfig(object):
    """system config"""
    def __init__(self, dir):
        """init"""
        self.config_dir = dir
        config_root = dir[:dir.rfind("/")]
        assert os.path.exists(dir), "Error:SystemConfig file path:[{}] not found".format(dir)
        with open(dir, "r") as f:
            value = json.load(f)
        print("SystemConfig: {}".format(value))
        self.model_config_path = value["model_config"]
        input_config = value["input"]
        self.input_type = input_config["type"]
        if "path" in input_config:
            if self.input_type == "image":
                self.input_path = os.path.join(config_root, input_config["path"])
            else:
                self.input_path = input_config["path"]

        self.use_fpga_preprocess = True if "fpga_preprocess" not in value else value["fpga_preprocess"]

        if "debug" in value:
            debug_config = value["debug"]
            self.predict_time_log_enable = True if "predict_time_log_enable" not in debug_config \
                else debug_config["predict_time_log_enable"]
            self.predict_log_enable = True if "predict_log_enable" not in debug_config \
                else debug_config["predict_log_enable"]
            self.display_enable = True if "display_enable" not in debug_config \
                else debug_config["display_enable"]
        else:
            self.predict_log_enable = True
            self.predict_time_log_enable = True
            self.display_enable = True

        print("SystemConfig Init Success !!!")


class Timer(object):
    """timer"""
    def __init__(self, name, maxRecordCounts=30):
        """init"""
        self.name_ = name
        self.max_record_counts_ = maxRecordCounts
        self.start_time_ = 0
        self.cur_counts_ = 0
        self.sum = 0

    def Continue(self):
        """start or continue"""
        if self.cur_counts_ <= self.max_record_counts_:
            self.start_time_ = time.time()

    def Pause(self):
        """temp stop"""
        if self.cur_counts_ <= self.max_record_counts_:
            stop_time = time.time()
            diff_time = stop_time - self.start_time_
            self.sum += diff_time
            self.cur_counts_ += 1

    def printAverageRunTime(self):
        """print info"""
        if self.cur_counts_ >= 1:
            str = "{} Total Record".format(self.name_) + \
                  " {} times, Cur".format(self.max_record_counts_) + \
                  "{}".format(self.cur_counts_) + \
                  " times, Use {}, Average".format(self.sum) + \
                  " {}".format(self.sum / self.cur_counts_)
            print(str)
