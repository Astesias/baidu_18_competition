"""prodictor"""
import numpy as np
from paddlelite import Place
from paddlelite import CxxConfig
from paddlelite import CreatePaddlePredictor
from paddlelite import TargetType
from paddlelite import PrecisionType
from paddlelite import DataLayoutType


class PaddleLitePredictor(object):
    """ PaddlePaddle interface wrapper """

    def __init__(self):
        self.predictor = None

    def set_model_file(self, path):
        """model file"""
        self.model_file = path

    def set_param_file(self, path):
        """param file"""
        self.param_file = path

    def set_model_dir(self, path):
        """model dir"""
        self.model_dir = path

    def load(self):
        """load"""
        valid_places = (
            Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
            Place(TargetType.kHost, PrecisionType.kFloat),
            Place(TargetType.kARM, PrecisionType.kFloat),
        )
        config = CxxConfig()
        if self.param_file is not None:
            config.set_model_file(self.model_file)
            config.set_param_file(self.param_file)
        else:
            config.set_model_dir(self.model_dir)
        config.set_valid_places(valid_places)
        self.predictor = CreatePaddlePredictor(config)

    def get_input(self, index):

        """get input"""
        return self.predictor.get_input(index)

    def set_input(self, data, index):
        """set input"""
        if isinstance(data, np.ndarray):
            input = self.predictor.get_input(index)
            input.resize(data.shape)
            input.set_data(data.copy())

    def run(self):
        """run"""
        self.predictor.run();

    def get_output(self, index):
        """get result"""
        return self.predictor.get_output(index)
