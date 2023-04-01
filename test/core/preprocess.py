"""cpu preprocess"""
import numpy as np
import cv2
from edgeboard import ImageTransformer


def cpu_preprocess(frame, model_config):
    """cpu preprocess"""
    means = np.array(model_config.means)[np.newaxis, np.newaxis, :]
    scales = np.array(model_config.scales)[np.newaxis, np.newaxis, :]
    input_h = model_config.input_height
    input_w = model_config.input_width
    resized_frame = cv2.resize(frame, (input_w, input_h))
    if model_config.format == "RGB":
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    resized_frame = resized_frame.astype(np.float32)
    resized_frame -= means
    resized_frame *= scales
    feed_frame = np.zeros((1, input_h, input_w, 3), dtype=np.float32)
    feed_frame[0, 0: input_h, 0: input_w, :] = resized_frame
    feed_frame = feed_frame.reshape((1, 3, input_h, input_w))
    return feed_frame

def fpga_preprocess(frame, input_tensor, model_config):
    """fpga preprocess"""
    means = model_config.means
    scales = model_config.scales
    dst_shape = [model_config.input_height, model_config.input_width]
    dst_format = model_config.format
    image_transformer = ImageTransformer(frame, means, scales, dst_shape, dst_format)
    image_transformer.transforms(input_tensor)
