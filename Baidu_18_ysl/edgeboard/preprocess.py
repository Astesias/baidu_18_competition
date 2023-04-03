"""preprocess"""
import numpy as np
import cv2
from paddlelite import ImageFormat, TransParam, ImagePreprocess

Format_map = {
    "RGB": ImageFormat.RGB,
    "BGR": ImageFormat.BGR
}


class ImageTransformer(object):
    """PaddleLite wrapped image preprocessing units"""

    def __init__(self, image, mean, scale, dst_shape, dst_format, src_format="BGR"):
        self.read_image(image)
        self.set_dst_format(dst_format)
        self.set_src_format(src_format)
        self.set_shape_config(self.img.shape[:2], dst_shape)
        self.preprocessor = ImagePreprocess(self.src_format, self.dst_format, self.tparam)
        self.set_mean_scale(mean, scale)


    def set_mean_scale(self, mean, scale):
        """
        set mean std
        out = (x / 255 - mean) * scale
        :param mean:
        :param scale:
        :return:
        """
        assert isinstance(mean, list)
        assert len(mean) == 3
        assert isinstance(scale, list)
        assert len(mean) == 3
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(scale, dtype=np.float32)

    def read_image(self, image):
        """set image to be convert, image should be a np.ndarray type"""
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        self.img = image


    def set_dst_format(self, dst_format):
        """
        dst format is the final image format feed into network
        :param dst_format:
        :return:
        """
        self.dst_format = Format_map[dst_format]

    def set_src_format(self, src_format):
        """
        set src format
        :param src_format:
        :return:
        """
        self.src_format = Format_map[src_format]

    def set_shape_config(self, src_shape, dst_shape):
        """

        :param in_shape: original shape
        :param out_shape: final image shape feed into network
        :return:
        """
        assert (src_shape, (list, tuple))
        assert len(src_shape) == 2
        assert (dst_shape, (list, tuple))
        assert len(dst_shape) == 2

        self.tparam = TransParam()
        if src_shape[0] > 1080 or src_shape[1] > 1920:
            self.tparam.ih = dst_shape[0]
            self.tparam.iw = dst_shape[1]
            self.with_cpu_ = True
        else:
            self.tparam.ih = src_shape[0]
            self.tparam.iw = src_shape[1]
            self.with_cpu_ = False
        self.tparam.oh = dst_shape[0]
        self.tparam.ow = dst_shape[1]

    def transforms(self, input_tensor):
        """return a tensor with dst format, dst shape
            and normalized by mean and scale
        """

        if self.with_cpu_:
            # fpga preprocess has limitation in height and width
            # do resize in python side first
            self.img = cv2.resize(self.img, (self.tparam.iw, self.tparam.ih))

        # TODO sparse array --> dense array transform will be moved to c++ side later
        dense_img = np.zeros((self.tparam.ih, self.tparam.iw, 3)).astype(self.img.dtype)

        dense_img[0: self.tparam.ih, 0: self.tparam.iw, :] = self.img
        self.img = dense_img
        # PaddleLite feed should be C,H,W to avoid first layout transform

        input_tensor.resize((1, 3, self.tparam.oh, self.tparam.ow))
        # img should be continuous
        self.preprocessor.image2Tensor(self.img, input_tensor, self.mean, self.std)
        return input_tensor
