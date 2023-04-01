"""capture class"""
import cv2
from .common import FrameWrapper


class CaptureInterface(object):
    """base class of capture"""
    def start(self):
        """start"""
        raise NotImplementedError

    def stop(self):
        """stop"""
        raise NotImplementedError

    def getType(self):
        """getType"""
        raise NotImplementedError

    def getFrame(self):
        """getFrame"""
        raise NotImplementedError

    def run(self):
        """run"""
        raise NotImplementedError


class USBCamera(CaptureInterface):
    """usb cam wrapper"""
    def __init__(self, type_, dev):
        """init"""
        self.dev_type = type_
        self.dev_name = dev
        # dev_ = int(dev.split("/")[-1][5:])
        dev_ = dev
        self.cap = cv2.VideoCapture(dev_,cv2.CAP_V4L)

    def start(self):
        """start"""
        self.cap = cv2.VideoCapture(self.dev_name)
        if self.cap is None:
            return -1
        return 0

    def stop(self):
        """stop"""
        self.cap.release()
        return 0

    def getFrame(self):
        """getFrame"""
        _, frame = self.cap.read()
        return frame

    def getType(self):
        """getType"""
        return self.dev_type



class ImageReader(CaptureInterface):
    """Image wrapper"""
    def __init__(self, type_, dir_):
        """init"""
        self.type = type_
        self.dir = dir_
        self.image = None

    def start(self):
        """start"""
        self.image = cv2.imread(self.dir)
        if self.image is not None:
            return -1
        return 0

    def stop(self):
        """stop"""
        pass

    def getFrame(self):
        """getFrame"""
        self.image = cv2.imread(self.dir)
        return self.image

    def getType(self):
        """getType"""
        return self.type


def createCapture(type_, path):
    """create capture object"""
    if type_ == "usb_camera":
        capture = USBCamera(type_, path)
    elif type_ == "image":
        capture = ImageReader(type_, path)
    else:
        raise ValueError("Error !!!! createCapture: unsupport capture tyep: {}".format(type))
    return capture
