"""display module"""
import cv2


class Display(object):
    """display class"""
    def __init__(self, name):
        """init"""
        self.name_ = name

    def putFrame(self, frame):
        """show result on the monitor"""
        cv2.imshow(self.name_, frame)
        cv2.waitKey(10)

    def stop(self):
        """stop"""
        pass
