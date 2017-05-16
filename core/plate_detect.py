import cv2
import numpy as np

from .base import Plate
from .plate_locate import PlateLocate
from .plate_judge import PlateJudge

from .core_func import *
from util.figs import imwrite, imshow

class PlateDetect(object):
    def __init__(self):
        self.m_plateLocate = PlateLocate()
        self.m_maxPlates = 3

    def setPDLifemode(self, param):
        self.m_plateLocate.setLifemode(param)

    def plateDetect(self, src, res, index=0):
        color_plates = []
        sobel_plates = []
        color_result_plates = []
        sobel_result_plates = []

        self.m_plateLocate.plateColorLocate(src, color_plates, index)

        for plate in color_plates:
            color_result_plates.append(plate.plate_image)

        if len(color_result_plates) != 0:
            color_result_plates = PlateJudge().judge(np.array(color_result_plates))

        for plate in color_result_plates:
            res.append(plate)

        if len(res) > self.m_maxPlates:
            return 0

        self.m_plateLocate.plateSobelLocate(src, sobel_plates, index)

        for plate in sobel_plates:
            sobel_result_plates.append(plate.plate_image)

        if len(sobel_result_plates) != 0:
            sobel_result_plates = PlateJudge().judge(np.array(sobel_result_plates))

        for plate in sobel_result_plates:
            res.append(plate)

        return 0




