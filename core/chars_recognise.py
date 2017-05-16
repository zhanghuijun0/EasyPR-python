from .chars_segment import CharsSegment
from .chars_identify import CharsIdentify

from util.read_etc import index2str

import numpy as np

class CharsRecognise(object):
    def __init__(self):
        self.charsSegment = CharsSegment()

    def charsRecognise(self, plate):
        chars = []
        result = self.charsSegment.charsSegment(plate, chars)

        temp = []
        plate_license = ""

        if result == 0:
            temp = CharsIdentify().identify(np.array(chars)[..., None])

        for index in temp:
            plate_license += index2str[index]

        return plate_license