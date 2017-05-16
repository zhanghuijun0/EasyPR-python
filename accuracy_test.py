#coding:utf8
import os
import cv2
import time
import numpy as np

from core.plate_detect import PlateDetect
from core.chars_recognise import CharsRecognise

from util.figs import imshow

def accuracy_test(dir):
    print("Begin to test accuracy")
    # total test, not recognize, match count
    count = [0, 0, 0]
    not_recognized_names = []
    image_names = os.listdir(dir)
    starttime = time.time()

    for image_name in image_names:
        print('-'*8)
        count[0] += 1
        label = image_name.split('.')[0]
        src = cv2.imdecode(np.fromfile(os.path.join(dir, image_name), dtype=np.uint8), cv2.IMREAD_COLOR)
        #src = cv2.imdecode(np.fromfile(os.path.join(dir, '川AEK882.jpg'), dtype=np.uint8), cv2.IMREAD_COLOR)
        result_detection = []
        pd = PlateDetect()
        pd.setPDLifemode(True)
        pd.plateDetect(src, result_detection)

        print("Label: ", label)
        if len(result_detection) == 0:
            not_recognized_names.append(image_name)
            count[1] += 1
            print('-' * 8)
            continue

        cr = CharsRecognise()
        ismatch = False

        for res in result_detection:
            pred_license = cr.charsRecognise(res)
            print("Chars Recognise: ", pred_license)
            if label == pred_license:
                ismatch = True

        if ismatch == True:
            count[2] += 1

        print('-' * 8)

    endtime = time.time()
    print("Accuracy test end!")
    print("Summary:")
    print("Total images: ", count[0])
    print("Total time: {:.2f}, Average time: {:.2f}".format(endtime - starttime, (endtime - starttime) / count[0]))
    print("Match rate: {:.2f}%({}), locate rate: {:.2f}%".format(count[2] / count[0] * 100, count[2],
                                                                 1 - count[1] / count[0] * 100))
    print("Not recognize: ")
    for pic in not_recognized_names:
        print(pic)
