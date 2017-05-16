# coding:utf8
import cv2
import numpy as np
from core.plate_locate import PlateLocate
from core.plate_detect import PlateDetect
from core.chars_segment import CharsSegment
from core.chars_recognise import CharsRecognise
from core.chars_identify import CharsIdentify
from core.plate_judge import PlateJudge
from train.net.lenet import Lenet
from train.net.judgenet import Judgenet
from train.cnn_train import eval_model
from train.dataset import DataSet
from util.figs import imshow
from util.read_etc import index2str


def test_plate_locate():
    '''
    车牌定位
    :return:
    '''
    print("Testing Plate Locate")

    file = "resources/image/test.jpg"
    # file = "/Users/huijunzhang/Downloads/pic/pic2.jpg"

    src = cv2.imread(file)

    result = []
    plate = PlateLocate()
    plate.setDebug(False)
    plate.setLifemode(True)

    if plate.plateLocate(src, result) == 0:
        for res in result:
            imshow("plate locate", res)


def test_plate_judge():
    '''
    车牌判断
    :return:
    '''
    print("Testing Plate Judge")

    file = "resources/image/plate_judge.jpg"

    src = cv2.imread(file)

    result = []
    plate = PlateLocate()
    plate.setDebug(False)
    plate.setLifemode(True)

    if plate.plateLocate(src, result) == 0:
        for res in result:
            imshow("plate judge", res)

    judge_result = PlateJudge().judge(np.array(result))

    if len(judge_result) != 0:
        for res in judge_result:
            imshow("plate judge", res)


def test_plate_detect():
    '''
    车牌检测
    :return:
    '''
    print("Testing Plate Detect")

    file = "resources/image/plate_detect.jpg"

    src = cv2.imread(file)

    result = []
    pd = PlateDetect()
    pd.setPDLifemode(True)

    if pd.plateDetect(src, result) == 0:
        for res in result:
            imshow("plate detect", res)


def test_char_segment():
    '''
    字符分隔
    :return:
    '''
    print("Testing Chars Segment")

    file = "resources/image/chars_segment.jpg"

    src = cv2.imread(file)
    imshow("src", src)
    result = []
    plate = CharsSegment()

    if plate.charsSegment(src, result) == 0:
        for i in range(len(result)):
            imshow("plate segment " + str(i), result[i])


def test_chars_identify():
    '''
    字符鉴别
    :return:
    '''
    print("Testing Chars Identify")

    file = "resources/image/chars_identify.jpg"

    src = cv2.imread(file)
    imshow("src", src)
    result = []

    plate = CharsSegment()

    ci = CharsIdentify()

    plate_license = []

    if plate.charsSegment(src, result) == 0:
        plate_license = ci.identify(np.array(result)[..., None])

    pred = ""
    for index in plate_license:
        pred += index2str[index]
    print("Plate License: ", "苏E771H6")
    print("Plate Identify: ", pred)
    if pred == "苏E771H6":
        print("Identify Correct")
    else:
        print("Identify Not Correct")


def test_chars_recognise():
    '''
    字符识别
    :return:
    '''
    print("Testing Chars Recognise")

    file = "resources/image/chars_recognise.jpg"

    src = cv2.imread(file)
    imshow("src", src)

    cr = CharsRecognise()

    print("Chars Recognise: ", cr.charsRecognise(src))


def test_plate_recognize():
    '''
    车牌识别
    :return:
    '''
    print("Testing Plate Recognize")

    file = "resources/image/test.jpg"

    src = cv2.imread(file)
    imshow("src", src)

    result_detection = []
    pd = PlateDetect()
    pd.setPDLifemode(True)
    pd.plateDetect(src, result_detection)

    cr = CharsRecognise()

    for res in result_detection:
        imshow("Plate Recognize", res)
        print("Chars Recognise: ", cr.charsRecognise(res))


def test_cnn_val():
    dataset_params = {
        'batch_size': -1,
        'path': 'resources/train_data/chars',
        'labels_path': 'resources/train_data/chars_list_val.pickle',
        'thread_num': 3,
        'gray': True
    }
    val_dataset_reader = DataSet(dataset_params)

    model = Lenet()
    model.compile()

    image, label = val_dataset_reader.batch()

    print("Total dataset number: ", val_dataset_reader.record_number)

    pred, acc = eval_model([model.pred_labels, model.accuracy],
                           {model.x: image, model.y: label, model.keep_prob: 1},
                           model_dir="train/model/chars/models/")

    print("Label: {}({}), Pred: {}({})".format(label[0], index2str[label[0]], pred[0], index2str[pred[0]]))
    # imshow("tmp", image[0])

    print("Accuary: {:.2f}%".format(acc * 100))


def test_judge_val():
    dataset_params = {
        'batch_size': -1,
        'path': 'resources/train_data/whether_car',
        'labels_path': 'resources/train_data/whether_list_val.pickle',
        'thread_num': 3,
        'gray': False
    }
    val_dataset_reader = DataSet(dataset_params)

    model = Judgenet()
    model.compile()

    image, label = val_dataset_reader.batch()
    print("Total dataset number: ", val_dataset_reader.record_number)

    pred, acc = eval_model([model.pred_labels, model.accuracy],
                           {model.x: image, model.y: label, model.keep_prob: 1},
                           model_dir="train/model/whether_car/models/")

    print("Label: {}, Pred: {}".format(label[0], pred[0]))

    print("Accuary: {:.2f}%".format(acc * 100))


if __name__ == '__main__':
    test_plate_locate()
