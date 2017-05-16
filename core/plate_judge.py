from .base import Singleton

from train.cnn_train import eval_model
from train.net.judgenet import Judgenet

import numpy as np

class PlateJudge(Singleton):

    def __init__(self):

        self.model = Judgenet()
        self.model.compile()
        self.eval_sess = None

    def judge(self, images):
        judgeRes = []
        tmp = images / 255 * 2 - 1

        pred = eval_model(self.model.pred_labels,
                           {self.model.x: tmp, self.model.keep_prob: 1},
                           model_dir="train/model/whether_car/models/",
                          eval_sess=self.eval_sess)

        for i, tmp in enumerate(pred):
            if tmp == 1:
                judgeRes.append(images[i])

        return judgeRes
