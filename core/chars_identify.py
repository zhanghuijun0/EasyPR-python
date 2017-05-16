from .base import Singleton

from train.cnn_train import eval_model
from train.net.lenet import Lenet

import numpy as np

class CharsIdentify(Singleton):

    def __init__(self):

        self.model = Lenet()
        self.model.compile()
        self.eval_sess = None

    def identify(self, images):
        tmp = images / 255 * 2 - 1
        pred = eval_model(self.model.pred_labels,
                           {self.model.x: tmp, self.model.keep_prob: 1},
                           model_dir="train/model/chars/models/",
                            eval_sess=self.eval_sess)

        return pred