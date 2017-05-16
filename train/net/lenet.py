from core.base import Singleton
from .layer import *

import tensorflow as tf

class Lenet(Singleton):

    def __init__(self):
        tf.reset_default_graph()

        self.num_classes = 65

        self.x = None
        self.y = None
        self.keep_prob = None

        self.pred_logits = None
        self.pred_labels = None

        self.accuracy = None

        self.l2_loss = None

        self.weights = []
        self.biases = []

    def compile(self):

        self.keep_prob = tf.placeholder(tf.float32)

        self.weights = []
        self.biases = []

        input = ImageLayer(20, 20, 1)
        label = LabelLayer()

        convpools1 = ConvPoolLayer(input, 5, 5, 20, 2, 2, layer_id=1)

        convpools2 = ConvPoolLayer(convpools1, 5, 5, 50, 2, 2, layer_id=2)

        dp = DropoutLayer(convpools2, self.keep_prob)

        flatten = FlattenLayer(dp)
        ip1 = DenseLayer(flatten, 500, layer_name="DENSE1")
        self.weights += ip1.weights
        self.biases += ip1.biases
        ip1_relu = ActivationLayer(ip1)

        pred = OutputLayer(ip1_relu, self.num_classes)
        self.weights += pred.weights
        self.biases += pred.biases

        self.x = input.output
        self.y = label.output
        self.pred_logits = pred.output
        self.pred_labels = tf.argmax(self.pred_logits, 1)

        correct_pred = tf.equal(self.pred_labels, self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        for w in self.weights + self.biases:
            l2_loss = tf.nn.l2_loss(w)
            if self.l2_loss is None:
                self.l2_loss = l2_loss
            else:
                self.l2_loss += l2_loss
