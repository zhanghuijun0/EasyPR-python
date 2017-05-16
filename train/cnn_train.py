import datetime
import os

from core.base import Singleton
from train.net.layer import *
from .dataset import DataSet
from .net.lenet import Lenet


class Train(object):

    # __metaclass__ = Singleton

    def __init__(self, params):
        self.learning_rate = params['lr']
        self.number_epoch = params['number_epoch']
        self.epoch_length = params['epoch_length']
        self.log_dir = params['log_dir']
        print("lr: {}, number_epochs: {}, epoch_length: {}, max_steps: {}".format(
            self.learning_rate, self.number_epoch, self.epoch_length, int(self.epoch_length * self.number_epoch)
        ))

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.model = None

        self.pred_labels = None
        self.loss = None
        self.total_loss = None
        self.l2_loss = None

        self.sess = None

        self.weights = []
        self.biases = []

    def compile(self, model):
        self.model = model

        self.pred_logits = self.model.pred_logits
        self.pred_labels = self.model.pred_labels
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.pred_logits, labels=self.model.y))
        tf.summary.scalar('loss', self.loss)

        self.total_loss = self.loss + 5e-4 * self.model.l2_loss

        correct_pred = tf.equal(self.pred_labels, self.model.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.__add_optimal()

    def __add_optimal(self):
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        train_op = optimizer.minimize(self.total_loss)

        self.train_op = train_op

    def train(self, train_dataset, val_dataset):
        if self.sess is None:
            self.sess = tf.Session()

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.log_dir + "/summary/train", self.sess.graph)
        test_writer = tf.summary.FileWriter(self.log_dir + "/summary/test")
        model_dir = self.log_dir + 'models/'

        saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for step in range(int(self.epoch_length * self.number_epoch)):

            train_x, train_y = train_dataset.batch()
            feed_dict = {self.model.x: train_x, self.model.y: train_y, self.model.keep_prob: 0.5}
            self.sess.run(self.train_op, feed_dict=feed_dict)

            if step % 10 == 0:
                feed_dict = {self.model.x: train_x, self.model.y: train_y, self.model.keep_prob: 1}
                summary, train_loss, acc = self.sess.run([merged, self.loss, self.accuracy], feed_dict=feed_dict)
                print("Step: %d / %d (epoch: %d / %d), Train_loss: %g, acc: %g" % (step % self.epoch_length,
                                            self.epoch_length, step // self.epoch_length,
                                            self.number_epoch, train_loss, acc))
                train_writer.add_summary(summary, step)

            if step % 100 == 0:
                val_x, val_y = val_dataset.batch()
                feed_dict = {self.model.x: val_x, self.model.y: val_y, self.model.keep_prob: 1}
                summary, valid_loss, acc = self.sess.run([merged, self.loss, self.accuracy], feed_dict=feed_dict)
                print("%s ---> Validation_loss: %g, acc: %g" % (datetime.datetime.now(), valid_loss, acc))
                test_writer.add_summary(summary, step)

            if step % self.epoch_length == self.epoch_length - 1:
                now_epoch = step // self.epoch_length
                print('Saving checkpoint: ', now_epoch)
                saver.save(self.sess, model_dir + "model.ckpt", now_epoch)

        train_writer.close()
        test_writer.close()
        self.close()

    def close(self):
        self.sess.close()

def eval_model(nodes, samples_feed, eval_sess=None, model_dir=None):

    if eval_sess is None:
        eval_sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            #print("Model restored...", ckpt.model_checkpoint_path)
            saver.restore(eval_sess, ckpt.model_checkpoint_path)

    return eval_sess.run(nodes, samples_feed)

def identify_train():
    batch_size = 32
    dataset_params = {
        'batch_size': batch_size,
        'path': 'resources/train_data/chars',
        'labels_path': 'resources/train_data/chars_list_train.pickle',
        'thread_num': 3
    }
    train_dataset_reader = DataSet(dataset_params)
    dataset_params['labels_path'] = 'resources/train_data/chars_list_val.pickle'
    dataset_params['batch_size'] = -1
    val_dataset_reader = DataSet(dataset_params)

    params = {
        'lr': 0.01,
        'number_epoch': 30,
        'epoch_length': train_dataset_reader.record_number,
        'log_dir': 'train/model/chars/'
    }

    model = Lenet()
    model.compile()
    train = Train(params)
    train.compile(model)
    train.train(train_dataset_reader, val_dataset_reader)

