from .cnn_train import Train
from .net.judgenet import Judgenet
from .dataset import DataSet

def judge_train():
    batch_size = 32
    dataset_params = {
        'batch_size': batch_size,
        'path': 'resources/train_data/whether_car',
        'labels_path': 'resources/train_data/whether_list_train.pickle',
        'thread_num': 3,
        'gray': False
    }
    train_dataset_reader = DataSet(dataset_params)
    dataset_params['labels_path'] = 'resources/train_data/whether_list_val.pickle'
    dataset_params['batch_size'] = -1
    val_dataset_reader = DataSet(dataset_params)

    params = {
        'lr': 0.01,
        'number_epoch': 2,
        'epoch_length': train_dataset_reader.record_number,
        'log_dir': 'train/model/whether_car/'
    }

    model = Judgenet()
    model.compile()
    train = Train(params)
    train.compile(model)
    train.train(train_dataset_reader, val_dataset_reader)