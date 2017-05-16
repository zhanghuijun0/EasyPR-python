from six.moves import cPickle as pickle
import os

def add_dirname(path):
    return os.path.join(os.path.dirname(__file__), path)

with open(add_dirname('../etc/province_mapping.pickle'), 'rb') as f:
    province_mapping = pickle.load(f)

with open(add_dirname('../etc/index2str.pickle'), 'rb') as f:
    index2str = pickle.load(f)