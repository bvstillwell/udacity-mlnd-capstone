from __future__ import print_function
from six.moves import cPickle as pickle

from common import *


with open('SVHN_train.pickle', 'rb') as f:
    tmp_save = pickle.load(f)
    train_dataset_56 = tmp_save['dataset']
    train_labels = tmp_save['labels']

valid_size = 2000
valid_dataset_56 = train_dataset_56[:valid_size]
valid_labels = train_labels[:valid_size]
train_dataset_56 = train_dataset_56[valid_size:]
train_labels = train_labels[valid_size:]

with open('SVHN_test.pickle', 'rb') as f:
    tmp_save = pickle.load(f)
    test_dataset_56 = tmp_save['dataset'][::2]
    test_labels = tmp_save['labels'][::2]


num_digits = 3

train_dataset_56, train_labels = max_digits(
    train_dataset_56, train_labels, num_digits)
valid_dataset_56, valid_labels = max_digits(
    valid_dataset_56, valid_labels, num_digits)
test_dataset_56, test_labels = max_digits(
    test_dataset_56, test_labels, num_digits)



num_labels = 11  # Add an extra character so we can deal with spaces
num_channels = 1  # grayscale

train_dataset_56, train_labels = reformat(
    num_digits, num_labels, train_dataset_56, train_labels)
valid_dataset_56, valid_labels = reformat(
    num_digits, num_labels, valid_dataset_56, valid_labels)
test_dataset_56, test_labels = reformat(
    num_digits, num_labels, test_dataset_56, test_labels)


test_dataset_56 = test_dataset_56[:6000]
test_labels = test_labels[:6000]

train_dataset_28 = train_dataset_56[:, ::2, ::2, :]
valid_dataset_28 = valid_dataset_56[:, ::2, ::2, :]
test_dataset_28 = test_dataset_56[:, ::2, ::2, :]

log('Training set:%s, %s' % (train_dataset_56.shape, train_labels.shape))
log('Validation set:%s, %s' % (valid_dataset_56.shape, valid_labels.shape))
log('Test set:%s, %s' % (test_dataset_56.shape, test_labels.shape))
log('Training set:%s, %s' % (train_dataset_28.shape, train_labels.shape))
log('Validation set:%s, %s' % (valid_dataset_28.shape, valid_labels.shape))
log('Test set:%s, %s' % (test_dataset_28.shape, test_labels.shape))

dataset_56 = (train_dataset_56, valid_dataset_56, test_dataset_56)
dataset_28 = (train_dataset_28, valid_dataset_28, test_dataset_28)
label_set = (train_labels, valid_labels, test_labels)

# The defaults are picked up from the function defaults.
# Override any by setting them in this dict
default_training_config = {}

default_data_config = {
    'label_set': label_set,
    'image_set': dataset_28,
}


def run(graph_config,
        training_config=default_training_config,
        data_config=default_data_config):

    log(graph_config)
    log(training_config)
    log(data_config['image_set'][0].shape)
    log(data_config['label_set'][0].shape)

    # Generate the graph
    graph = create_graph(
        training_config,
        data_config,
        **graph_config)

    # Train, score and save the graph
    run_graph(
        graph,
        data_config,
        **training_config)

    return graph
