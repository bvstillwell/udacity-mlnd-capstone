# ############################################################################
#
# This file is used to load and format all the data for use with tensorflow
#
from six.moves import cPickle as pickle
import numpy as np
from logging_setup import *

#
#
# Load the training, validation and testing data
#
#
with open('SVHN_train.pickle', 'rb') as f:
    tmp_save = pickle.load(f)
    train_dataset_56 = tmp_save['dataset']
    train_labels = tmp_save['labels']

#
# Create a validation set
#
valid_size = 2000
valid_dataset_56 = train_dataset_56[:valid_size]
valid_labels = train_labels[:valid_size]
train_dataset_56 = train_dataset_56[valid_size:]
train_labels = train_labels[valid_size:]

#
# Load the test data
#
with open('SVHN_test.pickle', 'rb') as f:
    tmp_save = pickle.load(f)
    test_dataset_56 = tmp_save['dataset']
    test_labels = tmp_save['labels']


#
#
# Limit the number of digits in the datasets
#
#
def max_digits(dataset, labels, max_digits):
    """Limit the dataset and labels to max number of digits"""
    keep = [i for i, label in enumerate(labels) if len(label) <= max_digits]
    return dataset[keep], labels[keep]
#
# The maximum number of digits to use
#
num_digits = 3

#
# Remove all data that has more digits that the maximum number
#
train_dataset_56, train_labels = max_digits(train_dataset_56, train_labels, num_digits)
valid_dataset_56, valid_labels = max_digits(valid_dataset_56, valid_labels, num_digits)
test_dataset_56, test_labels = max_digits(test_dataset_56, test_labels, num_digits)


#
#
# Format the labels for processing in the graph
#
#
def format_labels(num_digits, num_labels, dataset, labels):
    """format the labels into the format for the tensor"""
    dataset_output = dataset.reshape(
        list(dataset.shape) + [1]).astype(np.float32)
    labels_output = np.array([np.array([(np.arange(num_labels) == l).astype(np.float32)
                                        for l in np.append(row, [num_labels - 1] * (num_digits - len(row)), 0)])
                              for row in labels])
    return dataset_output, labels_output
num_labels = 11  # Add an extra character so we can deal with spaces
num_channels = 1  # grayscale
#
# Update the labels to be in a format for tensorflow
#
train_dataset_56, train_labels = format_labels(num_digits, num_labels, train_dataset_56, train_labels)
valid_dataset_56, valid_labels = format_labels(num_digits, num_labels, valid_dataset_56, valid_labels)
test_dataset_56, test_labels = format_labels(num_digits, num_labels, test_dataset_56, test_labels)

#
#
# Create smaller pictures for faster processing
#
#
train_dataset_28 = train_dataset_56[:, ::2, ::2, :]
valid_dataset_28 = valid_dataset_56[:, ::2, ::2, :]
test_dataset_28 = test_dataset_56[:, ::2, ::2, :]

#
#
# Use these variables to make this data more manageable
#
#
dataset_56 = (train_dataset_56, valid_dataset_56, test_dataset_56)
dataset_28 = (train_dataset_28, valid_dataset_28, test_dataset_28)
label_set = (train_labels, valid_labels, test_labels)
