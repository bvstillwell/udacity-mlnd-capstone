# #############################################################################
#
# This file will run a hyperparameter search based on the configurations listed
#
# The important configurations are:
#
#   grid_param_search   : The parameter values to search
#   default_data_config : The datasets and labels to use for test, validation
#                       : and training
#   max_running_mins    : Max training minutes per graph
#   training_config     : The configuration for training
#   results_filename    : The file to save the results in
#

from load_data import *
from train_and_score import run_hyperparameter_search


# *******************************************************
#
# Ok, let's setup all the configuration for the training
# and scoring
#
# *******************************************************

#
# All the possible parameters for the grid search
#
grid_param_search = {
    'use_dropout': [True, False],
    'use_max_pool': [True, False],
    'learning_rate': [0.05, 0.1, 0.025],
    'num_hidden': [32, 64, 128],
    'layers': [
        [8],
        [16],
        [32],
        [8, 16],
        [16, 32],
        [8, 16, 32]
    ]
}


#
# Our feature and labels dataset
#
default_data_config = {
    'label_set': label_set,
    'image_set': dataset_28,
}

#
# The max training time for each model
#
max_running_mins = 2


#
# Training loop configuration
#
training_config = {
    'eval_step': 500,
    'valid_step': 500,
    'batch_size': 2,
    'mins': max_running_mins,
    'save_model': True,
    'dry_run': True
}


#
# The results filename. This file is also used for resuming
#
results_filename = 'results-%dmins.csv' % max_running_mins


# *******************************************************
#
# Ok, all setup lets get cracking!
#
# *******************************************************
run_hyperparameter_search(
    grid_param_search,
    training_config,
    default_data_config,
    results_filename)
