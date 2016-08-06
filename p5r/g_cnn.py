#
# This file contains the code to run a parameter seach
#
from common import *
from run_graph import *
import numpy as np
import pandas as pd


# *******************************************************
#
# Ok, let's setup all the configuration for the training
# and scoring
#
# *******************************************************

#
# Our feature and labels
#
default_data_config = {
    'label_set': label_set,
    'image_set': dataset_28,
}

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
    ],
    'include_digit_length_classifier': [False],
}

#
# The max training time for each model
#
max_running_mins = 5

#
# Training loop configuration
#
training_config = {
    'eval_step': 500,
    'valid_step': 500,
    'batch_size': 16,
    'mins': max_running_mins,
    'save_model': True,
    #  'dry_run': True
}

#
# Our results filename. This file is also used for resuming
#
results_filename = 'results-%dmins.csv' % max_running_mins


# *******************************************************
#
# Ok, all setup lets get cracking!
#
# *******************************************************
log('*' * 40)
log('Runing parameter search')
log('*' * 40)

#
# Create all the permutations of grid configurations
#
graph_config_combinations = expand_param_search(grid_param_search)
log("Running %d combinations" % len(graph_config_combinations))

#
# Shuffle the graph combinations, to keep things interesting
#
np.random.shuffle(graph_config_combinations)

#
# Check expected running time
#
run_mins = len(graph_config_combinations) * max_running_mins
log("Expected runtime:%dd %dh %dm" % (
    run_mins // (60 * 24),
    (run_mins // 60) % 24,
    run_mins % 60))

#
# Load the results file if exists
#
if os.path.isfile(results_filename):
    df_results = pd.read_csv(results_filename)
else:
    df_results = None

#
# Loop through our graph_config permutations
#
for i, graph_config in enumerate(graph_config_combinations):

    #
    # Check if we have run with this configuration already, and skip if we have
    #
    if df_results is not None:
        # Turn our graph_config into a series and search for it in the pandas frame
        df_tmp = pd.Series(graph_config)
        df_tmp['layers'] = str(df_tmp['layers'])
        found_row = df_results.loc[(df_results[list(graph_config)] == df_tmp).all(axis=1)]
        if not found_row.empty:
            log("Already run %s" % pprint.pformat(graph_config))
            continue

    #
    # Run the configuration through create, train and score
    #
    result = create_train_score(graph_config, training_config, default_data_config)

    #
    # Record results
    #
    result.update(graph_config)
    if df_results is None:
        df_results = pd.DataFrame([result])
    else:
        df_results = df_results.append(result, ignore_index=True)

    #
    # output the results to a file after each training so
    # we can continue should something break
    #
    log("Writing results:%s" % results_filename)
    df_results.to_csv(results_filename, index=False)
    log("Finished:%d/%d" % (i + 1, len(graph_config_combinations)))
    log('')
