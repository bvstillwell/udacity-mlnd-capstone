# ############################################################################
#
# This file contains the functions to run the hyperparameter search.
#
# Important functions in this file are
#
#   run_hyperparameter_search   : Run the search
#   create_train_and_score_graph: Create the graph and pass it to the training
#                               : and scoring function
#   train_and_score_graph       : Train, score and save a graph
#

import numpy as np
import pandas as pd
import tensorflow as tf
import time
import pprint

from common import *
from create_graph import create_graph


def run_hyperparameter_search(
        grid_param_search,
        training_config,
        default_data_config,
        results_filename):
    """Run the hyperparameter search based on the grid_param_search"""

    def expand_param_search(param_search, graph_config={}):
        """Create a list of all permutations of the param_search"""
        # If we have no more param_search then lets return this value
        if not param_search:
            return graph_config.copy()

        key = param_search.keys()[0]
        results = []
        for value in param_search[key]:
            graph_config[key] = value

            param_search_next = param_search.copy()
            del param_search_next[key]

            result = expand_param_search(param_search_next, graph_config)
            if type(result) is list:
                results = results + result
            else:
                results.append(result)
        return results

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
    run_mins = len(graph_config_combinations) * training_config['mins']
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
            if 'layers' in df_tmp:
                df_tmp['layers'] = str(df_tmp['layers'])
            found_row = df_results.loc[(df_results[list(graph_config)] == df_tmp).all(axis=1)]
            if not found_row.empty:
                log("Already run %s" % pprint.pformat(graph_config))
                continue

        #
        # Run the configuration through create, train and score
        #
        result = create_train_and_score_graph(
            graph_config,
            training_config,
            default_data_config)

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


def create_train_and_score_graph(
        graph_config,
        training_config,
        data_config):
    """As it says in the title"""

    log("*" * 40)
    log("Train and score starting")
    log("*" * 40)

    #
    # Generate the graph
    #
    graph = create_graph(
        training_config,
        data_config,
        **graph_config)

    #
    # Train, score and save the graph
    #
    score = train_and_score_graph(
        graph,
        data_config,
        **training_config)

    return score


def train_and_score_graph(
        graph,
        data_config,
        mins=1,
        save_model=False,
        batch_size=16,
        eval_step=100,
        valid_step=100,
        dry_run=False):
    """Run a full training and scoring cycle on the graph"""

    def accuracy(predictions, labels):
        """Return the accuracy"""
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    def accuracy_list(num_digits, predictions, labels):
        """Return the average accuracy over many digits"""
        result = np.mean([accuracy(predictions[i],
                                   labels[:, i, :])
                          for i in range(num_digits)])
        return result

    def run_fetches(
            graph,
            session,
            num_digits,
            batch_data,
            batch_labels,
            fetches):
        """Execute ops listed in feteches with a batch of training data"""
        tf_train_dataset = graph.get_tensor_by_name('tf_train_dataset:0')
        tf_train_labels = [graph.get_tensor_by_name('tf_train_labels_%d:0' % i) for i in range(num_digits)]

        feed_dict = {tf_train_labels[i]: batch_labels[:, i, :] for i in range(num_digits)}
        feed_dict[tf_train_dataset] = batch_data

        #
        # Execute the graph
        #
        results = session.run(fetches, feed_dict=feed_dict)
        return results

    def run_and_score_dataset(
            graph,
            session,
            batch_size,
            num_digits,
            dataset,
            labels):
        """Run and score a dataset and labels against our model"""
        train_prediction = [graph.get_tensor_by_name('tf_train_prediction_%d:0' % i) for i in range(num_digits)]

        # Calculate test accuracy using batches
        offset = 0
        accuracy_results = []
        while offset <= labels.shape[0] - batch_size:
            batch_data = dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = labels[offset:(offset + batch_size), :]

            results = run_fetches(
                graph,
                session,
                num_digits,
                batch_data,
                batch_labels,
                train_prediction)

            accuracy = accuracy_list(
                num_digits,
                results,
                batch_labels)

            accuracy_results.append(accuracy)
            offset += batch_size

        # return the average of the tests
        return np.mean(accuracy_results)



    #
    # Extract the required variables from the configurations
    #
    training_shape = str(data_config['image_set'][0].shape)
    valid_shape = str(data_config['image_set'][1].shape)
    test_shape = str(data_config['image_set'][2].shape)
    training_label_shape = str(data_config['label_set'][0].shape)
    valid_label_shape = str(data_config['label_set'][1].shape)
    test_label_shape = str(data_config['label_set'][2].shape)

    #
    # Log the parameters
    #
    params = locals().copy()
    del params['data_config']
    del params['graph']
    log("Run graph params:")
    log(pprint.pformat(params))
    log('')

    #
    # More variables for use
    #
    train_dataset, valid_dataset, test_dataset = data_config['image_set']
    train_labels, valid_labels, test_labels = data_config['label_set']
    img_height, img_width = test_dataset[0].shape[:2]
    num_digits, num_labels = test_labels.shape[1:]

    #
    # The timeout for this training run
    #
    timeout = mins * 60  # 30 minutes * 60 seconds

    #
    # Get the required ops from our graph
    #
    tf_optimizer = graph.get_tensor_by_name('tf_optimizer:0')
    tf_loss = graph.get_tensor_by_name('tf_loss:0')
    tf_learning_rate = graph.get_tensor_by_name('tf_learning_rate:0')

    #
    # Initialise varaibles
    #
    test_accuracy = 0.
    valid_accuracy = 0.
    train_accuracy = 0.
    step = 0
    learning_rate = 0.
    save_file_id = ''

    #
    # Dry-run used for testing purposes
    #
    if dry_run:
        log('Dry run only')
    else:

        #
        # Create a session and get cracking
        #
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            log('Initialized')

            #
            # Let's start timing
            #
            start_time = time.time()

            #
            # The main training loop, the timer will break the loop
            #
            while True:
                step += 1

                #
                # Get our training batch
                #
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]

                #
                # Execute the ops
                #
                results = run_fetches(
                    graph,
                    session,
                    num_digits,
                    batch_data,
                    batch_labels,
                    [tf_optimizer, tf_loss])

                #
                # Calculate the time to see if we should be finished
                #
                elapsed_time = time.time() - start_time
                timeup = elapsed_time >= timeout

                #
                # Output our scores if required
                #
                if step > 0 or timeup:
                    if (step % eval_step == 0 or timeup):
                        log('Elapsed time(s):%d/%d (%.2f%%)' %
                            (elapsed_time, timeout, 1.0 * elapsed_time / timeout))
                        if timeup:
                            log("\nTIMEUP!")
                        learning_rate = tf_learning_rate.eval()
                        log('Learning rate:%.5f' % learning_rate)
                        log('Minibatch loss at step %d: %f' % (step, results[1]))

                        # Score training dataset
                        train_accuracy = run_and_score_dataset(
                            graph,
                            session,
                            batch_size,
                            num_digits,
                            batch_data,
                            batch_labels)

                        log('Minibatch accuracy: %.1f%%' % train_accuracy)

                    if (step % valid_step == 0 or timeup):
                        # Score valid dataset
                        valid_accuracy = run_and_score_dataset(
                            graph,
                            session,
                            batch_size,
                            num_digits,
                            valid_dataset,
                            valid_labels)

                        log('Validation accuracy: %.1f%%' % valid_accuracy)

                    if timeup:
                        break

            #
            # We will be outside the loop here
            # Score against test dataset
            #
            test_accuracy = run_and_score_dataset(
                graph,
                session,
                batch_size,
                num_digits,
                test_dataset,
                test_labels)
            log('Test accuracy: %.1f%%' % test_accuracy)

            #
            # Save the model if required
            #
            if save_model:
                if not os.path.exists('save'):
                    os.makedirs('save')
                save_file_id = get_datetime_filename()
                log("Saving graph:%s" % save_file_id)
                saver = tf.train.Saver()
                checkpoint_path = os.path.join('save', save_file_id + '.ckpt')
                saver.save(session, checkpoint_path, global_step=0)
                tf.train.write_graph(session.graph.as_graph_def(), 'save', save_file_id + '.pb')

    log("Finished\n")

    #
    # Return a nice result set
    #
    result = {
        'test_accuracy': round(test_accuracy / 100., 3),
        'valid_accuracy': round(valid_accuracy / 100., 3),
        'train_accuracy': round(train_accuracy / 100., 3),
        'step': step,
        'final_learning_rate': round(learning_rate, 5),
        'save_name': save_file_id
    }
    log("Result:")
    log(pprint.pformat(result))
    log('')
    return result
