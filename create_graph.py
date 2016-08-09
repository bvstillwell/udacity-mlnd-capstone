# #############################################################################
# This file contains the code for creating a graph based on a number of
# parameters
#
# Important function in the file
#
#   create_graph: Create a graph based on a number of parameters
#

import pprint
import tensorflow as tf
from logging_setup import *


def create_graph(
        training_config,
        data_config,
        use_dropout=False,
        learning_rate=0.05,
        learning_decay=0.596,
        use_max_pool=False,
        num_hidden=32,
        layers=[16],
        patch_size=5):
    """Create a graph based on the input parameters"""

    #
    # Extract variables for creating the graph
    #
    batch_size = training_config['batch_size']
    img_height = data_config['image_set'][0][0].shape[0]
    img_width = data_config['image_set'][0][0].shape[1]
    num_digits = data_config['label_set'][0].shape[1]
    num_labels = data_config['label_set'][0].shape[2]

    #
    # Log the parameters
    #
    params = locals().copy()
    del params['data_config']
    del params['training_config']
    log("Create graph params:")
    log(pprint.pformat(params))
    log('')

    #
    # The graph to populate and return
    #
    graph = tf.Graph()

    #
    # The std deviation to use when initially creating the weights
    #
    stddev = 0.1
    with graph.as_default():

        #
        # Placeholder for the input data
        #
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, 1), name="tf_train_dataset")

        #
        # Placeholder for our labels, to use with our loss function
        #
        tf_train_labels = [tf.placeholder(tf.float32, shape=(batch_size, num_labels), name="tf_train_labels_%d" % i) for i in range(num_digits)]

        #
        # Create the convolutional weights and biases
        #
        weights = []
        biases = []
        all_layers = layers[:]
        all_layers.insert(0, 1)
        layers = None
        # weight will look like [1, 8, 16, 32]
        # or [1, 8]
        pic_dim = img_width
        for i in range(len(all_layers) - 1):
            weights.append(tf.Variable(tf.truncated_normal([patch_size, patch_size, all_layers[i], all_layers[i + 1]], stddev=stddev), name="tf_layer%d_weights" % i))
            biases.append(tf.Variable(tf.constant(0.1, shape=[all_layers[i + 1]]), name="tf_layer%d_biases" % i))
            if use_max_pool:
                # Our pic size decreased when we use max_pooling
                pic_dim = int(round(pic_dim / 2))
                # Minimum pic size
                if pic_dim < 4:
                    pic_dim = 4

        #
        # Fully connected weghts and biases
        #
        connected_size = pic_dim * pic_dim * all_layers[-1]
        tf_connected_weights = tf.Variable(tf.truncated_normal([connected_size, num_hidden], stddev=stddev), name="tf_connected_weights")
        tf_connected_biases = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="tf_connected_biases")

        #
        # Output layer, multiple classifiers
        #
        tf_output_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels * num_digits], stddev=stddev), name="tf_output_weights")
        tf_output_biases = tf.Variable(tf.ones(shape=[num_labels * num_digits]), name="tf_output_biases")

        #
        # Our model structure
        #
        def model(tf_data, dropout=False):
            #
            # Input dropout
            #
            if dropout:
                tf_data = tf.nn.dropout(tf_data, 0.9)

            #
            # Our convolutional layers
            #
            for i in range(len(all_layers) - 1):

                tf_data = tf.nn.relu(tf.nn.conv2d(tf_data, weights[i], [1, 1, 1, 1], padding='SAME') + biases[i])

                if use_max_pool:
                    tf_data = tf.nn.max_pool(tf_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                if dropout:
                    tf_data = tf.nn.dropout(tf_data, 0.75)

            #
            # Reshape our layers to join to fully connected layer
            #
            shape = tf_data.get_shape().as_list()
            tf_reshape = tf.reshape(tf_data, [shape[0], shape[1] * shape[2] * shape[3]])
            tf_hidden = tf.nn.relu(tf.matmul(tf_reshape, tf_connected_weights) + tf_connected_biases)

            if dropout:
                tf_hidden = tf.nn.dropout(tf_hidden, 0.5)

            #
            # Create our output classifiers
            #
            tf_output = tf.matmul(tf_hidden, tf_output_weights) + tf_output_biases
            split_logits = tf.split(1, num_digits, tf_output)

            return split_logits

        #
        # Training computation.
        #
        logits = model(tf_train_dataset, use_dropout)

        #
        # Create the loss function
        #
        loss_digits = [tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits[i],
                tf_train_labels[i]
            ))for i in range(num_digits)]

        loss = tf.add_n(loss_digits, name='tf_loss')

        #
        # Create the optimizer and learning rate
        #
        global_step = tf.Variable(0)  # count the number of steps taken.
        tf_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100000, learning_decay, name='tf_learning_rate')
        tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='tf_optimizer').minimize(loss, global_step=global_step)

        #
        # The results of a batch of predictions. One classifier per digit
        #
        train_prediction = [tf.nn.softmax(model(tf_train_dataset)[i], name="tf_train_prediction_%d" % i) for i in range(num_digits)]

        #
        # Create a predictor for a single image, for use in the android app
        #
        tf_predict_single_dataset = tf.placeholder(tf.float32, shape=(1, img_height, img_width, 3), name='tf_predict_single_dataset')
        # Average the color
        tf_conv = tf.reduce_mean(tf_predict_single_dataset, 3)
        # Shape it correctly
        tf_conv = tf.reshape(tf_conv, tf_conv.get_shape().as_list() + [1])
        # Move the color range
        tf_conv = tf.add(tf.div(tf_conv, 255), -0.5)
        tf_predict_single_output = tf.squeeze(
            tf.concat(1,
                      [tf.nn.softmax(model(tf_conv)[i],
                                     name="tf_test_prediction_%d" % i)
                       for i in range(num_digits)]
                      ),
            name='predict_single_output')

    return graph
