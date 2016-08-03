from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time


def max_digits(dataset, labels, max_digits):
    keep = [i for i, label in enumerate(labels) if len(label) <= max_digits]
    return dataset[keep], labels[keep]


def show_image(img, label):
    print("Labels", label)
    print("Dtype", img.dtype)
    print("Shape", img.shape)
    print("Color range", np.min(img), np.max(img))
    if len(img.shape) > 2:
        plt.imshow(np.reshape(img, img.shape[:2]))
    else:
        plt.imshow(img)
    plt.show()


def show_images(imgs, labels, num=3):
    for i in range(num):
        num = np.random.randint(imgs.shape[0])
        show_image(imgs[num], labels[num])


def reformat(num_digits, num_labels, dataset, labels):
    dataset_output = dataset.reshape(
        list(dataset.shape) + [1]).astype(np.float32)
    labels_output = np.array([np.array([(np.arange(num_labels) == l).astype(np.float32)
                                        for l in np.append(row, [num_labels - 1] * (num_digits - len(row)), 0)])
                              for row in labels])
    return dataset_output, labels_output


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def accuracy_list(num_digits, predictions, labels):
    result = np.mean([accuracy(predictions[i],
                               labels[:, i, :])
                      for i in range(num_digits)])
    return result


# This graph will start of simple, and get more complex as we try
# different inputs


def create_graph(training_config,
                 data_config,
                 use_dropout=False,
                 learning_rate=0.05,
                 learning_decay=0.596,
                 use_max_pool=False,
                 num_hidden=64,
                 layers=[16],
                 patch_size=5):
    num_channels = 1

    graph = tf.Graph()
    stddev = 0.1

    batch_size = training_config['batch_size']
    img_height = data_config['image_set'][0][0].shape[0]
    img_width = data_config['image_set'][0][0].shape[1]
    num_digits = data_config['label_set'][0].shape[1]
    num_labels = data_config['label_set'][0].shape[2]

    # print(img_height, img_width)
    # print(num_digits, num_labels)

    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,
                                                             img_height,
                                                             img_width,
                                                             num_channels),
                                          name="tf_train_dataset")

        tf_train_labels = [tf.placeholder(tf.float32, shape=(batch_size,
                                                             num_labels),
                                          name="tf_train_labels_%d" % i)
                           for i in range(num_digits)]

        # Variables.
        weights = []
        biases = []
        layers.insert(0, 1)
        # weight will look like [1, 8, 16, 32]
        # or [1, 8]

        # We need to calculate our connected size
        pic_dim = img_width
        for i in range(len(layers) - 1):
            weights.append(tf.Variable(tf.truncated_normal([patch_size,
                                                            patch_size,
                                                            layers[i],
                                                            layers[i + 1]],
                                                           stddev=stddev),
                                       name="layer%d_weights" % i))

            biases.append(tf.Variable(tf.constant(0.1, shape=[layers[i + 1]]), name="layer%d_biases" % i))
            if use_max_pool:
                # Our pic size decreased when we use max_pooling
                pic_dim = int(round(pic_dim / 2))
                # Minimum pic size
                if pic_dim < 4:
                    pic_dim = 4

        # The number of weights for the fully connected layer
        connected_size = pic_dim * pic_dim * layers[-1]
        connected_weights = tf.Variable(tf.truncated_normal([connected_size, num_hidden], stddev=stddev), name="connected_weights")
        connected_biases = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="connected_biases")

        # Output layer, multiple classifiers
        output_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels * num_digits], stddev=stddev), name="output_weights")
        output_biases = tf.Variable(tf.ones(shape=[num_labels * num_digits]), name="output_biases")

        # Model.
        def model(data, dropout=False):
            if dropout:
                data = tf.nn.dropout(data, 0.9)

            for i in range(len(layers) - 1):
                data = tf.nn.relu(tf.nn.conv2d(data,
                                               weights[i],
                                               [1, 1, 1, 1],
                                               padding='SAME') + biases[i])
                if use_max_pool:
                    data = tf.nn.max_pool(data,
                                          ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME')
                if dropout:
                    data = tf.nn.dropout(data, 0.75)

            shape = data.get_shape().as_list()
            reshape = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, connected_weights) + connected_biases)
            if dropout:
                hidden = tf.nn.dropout(hidden, 0.5)

            output = tf.matmul(hidden, output_weights) + output_biases

            split_logits = tf.split(1, num_digits, output)
            return split_logits

        # Training computation.
        logits = model(tf_train_dataset, use_dropout)

        # Training computation.
        logits = model(tf_train_dataset)

        # loss = tf.reduce_mean([
        #     tf.nn.softmax_cross_entropy_with_logits(
        #         logits[i],
        #         tf_train_labels[i]
        #     )for i in range(num_digits)], name='loss')

        loss = tf.add_n([tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits[i],
                tf_train_labels[i]
            ))for i in range(num_digits)], name='loss')

        # Optimizer.
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   100000,
                                                   learning_decay,
                                                   name='learning_rate')
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate,
        #                                               name='optimizer').minimize(loss,
        #                                                                          global_step=global_step)
        optimizer = tf.train.AdagradOptimizer(learning_rate, name='optimizer').minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = [tf.nn.softmax(model(tf_train_dataset)[
                                          i], name="train_prediction_%d" % i) for i in range(num_digits)]

        tf_predict_single_dataset = tf.placeholder(
            tf.float32,
            shape=(1, img_height, img_width, 3),
            name='tf_predict_single_dataset')

        # Average the color
        conv = tf.reduce_mean(tf_predict_single_dataset, 3)
        # Shape it correctly
        conv = tf.reshape(conv, conv.get_shape().as_list() + [1])
        # Move the color range
        conv = tf.add(tf.div(conv, 255), -0.5)

        predict_single_output = tf.squeeze(
            tf.concat(1,
                      [tf.nn.softmax(model(conv)[i],
                                     name="test_prediction_%d" % i)
                       for i in range(num_digits)]
                      ),
            name='predict_single_output')

    return graph


def run_model(graph, session, num_digits, batch_data, batch_labels, fetches):
    """Execute ops listed in feteches with a batch of training data"""
    tf_train_dataset = graph.get_tensor_by_name('tf_train_dataset:0')
    tf_train_labels = [graph.get_tensor_by_name('tf_train_labels_%d:0' % i) for i in range(num_digits)]

    feed_dict = {tf_train_labels[i]: batch_labels[:, i, :] for i in range(num_digits)}
    feed_dict[tf_train_dataset] = batch_data

    results = session.run(fetches, feed_dict=feed_dict)
    return results


def run_and_score(graph, session, batch_size, num_digits, dataset, labels):
    """Due to memory constraints, we need to test our items in batches"""
    train_prediction = [graph.get_tensor_by_name('train_prediction_%d:0' % i) for i in range(num_digits)]

    # Calculate test accuracy using batches
    offset = 0
    accuracy_results = []
    while offset <= labels.shape[0] - batch_size:
        batch_data = dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = labels[offset:(offset + batch_size), :]

        results = run_model(
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


def run_graph(
        graph,
        data_config,
        mins=1,
        save_model=False,
        batch_size=16,
        eval_step = 100,
        valid_step = 100):

    start_time = time.time()

    train_dataset, valid_dataset, test_dataset = data_config['image_set']
    train_labels, valid_labels, test_labels = data_config['label_set']

    img_height, img_width = test_dataset[0].shape[:2]
    num_digits, num_labels = test_labels.shape[1:]

    max_steps = 1000001
    timeout = mins * 60  # 30 minutes * 60 seconds
    print("Batch_size:%d" % batch_size)
    print("Mins:%d" % mins)

    # tf_train_dataset = graph.get_tensor_by_name('tf_train_dataset:0')
    # tf_train_labels = [graph.get_tensor_by_name('tf_train_labels_%d:0' % i) for i in range(num_digits)]

    optimizer = graph.get_tensor_by_name('optimizer:0')
    loss = graph.get_tensor_by_name('loss:0')
    learning_rate = graph.get_tensor_by_name('learning_rate:0')

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()

        print('Initialized')
        for step in range(max_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            results = run_model(
                graph,
                session,
                num_digits,
                batch_data,
                batch_labels,
                [optimizer, loss])

            # feed_dict = {tf_train_labels[i]: batch_labels[
            #     :, i, :] for i in range(num_digits)}
            # feed_dict[tf_train_dataset] = batch_data

            # fetches = [optimizer, loss]
            # results = session.run(fetches, feed_dict=feed_dict)

            elapsed_time = time.time() - start_time
            timeup = elapsed_time >= timeout

            if step > 0 or timeup:
                if (step % eval_step == 0 or timeup):
                    print('Elapsed time(s):%d/%d (%.2f%%)' %
                          (elapsed_time, timeout, 1.0 * elapsed_time / timeout))
                    if timeup:
                        print("\nTIMEUP!")
                    print('Learning rate:', learning_rate.eval())
                    print('Minibatch loss at step %d: %f' % (step, results[1]))

                    # Score training dataset
                    train_accuracy = run_and_score(
                        graph,
                        session,
                        batch_size,
                        num_digits,
                        batch_data,
                        batch_labels)

                    print('Minibatch accuracy: %.1f%%' % train_accuracy)

                if (step % valid_step == 0 or timeup):
                    # Score valid dataset
                    valid_accuracy = run_and_score(
                        graph,
                        session,
                        batch_size,
                        num_digits,
                        valid_dataset,
                        valid_labels)

                    print('Validation accuracy: %.1f%%' % valid_accuracy)

                if timeup:
                    break

        # Score against test dataset
        test_accuracy = run_and_score(
            graph,
            session,
            batch_size,
            num_digits,
            test_dataset,
            test_labels)
        print('Test accuracy: %.1f%%' % test_accuracy)

        if save_model:
            print("Saving graph")
            saver = tf.train.Saver()
            checkpoint_path = os.path.join('save', 'model.ckpt')
            saver.save(session, checkpoint_path, global_step=0)
            tf.train.write_graph(session.graph.as_graph_def(), 'save', 'model.pb')

    print("Finished")
