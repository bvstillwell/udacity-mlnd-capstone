from __future__ import print_function
import numpy as np
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


def create_graph(num_digits,
                 num_labels,
                 valid_dataset,
                 test_dataset,
                 use_dropout=False,
                 learning_rate=0.05,
                 learning_decay=0.596,
                 use_max_pool=False,
                 num_hidden=64,
                 depth1=16,
                 patch_size=5):
    batch_size = 16
    num_channels = 1

    patch_size = patch_size
    depth = depth1

    # Taken from tensorflow tutorial
    num_hidden = num_hidden

    graph = tf.Graph()
    stddev = 0.1

    img_height, img_width = test_dataset[0].shape[:2]

    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,
                                                             img_height,
                                                             img_width,
                                                             num_channels), name="tf_train_dataset")

        tf_train_labels = [tf.placeholder(tf.float32, shape=(batch_size,
                                                             num_labels), name="tf_train_labels_%d" % i)
                           for i in range(num_digits)]

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal([patch_size,
                                                          patch_size,
                                                          num_channels,
                                                          depth], stddev=stddev), name="layer1_weights")
        layer1_biases = tf.Variable(tf.constant(
            0.1, shape=[depth]), name="layer1_biases")

        tmp_size = img_width * img_height * depth
        if use_max_pool:
            tmp_size = int(tmp_size / 4.)
        layer2_weights = tf.Variable(tf.truncated_normal([tmp_size, num_hidden], stddev=stddev),
                                     name="layer2_weights")
        layer2_biases = tf.Variable(tf.constant(
            0.1, shape=[num_hidden]), name="layer2_biases")

        layer4_weights = tf.Variable(tf.truncated_normal([num_hidden,
                                                          num_labels * num_digits], stddev=stddev), name="layer4_weights")
        layer4_biases = tf.Variable(
            tf.ones(shape=[num_labels * num_digits]), name="layer4_biases")

        # Model.
        def model(data, dropout=False):
            if dropout:
                data = tf.nn.dropout(data, 0.9)

            conv = tf.nn.relu(tf.nn.conv2d(data, layer1_weights, [
                              1, 1, 1, 1], padding='SAME') + layer1_biases)
            if use_max_pool:
                conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[
                                      1, 2, 2, 1], padding='SAME')
            if dropout:
                conv = tf.nn.dropout(conv, 0.75)

            shape = conv.get_shape().as_list()
            reshape = tf.reshape(
                conv, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(
                tf.matmul(reshape, layer2_weights) + layer2_biases)
            if dropout:
                hidden = tf.nn.dropout(hidden, 0.5)

            output = tf.matmul(hidden, layer4_weights) + layer4_biases

            split_logits = tf.split(1, num_digits, output)
            return split_logits

        # Training computation.
        logits = model(tf_train_dataset, use_dropout)

        # Training computation.
        logits = model(tf_train_dataset)

        loss = tf.reduce_mean([
            tf.nn.softmax_cross_entropy_with_logits(
                logits[i],
                tf_train_labels[i]
            )for i in range(num_digits)], name='loss')

        # Optimizer.
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   100000,
                                                   learning_decay,
                                                   name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate,
                                                      name='optimizer').minimize(loss,
                                                                                 global_step=global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = [tf.nn.softmax(model(tf_train_dataset)[
                                          i], name="train_prediction_%d" % i) for i in range(num_digits)]

        tf_valid_dataset = tf.constant(valid_dataset, name="tf_valid_dataset")
        valid_prediction = [tf.nn.softmax(model(tf_valid_dataset)[
                                          i], name="valid_prediction_%d" % i) for i in range(num_digits)]

        tf_test_dataset = tf.constant(test_dataset, name="tf_test_dataset")
        test_prediction = [tf.nn.softmax(model(tf_test_dataset)[
                                         i], name="test_prediction_%d" % i) for i in range(num_digits)]
    return graph


def run_graph(graph,
              num_digits,
              train_dataset,
              train_labels,
              valid_labels,
              test_labels,
              mins=1):
    start_time = time.time()

    batch_size = 16

    num_steps = 1000001
    save_steps = 100000
    eval_steps = 500
    valid_steps = 500
    timeout = mins * 60  # 30 minutes * 60 seconds

    tf_train_dataset = graph.get_tensor_by_name('tf_train_dataset:0')
    tf_train_labels = [graph.get_tensor_by_name(
        'tf_train_labels_%d:0' % i) for i in range(num_digits)]
    train_prediction = [graph.get_tensor_by_name(
        'train_prediction_%d:0' % i) for i in range(num_digits)]
    valid_prediction = [graph.get_tensor_by_name(
        'valid_prediction_%d:0' % i) for i in range(num_digits)]
    test_prediction = [graph.get_tensor_by_name(
        'test_prediction_%d:0' % i) for i in range(num_digits)]
    optimizer = graph.get_tensor_by_name('optimizer:0')
    loss = graph.get_tensor_by_name('loss:0')
    learning_rate = graph.get_tensor_by_name('learning_rate:0')

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()

        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {tf_train_labels[i]: batch_labels[
                :, i, :] for i in range(num_digits)}
            feed_dict[tf_train_dataset] = batch_data

            fetches = [optimizer, loss]
            results = session.run(fetches, feed_dict=feed_dict)

            elapsed_time = time.time() - start_time
            timeup = elapsed_time >= timeout

            if step > 0 or timeup:
                if (step % eval_steps == 0 or timeup):
                    print('Elapsed time(s):%d/%d (%.2f%%)' %
                          (elapsed_time, timeout, 1.0 * elapsed_time / timeout))
                    if timeup:
                        print("\nTIMEUP!")
                    print('Learning rate:', learning_rate.eval())
                    print('Minibatch loss at step %d: %f' % (step, results[1]))

                    fetches = train_prediction
                    results = session.run(fetches, feed_dict={
                                          tf_train_dataset: batch_data})
                    train_accuracy = accuracy_list(
                        num_digits,
                        results,
                        batch_labels)

                    print('Minibatch accuracy: %.1f%%' % train_accuracy)

                if (step % valid_steps == 0 or timeup):
                    valid_accuracy = accuracy_list(
                        num_digits,
                        [valid_prediction[i].eval()
                         for i in range(num_digits)],
                        valid_labels)
                    print('Validation accuracy: %.1f%%' % valid_accuracy)

                if step % save_steps == 0 or timeup:
                    test_accuracy = accuracy_list(
                        num_digits,
                        [test_prediction[i].eval() for i in range(num_digits)],
                        test_labels)
                    print('Test accuracy: %.1f%%' % test_accuracy)

            if timeup:
                break

    print("Finished")