from __future__ import print_function

import time

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import CONSTANT
from src import training_utils as utils

# Model param:
# todo
tf.app.flags.DEFINE_integer('batch_size', 256, "Batch_size")
tf.app.flags.DEFINE_string('db_path', CONSTANT.GENKI4K_db_path, "path to genki db")
tf.app.flags.DEFINE_string('labels_path', CONSTANT.GENKI4K_labels_path, "path to genki labels")

# About genki db
IMAGE_SIZE = CONSTANT.IMAGE_SHAPE
NUM_CLASS = CONSTANT.NUM_CLASS

# Training param
LEARNING_RATE = 0.05
LEARNING_RATE_DECAY = 0.1
REG_LAMBDA = 0.5

BATCH_SIZE = 256
NUM_EPOCH = 50

TRAIN_SIZE = 3000
TEST_SIZE = 1000

dropout = 0.8

# Log path
LOGS_PATH = "./tmp/logs/"


# Have no idea what this thing do todo
def _activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name, x)
    # tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def nn_emotion(images):
    regularization = tf.Variable(0, dtype=tf.float32)
    with tf.variable_scope('conv1') as scope:
        kernel = utils.weight_variable(shape=[11, 11, 3, 32])
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')

        bias = utils.bias_variable(shape=[32])

        conv1 = tf.nn.relu(conv + bias, name=scope.name)

        # No idea EDITED: Ok i got it =))
        # _activation_summary(x=conv1)
        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("conv", conv)

        regularization = tf.add(regularization, tf.nn.l2_loss(kernel))

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # norm1 todo better use batchnorm
    # norm1 = tf.nn.local_response_normalization(pool1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = utils.weight_variable(shape=[5, 5, 32, 96])
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')

        bias = utils.bias_variable(shape=[96])

        conv2 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)

        _activation_summary(conv2)
        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("conv", conv)

        regularization = tf.add(regularization, tf.nn.l2_loss(kernel))

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # norm2 todo better use batchnorm
    # norm2 = tf.nn.lrn(pool2, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = utils.weight_variable(shape=[5, 5, 96, 128])
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')

        bias = utils.bias_variable([128])

        conv3 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)

        # _activation_summary(conv3)
        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("conv", conv)

        regularization = tf.add(regularization, tf.nn.l2_loss(kernel))

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = utils.weight_variable(shape=[3, 3, 128, 96])
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')

        bias = utils.bias_variable([96])

        conv4 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)

        # _activation_summary(conv4)
        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("conv", conv)

        regularization = tf.add(regularization, tf.nn.l2_loss(kernel))

    # conv5
    with tf.variable_scope('conv5') as scope:
        kernel = utils.weight_variable(shape=[3, 3, 96, 96])
        conv = tf.nn.conv2d(conv4, kernel, strides=[1, 1, 1, 1], padding='SAME')

        bias = utils.bias_variable([96])

        conv5 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)

        # _activation_summary(conv5)
        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("conv", conv)

        regularization = tf.add(regularization, tf.nn.l2_loss(kernel))

    # FC1
    conv5_tensor_shape = conv5.get_shape().as_list()
    reshape = tf.reshape(conv5,
                         [-1, conv5_tensor_shape[1] * conv5_tensor_shape[2] * conv5_tensor_shape[3]])

    fc1_weights = utils.weight_variable(
        [conv5_tensor_shape[1] * conv5_tensor_shape[2] * conv5_tensor_shape[3], CONSTANT.FC_NEURON])
    fc1_bias = utils.bias_variable([CONSTANT.FC_NEURON])

    fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_bias)

    # FC2
    fc2_weights = utils.weight_variable([160, 2])
    fc2_bias = utils.bias_variable([2])

    fc2 = tf.matmul(fc1, fc2_weights) + fc2_bias

    fc2 = tf.nn.dropout(fc2, dropout)

    regularization = tf.multiply(0.001, (regularization + tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc2_weights)))

    return fc2, regularization


def main():
    # labels = utils.load_labels()
    data, labels = utils.load_data()

    # plt.imshow(data[1])
    # plt.show()
    # data = utils.preprocessing(data)
    # plt.imshow(data[1])
    # plt.show()

    # TODO need to implement 4-fold cross-validation later on
    X_test_set = data[TRAIN_SIZE:]
    y_test_set = labels[TRAIN_SIZE:]

    X_train_set = data[:TRAIN_SIZE]
    y_train_set = labels[:TRAIN_SIZE]
    #######################

    x = tf.placeholder(tf.float32, [None, 90, 90, 3], name='x')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y_')
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    y_nn, regularization = nn_emotion(x)

    pred = tf.nn.softmax(y_nn)

    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_nn)) #+ regularization

    # TODO regularization: DONE???

    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                               150, 0.96, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ''' Summary '''
    # print(cross_entropy.get_shape().as_list())
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)

    ''' Merge all summaries '''
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())

        ### for each epoch, do:
        ###   for each batch, do:
        ###     create pre-processed batch
        ###     run optimizer by feeding batch
        ###     find cost and reiterate to minimize

        writer = tf.summary.FileWriter(logdir=LOGS_PATH, graph=tf.get_default_graph())

        for epoch in range(NUM_EPOCH):
            total_batch = int(TRAIN_SIZE / BATCH_SIZE)

            batches_x, batches_y = utils.batch_generator(X_train_set, y_train_set, BATCH_SIZE)
            for step in range(total_batch):  # TODO batch actually,not step
                counter = 0
                while counter < 6:
                    batch_x = batches_x[step]
                    batch_y = batches_y[step]

                    batch_x = utils.augmentation(batch_x, counter=counter)
                    # if counter == 1:
                    #     if np.random.random() >= 0.5:
                    #         batch_x = utils.augmentation(batch_x, 'fliplr')
                    #     else:
                    #         break
                    #
                    # if counter == 2:
                    #     if np.random.random() >= 0.5:
                    #         batch_x = utils.augmentation(batch_x, 'flipud')
                    #     else:
                    #         break

                    _, summary = sess.run([train_step, summary_op],
                                          feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout})

                    writer.add_summary(summary, epoch * total_batch + step)
                    counter += 1

            training_accuracy = accuracy.eval(feed_dict={x: X_train_set, y_: y_train_set, keep_prob: dropout})
            loss = cross_entropy.eval(feed_dict={x: X_train_set, y_: y_train_set, keep_prob: dropout})
            test_accuracy = accuracy.eval(feed_dict={x: X_test_set, y_: y_test_set, keep_prob: 1})
            lr = learning_rate.eval()

            print("Epoch:", epoch, "acc:", training_accuracy, test_accuracy, "loss:", loss, "learning rate:", lr)

            if test_accuracy > 0.95:
                for index, predict in enumerate(
                        np.argmax(pred.eval(feed_dict={x: X_test_set, y_: y_test_set, keep_prob: 1}), axis=1)):
                    if predict != np.argmax(y_test_set[index]):
                        if predict == 1:
                            plt.title("Smile")
                        else:
                            plt.title("Non-smile")
                        plt.imshow(X_test_set[index])

                    if index > 10:
                        break
                plt.show()

        test_accuracy = accuracy.eval(feed_dict={x: X_test_set, y_: y_test_set, keep_prob: 1})

        print(test_accuracy)

        print("Training time: ", time.time() - start_time)


main()
