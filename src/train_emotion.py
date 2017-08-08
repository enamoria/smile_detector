import numpy as np
import tensorflow as tf

import CONSTANT
from src import training_utils as utils

# Model param:
tf.app.flags.DEFINE_integer('batch_size', 256, "Batch_size")
tf.app.flags.DEFINE_string('db_path', CONSTANT.GENKI4K_db_path, "path to genki db")
tf.app.flags.DEFINE_string('labels_path', CONSTANT.GENKI4K_labels_path, "path to genki labels")

# About genki db
IMAGE_SIZE = CONSTANT.IMAGE_SHAPE
NUM_CLASS = CONSTANT.NUM_CLASS

# Training param
BATCH_SIZE = 256
LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.1
NUM_EPOCH = 10

TRAINING_SET = 3000
TESTING_SET = 1000


# Have no idea what this thing do todo
def _activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def nn_emotion(images, labels):
    # data = utils.load_data()
    # utils.preprocessing(data)

    # x = tf.placeholder(tf.float32, shape=[None, CONSTANT.FLATTEN_SHAPE])
    # y_ = tf.placeholder(tf.float32, shape=[None, 2])
    #
    # # NN architecture
    # W_conv1 = utils.weight_variable([])

    with tf.variable_scope('conv1') as scope:
        kernel = utils.weight_variable(shape=[11, 11, 1, 32])
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')

        bias = utils.bias_variable(shape=[32])

        conv1 = tf.nn.relu(conv + bias, name=scope.name)

        # No idea
        _activation_summary(x=conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # norm1 todo better use batchnorm
    norm1 = tf.nn.local_response_normalization(pool1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = utils.weight_variable(shape=[5, 5, 32, 96])
        conv = tf.nn.conv2d(norm1, kernel, strides=[1, 1, 1, 1], padding='SAME')

        bias = utils.bias_variable(shape=[32])

        conv2 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)
        _activation_summary(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # norm2 todo better use batchnorm
    norm2 = tf.nn.lrn(pool2, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = utils.weight_variable(shape=[5, 5, 96, 128])
        conv = tf.nn.conv2d(norm2, kernel, strides=[1, 1, 1, 1], padding='SAME')

        bias = utils.bias_variable([128])

        conv3 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)
        _activation_summary(conv3)

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = utils.weight_variable(shape=[5, 5, 128, 96])
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')

        bias = utils.bias_variable([96])

        conv4 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)
        _activation_summary(conv4)

    # FC1
    conv4_tensor_shape = conv4.shape().as_list()
    reshape = tf.reshape(conv4,
                         [conv4_tensor_shape[0], conv4_tensor_shape[1] * conv4_tensor_shape[2] * conv4_tensor_shape[3]])

    fc1_weights = utils.weight_variable(
        [conv4_tensor_shape[1] * conv4_tensor_shape[2] * conv4_tensor_shape[3], CONSTANT.FC_NEURON])
    fc1_bias = utils.bias_variable([CONSTANT.FC_NEURON])

    fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_bias)

    fc2_weights = utils.weight_variable([160, 2])
    fc2_bias = utils.bias_variable([2])

    fc2 = tf.matmul(fc1, fc2_weights) + fc2_bias

    return fc2


def main():
    labels = utils.load_labels()
    data = utils.load_data()

    data = utils.preprocessing(data)

    y_nn = nn_emotion(data, labels)

    x = tf.placeholder(tf.float32, [None, 96, 96, 3])
    y = tf.placeholder(tf.int8, [None, 2])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_nn))

    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        np.random.shuffle(data)

        for step in range(200):
            X_train = data[:3000]
            y_train = labels[:3000]

            X_test = data[3000:]
            y_test = labels[3000:]

            if step % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: X_train, y: y_train})
                print('step %d: training accuracy %g' % (step, train_accuracy))

            print('test_accuracy %g' % accuracy.eval(feed_dict={x: X_test, y: y_test}))

main()

# nn_emotion()
