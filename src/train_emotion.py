from __future__ import print_function

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
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.1
REG_LAMBDA = 0.5

BATCH_SIZE = 256
NUM_EPOCH = 50

TRAIN_SIZE = 3000
TEST_SIZE = 1000

# Log path
LOGS_PATH = "./tmp/logs/"


# Have no idea what this thing do todo
def _activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name, x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def nn_emotion(images):
    # data = utils.load_data()
    # utils.preprocessing(data)

    # x = tf.placeholder(tf.float32, shape=[None, CONSTANT.FLATTEN_SHAPE])
    # y_ = tf.placeholder(tf.float32, shape=[None, 2])
    #
    # # NN architecture
    # W_conv1 = utils.weight_variable([])

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
        tf.summary.histogram("activation", conv1)

        regularization = tf.add(regularization, tf.nn.l2_loss(kernel))

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # norm1 todo better use batchnorm
    norm1 = tf.nn.local_response_normalization(pool1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = utils.weight_variable(shape=[5, 5, 32, 96])
        conv = tf.nn.conv2d(norm1, kernel, strides=[1, 1, 1, 1], padding='SAME')

        bias = utils.bias_variable(shape=[96])

        conv2 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)

        # _activation_summary(conv2)
        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("conv", conv)
        tf.summary.histogram("activation", conv2)

        regularization = tf.add(regularization, tf.nn.l2_loss(kernel))

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

        # _activation_summary(conv3)
        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("conv", conv)
        tf.summary.histogram("activation", conv3)

        regularization = tf.add(regularization, tf.nn.l2_loss(kernel))
        # tf.Print(regularization, [regularization])

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = utils.weight_variable(shape=[5, 5, 128, 96])
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')

        bias = utils.bias_variable([96])

        conv4 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)

        # _activation_summary(conv4)
        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("conv", conv)
        tf.summary.histogram("activation", conv4)

        regularization = tf.add(regularization, tf.nn.l2_loss(kernel))

    # FC1
    conv4_tensor_shape = conv4.get_shape().as_list()
    reshape = tf.reshape(conv4,
                         [-1, conv4_tensor_shape[1] * conv4_tensor_shape[2] * conv4_tensor_shape[3]])

    fc1_weights = utils.weight_variable(
        [conv4_tensor_shape[1] * conv4_tensor_shape[2] * conv4_tensor_shape[3], CONSTANT.FC_NEURON])
    fc1_bias = utils.bias_variable([CONSTANT.FC_NEURON])

    fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_bias)

    # FC2
    fc2_weights = utils.weight_variable([160, 2])
    fc2_bias = utils.bias_variable([2])

    fc2 = tf.matmul(fc1, fc2_weights) + fc2_bias

    regularization = tf.multiply(0.01, (regularization + tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc2_weights)))

    # + tf.nn.l2_loss(fc2_bias) + tf.nn.l2_loss(fc1_bias)  # No more regularization for biases
    # print(regularization.get_shape().as_list())

    return fc2, regularization


def main():
    # labels = utils.load_labels()
    data, labels = utils.load_data()

    # data = utils.preprocessing(data)
    # TODO need to implement 4-fold cross-validation later on
    X_test_set = data[TRAIN_SIZE:]
    y_test_set = labels[TRAIN_SIZE:]

    X_train_set = data[:TRAIN_SIZE]
    y_train_set = labels[:TRAIN_SIZE]

    for sample in y_train_set:
        print(sample)
    #######################

    x = tf.placeholder(tf.float32, [None, 96, 96, 3], name='x')
    y_ = tf.placeholder(tf.int8, [None, 2], name='y_')

    y_nn, regularization = nn_emotion(x)

    # print(regularization.get_shape().as_list())
    # y_nn_softmax = tf.nn.softmax(y_nn)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_nn) + regularization)

    # + REG_LAMBDA / float(2 * TRAIN_SIZE) * (tf.nn.l2_loss(y_nn)) +

    # TODO regularization: DONE???

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Summary
    # print(cross_entropy.get_shape().as_list())
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)

    # Merge all summaries
    summary_op = tf.summary.merge_all()

    # FUCKING TIRED OF DEBUGGING
    f_debug = open("fuckDebug", "w")
    f_debug1 = open("trueLabels", "w")
    # FUCKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print(cross_entropy.eval())

        # tf.global_variables_initializer().run()

        ### for each epoch, do:
        ###   for each batch, do:
        ###     create pre-processed batch
        ###     run optimizer by feeding batch
        ###     find cost and reiterate to minimize

        writer = tf.summary.FileWriter(logdir=LOGS_PATH, graph=tf.get_default_graph())
        print(accuracy.eval(feed_dict={x: X_test_set, y_: y_test_set}))

        for epoch in range(NUM_EPOCH):
            # batch
            total_batch = int(TRAIN_SIZE / BATCH_SIZE)

            batches_x, batches_y = utils.batch_generator(X_train_set, y_train_set, BATCH_SIZE)
            for step in range(total_batch):  # TODO batch actually,not step
                # batch_x, batch_y = utils.batch_generator(X_train_set, y_train_set, BATCH_SIZE)

                batch_x = batches_x[step]
                batch_y = batches_y[step]
                # print(batch_y)

                _, summary = sess.run([train_step, summary_op], feed_dict={x: batch_x, y_: batch_y})

                if step % 100 == 0:
                    training_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
                    loss = cross_entropy.eval(feed_dict={x: batch_x, y_: batch_y})

                    print("Epoch:", epoch, "step:", step, "acc:", training_accuracy, "loss:", loss)

                writer.add_summary(summary)

            # input("FUCK")
            # print("Epoch:", epoch, " accuracy = ", training_accuracy, " loss = ", loss)

        test_accuracy = accuracy.eval(feed_dict={x: X_test_set, y_: y_test_set})

        print(test_accuracy)

    f_debug.close()
    f_debug1.close()


main()

# nn_emotion()
