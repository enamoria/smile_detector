import training_utils as utils
import CONSTANT
import matplotlib.pyplot as plt
import tensorflow as tf

# Model param:
tf.app.flags.DEFINE_integer('batch_size', 127, "Batch_size")
tf.app.flags.DEFINE_string('db_path', CONSTANT.GENKI4K_db_path, "path to genki db")
tf.app.flags.DEFINE_string('labels_path', CONSTANT.GENKI4K_labels_path, "path to genki labels")

# About genki db
IMAGE_SIZE = CONSTANT.IMAGE_SHAPE
NUM_CLASS = CONSTANT.NUM_CLASS

# Training param
LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.1


# Have no idea what this thing do todo
def _activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def nn_emotion(images):
    data = utils.load_data()
    utils.preprocessing(data)

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
    with
nn_emotion()
