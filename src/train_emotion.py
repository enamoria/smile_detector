import src.Emotion.training_utils as utils
import src.Emotion.CONSTANT as CONSTANT
import matplotlib.pyplot as plt
import tensorflow as tf


def nn_emotion():
    data = utils.load_data()
    utils.preprocessing(data)

    x = tf.placeholder(tf.float32, shape=[None, CONSTANT.FLATTEN_SHAPE])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])

    # NN architecture
    W_conv1 = utils.weight_variable([])




nn_emotion()
