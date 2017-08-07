# from scipy.misc import imread
from PIL import Image

import tensorflow as tf
import os
import numpy as np
import CONSTANT

def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape, mean=0, stddev=1.0, dtype=tf.float32))
    return initial


def bias_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32))
    return initial


def conv2d(x, W, strides=(1, 1)):
    stride_1, stride_2 = strides
    conv = tf.nn.conv2d(x, W, strides=[1, stride_1, stride_2, 1], padding='SAME')
    return conv


def max_pooling(x, filter_sizes, strides=(1, 1)):
    stride_1, stride_2 = strides
    size_1, size_2 = filter_sizes
    pooled = tf.nn.max_pool(x, ksize=[1, size_1, size_2, 1], strides=[1, stride_1, stride_2, 1], padding='SAME')
    return pooled


def local_norm(output):
    return
    # tf.nn.local_response_normalization()
    # BETTER USE BATCH_NORM
    # TODO


def load_data():
    def crop_image(image, height, width):  # crop image
        half_the_width = image.size[0] / 2
        half_the_height = image.size[1] / 2
        img_temp = image.crop(
            (
                half_the_width - width/2,
                half_the_height - height/2,
                half_the_width + width/2,
                half_the_height + height/2
            )
        )
        # img_temp.save("img4.jpg")
        return img_temp  # crop image

    shape_0, shape_1, shape_2 = CONSTANT.IMAGE_SHAPE

    db_path = CONSTANT.GENKI4K_db_path
    images_list = os.listdir(db_path)

    images = np.empty([shape_0, shape_1, shape_2])
    print("xxx", images.shape)
    for index, image_name in enumerate(images_list):
        img = Image.open(db_path + image_name)
        # imread(db_path + image_name)
        tmp = crop_image(img, shape_0, shape_1)
        try:
            np.concatenate((images, np.asarray(tmp)), axis=0)
        except Exception:
            pass

    return images


def load_labels():
    db_path = CONSTANT.GENKI4K_labels_path

    f = open(db_path + "labels.txt")


def preprocessing(numpy_data):
    # return [tf.image.per_image_standardization(image) for image in numpy_data]
    flatten = np.reshape(numpy_data, (numpy_data.shape[0], -1))
    # print(flatten.shape, numpy_data.shape)

    mean = np.mean(flatten, axis=0)
    stddev = np.std(flatten, axis=0)

    flatten -= mean
    flatten /= stddev

    return flatten
