import os
import sys

import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import CONSTANT


def weight_variable(shape, name='name'):
    initial = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01, dtype=tf.float32), name=name)
    return initial


def bias_variable(shape, name='name'):
    # initial = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32), name=name)
    initial = tf.Variable(np.ones(shape), dtype=tf.float32, name=name)
    return initial


def conv2d(x, W, strides=(1, 1), name='name'):
    stride_1, stride_2 = strides
    conv = tf.nn.conv2d(x, W, strides=[1, stride_1, stride_2, 1], padding='SAME')
    return conv


def max_pooling(x, filter_sizes, strides=(1, 1), name='name'):
    stride_1, stride_2 = strides
    size_1, size_2 = filter_sizes
    pooled = tf.nn.max_pool(x, ksize=[1, size_1, size_2, 1], strides=[1, stride_1, stride_2, 1], padding='SAME')
    return pooled


def local_norm(output):
    return
    # tf.nn.local_response_normalization()
    # BETTER USE BATCH_NORM
    # TODO Done


def load_data():
    # shape_0, shape_1, shape_2 = CONSTANT.IMAGE_SHAPE
    # db_path = CONSTANT.GENKI4K_db_path
    db_path = CONSTANT.ALIGNED_CROPPED_db_path
    images_list = os.listdir(db_path)

    labels = load_labels()

    images = []
    for index, image_name in enumerate(images_list):
        img = cv2.imread(db_path + image_name)
        try:
            tmp = cv2.resize(img, (90, 90))
            # print(img.shape)

            images.append(np.asarray(tmp))
            # images.append(np.fliplr(tmp))

            # plt.imshow(np.fliplr(tmp))
            # plt.show()
            # labels = np.concatenate((labels[:index + 1], [labels[index]], labels[index + 1:]))
        except Exception as e:
            print("Exception found in", sys._getframe().f_code.co_name, e)

    images = np.asarray(images, dtype=np.float32)
    index = np.random.permutation(len(images))
    print(index)
    images = images[index]
    labels = labels[index]

    return images, labels
    # return np.asarray(images, dtype=np.float32), labels


def load_labels():
    labels_path = CONSTANT.GENKI4K_labels_path
    labels = []

    with open(labels_path, "r") as f:
        while True:
            seed = np.array([0, 0])
            tmp = f.readline()

            if tmp == "":
                break

            seed[int(tmp.split(' ')[0])] = 1
            labels.append(seed)

    return np.asarray(labels)


def augmentation(data, augmentation_method='fliplr'):
    # print(type(data))
    if augmentation_method == 'fliplr':
        return [np.fliplr(i) for i in data]
    if augmentation_method == 'flipud':
        return [np.flipud(i) for i in data]


def preprocessing(numpy_data):
    # return [tf.image.per_image_standardization(image) for image in numpy_data]
    flatten = np.reshape(numpy_data, (numpy_data.shape[0], -1))
    # print(flatten.shape, numpy_data.shape)

    mean = np.mean(flatten, axis=1)
    stddev = np.std(flatten, axis=1)

    # print(type(numpy_data))
    # print(type(flatten))
    # print(type(mean))
    # print(type(stddev))

    print(flatten.shape)
    print(mean.shape)
    print(stddev.shape)

    try:
        flatten = flatten[:, :, np.newaxis] - mean
        flatten = flatten / stddev
    except Exception as e:
        print("Exception found in", sys._getframe().f_code.co_name, e)

    return flatten


def fold_generator(data, batch_size):
    # TODO need to be edited
    batch_num = int(len(data) / batch_size)
    batches = [data[i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]

    if len(data) > batch_num * batch_size:
        batches.append(data[batch_num * batch_size:])

    return batches


def batch_generator(data, labels, batch_size):
    # pass
    # np.random.shuffle(np.array(np.arange(len(data))))

    batch_num = int(len(data) / batch_size)

    np.random.permutation(data)

    indices = np.array(np.arange(len(data)))
    np.random.shuffle(indices)
    # indices = indices[:batch_size]
    data = data[indices]
    labels = labels[indices]

    batches_x = [data[i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]
    batches_y = [labels[i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]

    if len(data) > batch_num * batch_size:
        batches_x.append(data[batch_num * batch_size:])
        batches_y.append(data[batch_num * batch_size:])

    return batches_x, batches_y
