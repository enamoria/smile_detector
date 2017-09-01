import os
import sys

import cv2
import matplotlib.colors as mc
import numpy as np
import tensorflow as tf

import CONSTANT


def weight_variable(shape, name='name'):
    initial = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01, dtype=tf.float32), name=name)
    return initial


def bias_variable(shape, name='name'):
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
    # BETTER USE BATCH_NORM
    # TODO Done


def load_data():
    db_path = CONSTANT.ALIGNED_CROPPED_db_path
    images_list = os.listdir(db_path)

    labels = load_labels()

    images = []
    for index, image_name in enumerate(images_list):
        img = cv2.imread(db_path + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            tmp = cv2.resize(img, (90, 90))

            images.append(tmp)
        except Exception as e:
            print("Exception found in", sys._getframe().f_code.co_name, e)

    images = np.asarray(images)  # , dtype=np.float32)
    index = np.random.permutation(len(images))
    images = images[index]
    labels = labels[index]

    return images, labels


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


def augmentation(data, counter):
    # augmentation_method = 'fliplr',
    # if augmentation_method == 'fliplr':
    #     return [np.fliplr(i) for i in data]
    # if augmentation_method == 'flipud':
    #     return [np.flipud(i) for i in data]
    # if augmentation_method == 'rotate90':
    #     return [np.rot90(i, 1) for i in data]
    # if augmentation_method == 'rotate180':
    #     return [np.rot90(i, 2) for i in data]
    # if augmentation_method == 'rotate270':
    #     return [np.rot90(i, 3) for i in data]
    # if np.random.random() >= 0:
    def rgb2grayscale(image):
        # r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        image[:, :, 0] = image[:, :, 0] * 0.3
        image[:, :, 1] = image[:, :, 1] * 0.59
        image[:, :, 2] = image[:, :, 2] * 0.11

        return image

    def rgb2yuv(image):
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16
        U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128
        V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128

        return np.concatenate((Y, U, V), 1)

    def rgb2yiq(image):
        # start_time = time.time()
        conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                      [0.596, -0.274, -0.322],
                                      [0.211, -0.523, 0.312]])

        ''' Better use broadcasting '''
        result = np.empty(image.shape)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.dot(conversion_matrix, image[i, j]) / 255

        # plt.imshow(result)
        # plt.show()
        # print("Convert time:", time.time() - start_time)
        return result

    if counter == 0:
        return data
    if counter == 1:
        return [np.fliplr(i) for i in data]
    if counter == 2:
        return [mc.rgb_to_hsv(i) for i in data]
    if counter == 3:
        return [rgb2grayscale(i) for i in data]
    # if counter == 4:
    #     return [tf.image.random_contrast(i, 0.2, 1.8) for i in data]
    # if counter == 5:
    #     return [np.rot90(i, 3) for i in data]

    return data


def preprocessing(numpy_data):
    shape = numpy_data.shape
    numpy_data = numpy_data.astype(float)
    flatten = np.reshape(numpy_data, (numpy_data.shape[0], -1))

    mean = np.mean(flatten, axis=1)
    stddev = np.std(flatten, axis=1)

    for i in range(flatten.shape[0]):
        flatten[i] = (flatten[i] - mean[i]) / (stddev[i])

    return np.reshape(flatten, (shape[0], shape[1], shape[2], shape[3]))


def fold_generator(data, batch_size):
    # TODO need to be edited. It's fold, not batch. Implementation is right (up to this point)
    batch_num = int(len(data) / batch_size)
    batches = [data[i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]

    if len(data) > batch_num * batch_size:
        batches.append(data[batch_num * batch_size:])

    return batches


def batch_generator(data, labels, batch_size):
    batch_num = int(len(data) / batch_size)

    np.random.permutation(data)

    indices = np.array(np.arange(len(data)))
    np.random.shuffle(indices)

    data = data[indices]
    labels = labels[indices]

    batches_x = [data[i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]
    batches_y = [labels[i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]

    if len(data) > batch_num * batch_size:
        batches_x.append(data[batch_num * batch_size:])
        batches_y.append(data[batch_num * batch_size:])

    return batches_x, batches_y
