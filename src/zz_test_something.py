from __future__ import print_function
from src.training_utils import weight_variable
# import urllib2
# from urllib.request import urlopen
# from io import BytesIO
# from PIL import Image
# from scipy.misc import imread
# import matplotlib.pyplot as plt
import numpy as np
from src.training_utils import batch_generator as bg

import tensorflow as tf

# url = ''
# f = open('F:\Datasection\Learning_TensorFlow\TensorFlow.org_Tutorial\src\Emotion\GENKI4K\\files\\file0001.jpg')
# # im = Image.open(BytesIO(urlopen(url).read()))
# # print(f.read())
# im_from_dir = Image.open('F:\Datasection\Learning_TensorFlow\TensorFlow.org_Tutorial\src\Emotion\GENKI4K\\files\\file0001.jpg')
# # plt.imshow(im, cmap='Greys_r')
# print(im_from_dir)
#
# print(imread('F:\Datasection\Learning_TensorFlow\TensorFlow.org_Tutorial\src\Emotion\GENKI4K\\files\\file0001.jpg'))
# plt.imshow(im_from_dir)
# plt.show()

# images = np.empty(shape=(0,0))

# x = tf.placeholder(tf.float32, shape=[9, 3, 1])
# print(tf.shape(x))
# dim = tf.reduce_prod(tf.shape(x)[1:])
# x2 = tf.reshape(x, [-1, dim])

# x = np.array([[[1, 2, 3], [2, 3, 4]], [[1, 4, 3], [2, 45, 5]], [[4, 5, 56], [6, 7, 8]]])
# y = np.array([1, 2, 3])
#
# np.insert(x, y)
# print(x)

# x = [[[1, 2, 3], [2, 3, 4]], [[1, 4, 3], [2, 45, 5]], [[4, 5, 56], [6, 7, 8]]]
# print(x.shape)
# print(np.array(x).shape)
# print((np.reshape(x, (x.shape[0], -1))).shape)

# x = np.array([1, 2, 3])
# x.astype(float)
#
# print(x)

# xxx = np.array(np.arange(1, 22))
# yyy = bg(xxx, 3)
#
# print(yyy[0])

# xx = 3
# print(xx)
#
# def test_pass_by_value(xxx):
#     xxx = 5
#     return xxx
#
# test_pass_by_value(xx)
# print(xx)

# A = np.diag([1, 2, 3])
# print(A)
# np.fliplr(A)
# print(A)
# A = np.fliplr(A)
# print(A)

# A = np.array((1, 2, 3, 4, 5, 6, 7, 8))
# print(A)
# index = 4
# A = np.concatenate((A[:index+1], [A[index]], A[index+1:]))
# print(A)

# x = tf.Variable(3)
# y = tf.Variable(4)
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     # print(y.eval(sess))
#     print(sess.run(tf.add(x, y)))
#
# regularization = tf.Variable(0, dtype=tf.float32)
# weight = weight_variable([11, 11, 3, 32])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(regularization))
#     print(sess.run(tf.add(tf.nn.l2_loss(weight), regularization)))

# a = [1, 2, 3, 4, 4]
# f = open("xxx", "w")
# # print
# print(".".join(str(i) for i in a))
# f.write(".".join(str(i) for i in a))
# f.close()

a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9,10])

index = np.random.permutation(len(a))
print(a)
print(b)

print(index)
print(a[index])
print(b[index])