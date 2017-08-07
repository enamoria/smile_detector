# import urllib2
from urllib.request import urlopen
from io import BytesIO
from PIL import Image
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# url = ''
# f = open('F:\Datasection\Learning_TensorFlow\TensorFlow.org_Tutorial\src\Emotion\GENKI4K\\files\\file0001.jpg')
# # im = Image.open(BytesIO(urlopen(url).read()))
# # print(f.read())
im_from_dir = Image.open('F:\Datasection\Learning_TensorFlow\TensorFlow.org_Tutorial\src\Emotion\GENKI4K\\files\\file0001.jpg')
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

x = np.array([[[1, 2, 3], [2, 3, 4]], [[1, 4, 3], [2, 45, 5]], [[4, 5, 56], [6, 7, 8]]])
y = np.array([1, 2, 3])

np.insert(x, y)
print(x)
# x = [[[1, 2, 3], [2, 3, 4]], [[1, 4, 3], [2, 45, 5]], [[4, 5, 56], [6, 7, 8]]]
# print(x.shape)
# print(np.array(x).shape)
# print((np.reshape(x, (x.shape[0], -1))).shape)
