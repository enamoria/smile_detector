"""
This script is a face detector using provided pre-trained Haar_cascade
Results aren't good enough to be used in the future
"""

import numpy as np
import os, sys
import cv2
import CONSTANT
sys.path.append(CONSTANT.ROOT_DIR)

# Load the xml pre-train cascade classifier
def face_region_detector(pic_name):
    face_cascade = cv2.CascadeClassifier(CONSTANT.EMOTION_DIRECTORY_PATH + 'Face_Region_Detection/haar_cascades/haarcascade_frontalface_default.xml')

    img = cv2.imread(pic_name)
    # img.resize((CONSTANT.IMAGE_RESIZE_SHAPE[0], CONSTANT.IMAGE_RESIZE_SHAPE[1]))

    cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_rois = []  # Store rectangle(s) which contains face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        face_rois.append((x, y, x+w, y+h))

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return face_rois

# face_region_detector('picture.jpg')

# import numpy as np
# import cv2
#
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# cap = cv2.VideoCapture(0)
#
# while 1:
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = img[y:y + h, x:x + w]
#
#     cv2.imshow('img', img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
# cv2.cv2.destroyAllWindows()
