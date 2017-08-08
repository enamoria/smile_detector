"""
This script represent a face detector, using mtcnn. Source: https://github.com/pangyupo/mxnet_mtcnn_face_detection
Results seem to be plausible for future use
"""

import os
import sys
import cv2
import CONSTANT
from Face_Region_Detection.mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector

sys.path.append(CONSTANT.ROOT_DIR)

genki4k_image_list = [f for f in os.listdir(CONSTANT.GENKI4K_db_path) if os.path.isfile(os.path.join(CONSTANT.GENKI4K_db_path, f))]

face_detector = MtcnnDetector(model_folder="./mxnet_mtcnn_face_detection/model")

for image_name in genki4k_image_list:
    try:
        img = cv2.imread(CONSTANT.GENKI4K_db_path + image_name)
        result = face_detector.detect_face(img)

        if result is not None:

            total_boxes = result[0]
            points = result[1]

            # extract aligned face chips
            chips = face_detector.extract_image_chips(img, points, 96, 0.37)
            for i, chip in enumerate(chips):
                # cv2.imshow('chip_' + str(i), chip)
                # cv2.imwrite('./aligned_face/chip_' + str(i) + '.png', chip)
                cv2.imwrite('./aligned_face/' + image_name, chip)

        print(image_name)

            # draw = img.copy()
            # for b in total_boxes:
            #     cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
            #
            # for p in points:
            #     for i in range(5):
            #         cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

            # cv2.imshow("detection result", draw)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    except Exception as e:
        print("exception: ", e)

    # cv2.waitKey(0)
