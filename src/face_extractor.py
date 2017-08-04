from src.Emotion.Face_Region_Detection.face_region_detector import face_region_detector
from PIL import Image

import numpy as np

import cv2
import src.Emotion.CONSTANT as CONSTANT
import os

def face_extractor(db_path=CONSTANT.GENKI4K_db_path):
    images_list = os.listdir(db_path)

    for image_name in images_list:
        face_roi = face_region_detector(db_path + image_name)

        img = Image.open(db_path + image_name)

        try:
            (left, top, right, bottom) = face_roi[0]
            # TODO: handling multiple faces in a picture
            img.crop((left, top, right, bottom)).save("./cropped_face/" + image_name)

        except Exception:
            pass



face_extractor()
