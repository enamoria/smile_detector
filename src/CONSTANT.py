""" Constant """
import os
# GENKI4K_db_path = 'F:/Datasection/Learning_TensorFlow/TensorFlow.org_Tutorial/src/Emotion/GENKI4K/files/'
# GENKI4K_labels_path = 'F:/Datasection/Learning_TensorFlow/TensorFlow.org_Tutorial/src/Emotion/GENKI4K/'

# This is your Project Root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# OTHER PATH RELATED TO ROOT_DIR
GENKI4K_db_path = ROOT_DIR + "/data/genki4k/files/"
GENKI4K_labels_path = ROOT_DIR + "/data/genki4k/labels.txt"

FLATTEN_SHAPE = 192 * 178 * 3
IMAGE_SHAPE = [192, 178, 3]
NUM_CLASS = 2
