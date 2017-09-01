import CONSTANT
import os

db_path = CONSTANT.ALIGNED_CROPPED_db_path
images_list = os.listdir(db_path)

images_list = [int(image_name[4:8]) for image_name in images_list]
# print(images_list)
# print(len(images_list))
prev = 0
noise = []
for index, value in enumerate(images_list):
    if value - prev != 1:
        for i in range(prev+1, value):
            print(i)
            noise.append(i-1)
    prev = value

labels_path = CONSTANT.GENKI4K_labels_path
f_labels = open(labels_path, "r")
f_labels_modified = open(labels_path + "_modified", "w")

i = 0
while True:
    sample = f_labels.readline().strip("\n")
    if sample == "":
        break

    if i not in noise:
        f_labels_modified.write(sample + "\n")

    i += 1
