import os

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import json
from constant import class_dict, path_head_save

BATCHSIZE = 10
root_path = '/home/eric/data/NUS-WIDE/image'


class DataGeneratorVGG:

    def __init__(self, file_path, _max_example, image_size, classes):
        self.load_data(file_path=file_path)
        self.index = 0
        self.batch_size = BATCHSIZE
        self.image_size = image_size
        self.classes = classes
        self.load_images_labels(_max_example)
        self.num_of_examples = _max_example

    def load_data(self, file_path):
        self.datasets = pd.read_csv(file_path)
        # with open(file_path, 'r') as f:
        #     self.datasets = f.readlines()

    def load_images_labels(self, _max_example):
        m=0
        images = []
        labels = []
        # for i in range(0, len(self.datasets[:_max_example])):
        #     if i==0:continue
        for i,row in self.datasets.head(_max_example).iterrows():
            # data_arr = self.datasets[i].strip().split(',')
            data_arr = [row['path'], row['class'], row['box']]
            image_path = os.path.join(root_path, data_arr[0]).replace("\\", "/")
            if not os.path.exists(image_path):
                m+=1
                continue
            img = Image.open(image_path)
            box = eval(data_arr[-1])
            y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
            img = np.array(img)
            img = img[x1:x2, y1:y2]
            img = cv2.resize(img, (self.image_size[0], self.image_size[1]), Image.ANTIALIAS)
            images.append(img)
            tags = data_arr[1]
            label = int(tags)
            # label = np.zeros((self.classes))
            # for i in range(1, len(tags)):
            #     #         print(word_id[tags[i]])
            #     id = int(word_id[tags[i]])
            #     label[id] = 1
            labels.append(label)
        print(m, 'img path does not exists. ')
        self.images = images
        self.labels = labels

    def get_mini_batch(self):
        while True:
            batch_images = []
            batch_labels = []
            for i in range(self.batch_size):
                if (self.index == len(self.images)):
                    self.index = 0
                batch_images.append(self.images[self.index])
                batch_labels.append(self.labels[self.index])
                self.index += 1
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            yield batch_images, batch_labels


# id_tag_path = 'word_id.txt'
# word_id = {}
# with open(id_tag_path, 'r') as f:
#     words = f.readlines()
#     for item in words:
#         arr = item.strip().split(' ')
#         word_id[arr[1]] = arr[0]

# word_id = class_dict
#
# if __name__ == "__main__":
#     txt_path = path_head_save + 'live2ssd.csv'
#     width, height = 224, 224
#     IMAGE_SIZE = (width, height, 3)
#     classes = 81
#     train_gen = DataGeneratorVGG(txt_path, 100, IMAGE_SIZE, classes)
#     x, y = next(train_gen.get_mini_batch())
#     print(x.shape)
#     print(y.shape)