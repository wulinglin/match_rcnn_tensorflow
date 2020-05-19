import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops

def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        idx = 0
        # for i in image:
            # try:
            #     i = cv2.resize(i, size)
            # except Exception:
            #     if idx-1>=0:
            #         i = images[idx-1]
            #     else:
            #         i = images[idx + 1]
            #     i = cv2.resize(i, size)
            # images.append(i)
            # idx += 1
        images = np.array(images)
        return images

def print_answer(argmax):
    with open("./data/model/index_word.txt","r",encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
        
    print(synset[argmax])
    return synset[argmax]