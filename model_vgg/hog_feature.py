# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:43:04 2019
@author: yexiaohan
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def rgb2gray(rgb):
    return np.matmul(rgb, np.array([0.299, 0.587, 0.114]))


def div(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x, axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell


def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = grad_cell[i, j].flatten()
            ang_list = ang_cell[i, j].flatten()
            left = np.int8(ang_list / 20.0)
            right = left + 1
            right[right >= 8] = 0
            left_rit = (ang_list - 20 * left) / 20.0
            right_rit = 1.0 - left_rit
            binn[left] += left_rit * grad_list
            binn[right] += right_rit * grad_list
            bins[i, j] = binn
    return bins


def hog(img, cell_x, cell_y, cell_w):
    img = rgb2gray(img)
    gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gx, gy)
    ang[ang < 0] = np.pi + ang[ang < 0]
    ang *= (180.0 / np.pi)
    ang[ang >= 180] -= 180
    grad_cell = div(grad, cell_x, cell_y, cell_w)
    ang_cell = div(ang, cell_x, cell_y, cell_w)
    bins = get_bins(grad_cell, ang_cell)
    feature = []
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = []
            tmp.append(bins[i, j])
            tmp.append(bins[i + 1, j])
            tmp.append(bins[i, j + 1])
            tmp.append(bins[i + 1, j + 1])
            tmp -= np.mean(tmp)
            feature.append(tmp.flatten())
    # plt.imshow(grad,cmap=plt.cm.gray)
    # plt.show()
    return np.array(feature).flatten()


img = Image.open('img.jpg')
img = np.array(img)
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
cell_w = 8
cell_x = int(img.shape[0] / cell_w)
cell_y = int(img.shape[1] / cell_w)
feature = hog(img, cell_x, cell_y, cell_w)
print(feature.shape)