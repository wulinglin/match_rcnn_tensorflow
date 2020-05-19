import os

from model_vgg.train_v2 import main

from constant import *

# vgg train
from tools.live2vgg_v2 import dataset_prepare

# 数据准备

# mode='train'
# train_file_list = []
# for i in range(1, 7):
#     path_head = '/data/wl_data/live_data/train_dataset_part%d/' % i
#     path_head_save = '/data/wl_data/myspace/train_dataset_part%d/' % i
#     image_path_head = path_head + 'image/'
#     image_annos_path_head = path_head + 'image_annotation/'
#     annos_save_path = path_head_save + 'annos/'
#
#     video_path_raw = path_head + 'video/'
#     video_path_head = path_head_save + 'video_cut/'
#     video_annos_path_head = path_head + 'video_annotation/'
#     if not os.path.exists(path_head_save):
#         os.makedirs(path_head_save)
#
#     dataset_prepare(mode,video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
#                     path_head_save)
#
#     train_file_list.append(path_head_save + 'live2vgg_{}.csv'.format(mode))
#
#
# # vgg validation
# mode='valid'
# valid_file_list = []
# for i in range(1, 5):
#     path_head = '/data/wl_data/live_data/validation_dataset_part%d/' % i
#     path_head_save = '/data/wl_data/myspace/validation_dataset_part%d/' % i
#     image_path_head = path_head + 'image/'
#     image_annos_path_head = path_head + 'image_annotation/'
#     annos_save_path = path_head_save + 'annos/'
#
#     video_path_raw = path_head + 'video/'
#     video_path_head = path_head_save + 'video_cut/'
#     video_annos_path_head = path_head + 'video_annotation/'
#     dataset_prepare(mode,video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
#                     path_head_save)
#     valid_file_list.append(path_head_save + 'live2vgg_{}.csv'.format(mode))
#
#
# # 训练模型
# main(train_file_list, valid_file_list)



# # 本地测试
# mode='train'
# train_file_list = []
# path_head = '/Users/lingwu/Downloads/train_dataset_part5_1/'
# path_head_save = '/Users/lingwu/Downloads/train_dataset_part5_1/'
# image_path_head = path_head + 'image/'
# image_annos_path_head = path_head + 'image_annotation/'
# annos_save_path = path_head_save + 'annos/'
#
# video_path_raw = path_head + 'video/'
# video_path_head = path_head_save + 'video_cut/'
# video_annos_path_head = path_head + 'video_annotation/'
# if not os.path.exists(path_head_save):
#     os.makedirs(path_head_save)
#
# dataset_prepare(mode,video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
#                 path_head_save)
#
# train_file_list.append(path_head_save + 'live2vgg_{}.csv'.format(mode))


# vgg validation
mode='valid'
valid_file_list = []
path_head = '/Users/lingwu/Downloads/train_dataset_part4/'
path_head_save = '/Users/lingwu/Downloads/train_dataset_part4/'
image_path_head = path_head + 'image/'
image_annos_path_head = path_head + 'image_annotation/'
annos_save_path = path_head_save + 'annos/'

video_path_raw = path_head + 'video/'
video_path_head = path_head_save + 'video_cut/'
video_annos_path_head = path_head + 'video_annotation/'
dataset_prepare(mode,video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
                path_head_save)
valid_file_list.append(path_head_save + 'live2vgg_{}.csv'.format(mode))


# 训练模型
# main(train_file_list, valid_file_list)