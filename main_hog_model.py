import os

import constant
# vgg train
from lib.get_hog_feature import save_hog_feature
from tools.live2vgg_v2 import dataset_prepare_v3_for_test

# 数据准备

"""本地测试"""
mode = 'train'
train_file_list = []
# path_head = '/Users/lingwu/Downloads/train_dataset_part5_1/'
# path_head_save = '/Users/lingwu/Downloads/train_dataset_part5_1/'
# df_list = dataset_prepare_v2(mode, video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
#                           path_head_save)
df_list = dataset_prepare_v3_for_test(constant.test_video_path_head, constant.test_image_path)
save_hog_feature(df_list,constant.test_path_head_save + 'image_feature.h5',constant.test_path_head_save + 'video_feature.h5')
