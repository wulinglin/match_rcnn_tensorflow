import os

import constant
# vgg train
from lib.get_hog_feature import save_hog_feature
from tools.live2vgg_v2 import dataset_prepare_v3_for_test, dataset_prepare_v2, dataset_prepare_v4

# 数据准备

mode = 'train'
train_file_list = []
path_head = '/Users/lingwu/Downloads/train_dataset_part5_1/'
path_head_save = '/Users/lingwu/Downloads/train_dataset_part5_1/'
image_path_head = path_head + 'image/'
image_annos_path_head = path_head + 'image_annotation/'
annos_save_path = path_head_save + 'annos/'

video_path_raw = path_head + 'video/'
video_path_head = path_head_save + 'video_cut/'
video_annos_path_head = path_head + 'video_annotation/'
if not os.path.exists(path_head_save):
    os.makedirs(path_head_save)


# 本地测试multi-view（rcca）版本的hog
df_list = dataset_prepare_v4(video_path_head, image_path_head, video_annos_path_head, image_annos_path_head)
save_rcca_hog_feature(df_list,constant.test_path_head_save + 'image_feature_rcaa.h5',constant.test_path_head_save + 'video_feature_rcaa.h5')

# 本地测试hog
# df_list = dataset_prepare_v2(mode, video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
#                              path_head_save)
# save_hog_feature(df_list,constant.test_path_head_save + 'image_feature.h5',constant.test_path_head_save + 'video_feature.h5')


# 线上测试hog
df_list = dataset_prepare_v3_for_test(constant.test_video_path_head, constant.test_image_path)
save_hog_feature(df_list,constant.test_path_head_save + 'image_feature.h5',constant.test_path_head_save + 'video_feature.h5')