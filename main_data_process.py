# from tools.video_split import video_split_and_save_multiprocess
#
# video_split_and_save_multiprocess('/tcdata/train_dataset_part5/video/', '/tcdata/train_dataset_part5/video_cut_0/')

from tools.live2vgg_v2 import dataset_prepare

# for i in range(1, 7):
#     path_head = '/tcdata_train/train_dataset_part%d/' % i
#     path_head_save = '/myspace/train_dataset_part%d/' % i
#
#     if not os.path.exists(path_head):
#         print(path_head, 'not exists !')
#         continue
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
#     print(video_path_raw)
#     print('video split')
#     # # 先切割视频
#     video_split_and_save_multiprocess(video_path_raw, video_path_head)
#     # # # # 转换成deepfashion数据集
#
#     video_data_prepare(video_path_head, video_annos_path_head, annos_save_path)
#     image_data_prepare(image_path_head, image_annos_path_head, annos_save_path)
#     print('tranfer to  deepfashion')
#
#     # 转换成coco数据集
#     live2coco_main(video_path_head, video_annos_path_head, annos_save_path, path_head_save)
#     print('tranfer to  coco')
#
# for i in range(1, 5):
#     path_head = '/tcdata_train/validation_dataset_part%d/' % i
#     path_head_save = '/myspace/validation_dataset_part%d/' % i
#     if not os.path.exists(path_head):
#         print(path_head, 'not exists !')
#         continue
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
#     print(video_path_raw)
#     print('video split')
#     # # 先切割视频
#     video_split_and_save_multiprocess(video_path_raw, video_path_head)
#     # # 转换成deepfashion数据集
#     video_data_prepare(video_path_head, video_annos_path_head, annos_save_path)
#     image_data_prepare(image_path_head, image_annos_path_head, annos_save_path)
#     print('tranfer to  deepfashion')
#
#     # 转换成coco数据集
#     live2coco_main(video_path_head, video_annos_path_head, annos_save_path, path_head_save)
#     print('tranfer to  coco')
#
# # for i in range(5, 7):
# test_video_path_raw = '/tcdata/test_dataset_fs/video/'
# test_video_path_head = '/myspace/test_dataset_fs/video_cut/'
# if os.path.exists(test_video_path_raw):
#     print(test_video_path_raw)
#     video_split_and_save_multiprocess(test_video_path_raw, test_video_path_head)
# else:
#     print('does not exists:', test_video_path_raw)

# from tools.video_new import cut_video_with_multiprocessing
#
# cut_video_with_multiprocessing('/tcdata/train_dataset_part5/video/',thead_pool_size=6)

"""
多进程测试：
from tools.video_split import video_split_and_save_multiprocess
video_split_and_save_multiprocess('/Downloads/train_dataset_part5_1/video/',
                                  '/myspace/train_dataset_part5/video_cut_0/')
                                  
from tools.video_split import video_split_and_save
video_split_and_save('/Downloads/train_dataset_part5_1/video/',
                                  '/myspace/train_dataset_part5/video_cut_1/')
"""

from constant import *

# vgg train
mode='train'
dataset_prepare(mode,video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
                path_head_save)

# vgg validation
mode='valid'
dataset_prepare(mode,valid_video_path_head, valid_image_path_head, valid_video_annos_path_head, valid_image_annos_path_head,
                valid_path_head_save)

# # 转换成deepfashion数据集
# video_data_prepare(video_path_head, video_annos_path_head, annos_save_path)
# image_data_prepare(image_path_head, image_annos_path_head, annos_save_path)
# live2coco_main(video_path_head, video_annos_path_head, annos_save_path, path_head_save)
# print('tranfer to  deepfashion')
#
# json_name = path_head_save + 'train.json'
# import json
# with open(json_name, 'r') as f:
#     content = json.load(f)
#     print(len(content['images']))
#     for i in content['images'][:2]:
#         print(i)
