# from tools.video_split import video_split_and_save_multiprocess
#
# video_split_and_save_multiprocess('/tcdata/train_dataset_part5/video/', '/tcdata/train_dataset_part5/video_cut_0/')

import os

from tools.live2coco_v2 import live2coco_main
from tools.live2deepfashion import video_data_prepare, image_data_prepare
from tools.video_split import video_split_and_save_multiprocess

for i in range(1, 7):
    path_head = '/tcdata_train/train_dataset_part%d/' % i
    path_head_save = '/myspace/train_dataset_part%d/' % i

    if not os.path.exists(path_head):
        print(path_head, 'not exists !')
        continue
    image_path_head = path_head + 'image/'
    image_annos_path_head = path_head + 'image_annotation/'
    annos_save_path = path_head_save + 'annos/'

    video_path_raw = path_head + 'video/'
    video_path_head = path_head_save + 'video_cut/'
    video_annos_path_head = path_head + 'video_annotation/'
    if not os.path.exists(path_head_save):
        os.makedirs(path_head_save)

    print(video_path_raw)
    print('video split')
    # # 先切割视频
    video_split_and_save_multiprocess(video_path_raw, video_path_head)
    # # # # 转换成deepfashion数据集

    video_data_prepare(video_path_head, video_annos_path_head, annos_save_path)
    image_data_prepare(image_path_head, image_annos_path_head, annos_save_path)
    print('tranfer to  deepfashion')

    # 转换成coco数据集
    live2coco_main(video_path_head, video_annos_path_head, annos_save_path, path_head_save)
    print('tranfer to  coco')

for i in range(1, 5):
    path_head = '/tcdata_train/validation_dataset_part%d/' % i
    path_head_save = '/myspace/validation_dataset_part%d/' % i
    if not os.path.exists(path_head):
        print(path_head, 'not exists !')
        continue
    image_path_head = path_head + 'image/'
    image_annos_path_head = path_head + 'image_annotation/'
    annos_save_path = path_head_save + 'annos/'

    video_path_raw = path_head + 'video/'
    video_path_head = path_head_save + 'video_cut/'
    video_annos_path_head = path_head + 'video_annotation/'
    if not os.path.exists(path_head_save):
        os.makedirs(path_head_save)

    print(video_path_raw)
    print('video split')
    # # 先切割视频
    video_split_and_save_multiprocess(video_path_raw, video_path_head)
    # # 转换成deepfashion数据集
    video_data_prepare(video_path_head, video_annos_path_head, annos_save_path)
    image_data_prepare(image_path_head, image_annos_path_head, annos_save_path)
    print('tranfer to  deepfashion')

    # 转换成coco数据集
    live2coco_main(video_path_head, video_annos_path_head, annos_save_path, path_head_save)
    print('tranfer to  coco')

for i in range(5, 7):
    test_video_path_raw = '/tcdata/test_dataset_part%d/video/' % i
    test_video_path_head = '/myspace/test_dataset_part%d/video_cut/' % i
    if os.path.exists(test_video_path_raw):
        print(test_video_path_raw)
        video_split_and_save_multiprocess(test_video_path_raw, test_video_path_head)
        print('test over. ', i)

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
