# from tools.video_split import video_split_and_save_multiprocess
#
# video_split_and_save_multiprocess('/tcdata/train_dataset_part5/video/', '/tcdata/train_dataset_part5/video_cut_0/')

import os

from tools.video_split import video_split_and_save_multiprocess

# if not os.path.exists(test_path_head_save):
#    os.makedirs(test_path_head_save)

for i in range(1, 7):
    path_head = '/tcdata/train_dataset_part%d/' % i
    path_head_save = '/myspace/train_dataset_part%d/' % i

    image_path_head = path_head + 'image/'
    image_annos_path_head = path_head + 'image_annotation/'
    annos_save_path = path_head_save + 'annos/'

    video_path_raw = path_head + 'video/'
    video_path_head = path_head_save + 'video_cut/'
    video_annos_path_head = path_head + 'video_annotation/'
    if not os.path.exists(path_head_save):
        os.makedirs(path_head_save)

    print('train 111')
    # # 先切割视频
    video_split_and_save_multiprocess(video_path_raw, video_path_head)
    # # # # 转换成deepfashion数据集
    # video_data_prepare(video_path_head, video_annos_path_head, annos_save_path)
    # image_data_prepare(image_path_head, image_annos_path_head, annos_save_path)
    # print('train 222')
    #
    # # 转换成coco数据集
    # live2coco_main(video_path_head, video_annos_path_head, annos_save_path, path_head)
    # print('train over {}'.format(i))

for i in range(1, 5):
    path_head = '/tcdata/validation_dataset_part%d/' % i
    path_head_save = '/myspace/validation_dataset_part%d/' % i

    image_path_head = path_head + 'image/'
    image_annos_path_head = path_head + 'image_annotation/'
    annos_save_path = path_head_save + 'annos/'

    video_path_raw = path_head + 'video/'
    video_path_head = path_head_save + 'video_cut/'
    video_annos_path_head = path_head + 'video_annotation/'
    if not os.path.exists(path_head_save):
        os.makedirs(path_head_save)

    print('valid 111')
    # # 先切割视频
    video_split_and_save_multiprocess(video_path_raw, video_path_head)
    # # # 转换成deepfashion数据集
    # video_data_prepare(video_path_head, video_annos_path_head, annos_save_path)
    # image_data_prepare(image_path_head, image_annos_path_head, annos_save_path)
    # print('valid 222')
    #
    # # 转换成coco数据集
    # live2coco_main(video_path_head, video_annos_path_head, annos_save_path, path_head)
    print('valid over {}'.format(i))

test_video_path_raw = '/tcdata/test_dataset_3w/video/'
test_video_path_head = '/myspace/test_dataset_3w/video_cut/'
if os.path.exists(test_video_path_raw):
    video_split_and_save_multiprocess(test_video_path_raw, test_video_path_head)
    print('test over. ')

# from tools.video_new import cut_video_with_multiprocessing
#
# cut_video_with_multiprocessing('/tcdata/train_dataset_part5/video/',thead_pool_size=6)

"""
多进程测试：
from tools.video_split import video_split_and_save_multiprocess
video_split_and_save_multiprocess('/Users/lingwu/Downloads/train_dataset_part5_1/video/',
                                  '/Users/lingwu/myspace/train_dataset_part5/video_cut_0/')
                   
                                  
from tools.video_split import video_split_and_save
video_split_and_save('/Users/lingwu/Downloads/train_dataset_part5_1/video/',
                                  '/Users/lingwu/myspace/train_dataset_part5/video_cut_1/')
"""
