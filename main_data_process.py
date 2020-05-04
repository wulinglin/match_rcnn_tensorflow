import os

from constant import video_path_head, video_path_raw, test_video_path_raw, test_video_path_head, test_path_head_save, \
    path_head_save
from tools.live2coco_v2 import live2coco_main
from tools.live2deepfashion import video_data_prepare, image_data_prepare
from tools.video_split import video_split_and_save

if not os.path.exists(test_path_head_save):
    os.makedirs(test_path_head_save)

if not os.path.exists(path_head_save):
    os.makedirs(path_head_save)

# # 先切割视频
video_split_and_save(video_path_raw, video_path_head)
if os.path.exists(test_video_path_raw):
    video_split_and_save(test_video_path_raw, test_video_path_head)

# # # 转换成deepfashion数据集
video_data_prepare()
image_data_prepare()

# 转换成coco数据集
live2coco_main()
