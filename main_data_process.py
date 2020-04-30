from constant import video_path_head, video_path_raw
from tools.live2coco_v2 import live2coco_main
from tools.live2deepfashion import video_data_prepare, image_data_prepare
from tools.video_split import video_split_and_save

# # 先 切割视频
# video_split_and_save(video_path_raw, video_path_head)

# # # 转换成deepfashion数据集
# video_data_prepare()
# image_data_prepare()

# 转换成coco数据集
live2coco_main()
