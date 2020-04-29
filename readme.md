# 执行步骤
1.切割video
 tools/video_split.py
2.转换成deepfashion数据格式
 tools/live2deepfashion.py
3.转换成coco数据格式
 tools/live2coco_v2.py
4.训练目标检测模型
 train.py
5.训练检索模型
   