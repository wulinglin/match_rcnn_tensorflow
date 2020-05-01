# 执行步骤
1.在constant.py文件中更改数据集的地址

2.执行main_data_process.py预处理数据
  python3 main_data_process.py
  先切割视频、 转换成deepfashion数据集、转换成coco数据集

3.执行mask-rcnn模型
  python3 main.py --command training --weights last
 
4.保存特征到本地
  python3 main_mn_get_feature.py  --weights last

5.执行match-rcnn检索模型
  python3 main_mn.py --command inference --weights last