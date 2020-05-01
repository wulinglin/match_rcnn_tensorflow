# 执行步骤
1.在constant.py文件中更改数据集的地址

2.执行main_data_process.py预处理数据
  先切割视频、 转换成deepfashion数据集、转换成coco数据集

3.执行match-rcnn
  执行python main_mn.py --command training --weights coco即可