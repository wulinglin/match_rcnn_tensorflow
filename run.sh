# (一)在constant.py文件中更改数据集的地址

##（二）预处理数据: 先切割视频、 转换成deepfashion数据集、转换成coco数据集z
#python3 main_data_process.py


## match 部分采用HOG特征，再用matrix_dot点积来算相似度
python3 main_hog_model.py
python3 match_cos_predict.py

# ---------------------以下match rcnn 相关弃用----------------------------
#（三）以下是mask模型：根据video做目标检测
## 1.训练数据（因为选的last所以需要丢一个训练好的模型继续训练）
# python3 main.py --command train --weights last
## 2.测试数据
#python3 main.py --command test --weights last

##（四）以下是match模型：根据video匹配商品库图片
### 1.训练数据特征构造
##python3 main_mn_get_feature_nopair.py  --weights last --command train
### 2.训练数据
##python3 match_model_train_v1.py pair_data_info_train.csv
### 3.测试数据特征构造
#python3 main_mn_get_feature_nopair_v1.py --command test
### 4.测试数据
#python3 match_model_predict.py pair_data_info_test.csv
#
##（五）合并mask和match的结果以提交
#python3 main_merge_result.py
