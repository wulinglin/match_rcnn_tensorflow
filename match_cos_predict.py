# -*- coding: utf-8 -*-
import json
import h5py
import time

import numpy as np
import tensorflow as tf

import constant

feature1_name = ".rois_feature.npy"
feature2_name = ".fpn5_feature.npy"


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def cosine(q, a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    pooled_mul_12 = tf.reduce_sum(q * a, 1)
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
    return score


if __name__ == '__main__':

    time_start = time.time()

    result_all_pred = []
    count = 0
    result_all_pred_dict = {}

    image_fea_path = 'image_feature.h5'
    frame_fea_path = 'video_feature.h5'

    image_feature_json = h5py.File(image_fea_path, 'r')
    video_feature_json = h5py.File(frame_fea_path, 'r')

    for index_frame in video_feature_json.keys():
        frame_feature = video_feature_json[index_frame].value
        temp_index = list(image_feature_json.keys())[0]
        temp_value = 0
        start_time = time.time()
        for index_image in image_feature_json.keys():
            image_feature = image_feature_json[index_image].value
            temp_cos = cos_sim(frame_feature, image_feature)
            if temp_cos > temp_value:
                temp_value = temp_cos
                temp_index = index_image
        end_time = time.time()
        print(end_time - start_time)
        result_all_pred_dict[index_frame] = temp_index
        count += 1
        if count % 10 == 0:
            print('cos_predict: %d' % count)
    output_path = constant.test_match_result_path
    image_feature_json.close()
    video_feature_json.close()

    with open(output_path, 'w+') as fp:
        json.dump(result_all_pred_dict, fp, ensure_ascii=False)
    print("pross done! time used ", time.time() - time_start)
