# -*- coding: utf-8 -*-
import json
import time

import h5py
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
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
    return score


def array_norm(vec_all):
    """
    两种方法归一化：
    法1的归一化之后的index完全相同，emmm...
    法2和cosine的结果也不尽相同，崩溃。。
    """
    # # 法1 todo mean 是否该是每一列的mean
    # m = np.mean(vec_all)
    # # print('m=',m)
    # mx = np.max(vec_all)
    # mn = np.min(vec_all)
    # return (vec_all-m)/(mx-mn)

    # 法2
    all_norm = np.linalg.norm(vec_all, axis=1, keepdims=True)
    vec_array_norm = vec_all / (all_norm + 1e-10)
    return vec_array_norm


def matrix_dot(image_fea_path, frame_fea_path):
    t_start = time.time()
    # image_fea_path = 'data/image_feature(1).h5'
    # frame_fea_path = 'data/video_feature(1).h5'

    image_feature_json = h5py.File(image_fea_path, 'r')
    video_feature_json = h5py.File(frame_fea_path, 'r')
    video_array = np.array([video_feature_json[index_frame].value for index_frame in video_feature_json.keys()])
    video_array_norm = array_norm(video_array)
    video_name = [index_frame for index_frame in video_feature_json.keys()]
    image_array = np.array([image_feature_json[index_frame].value for index_frame in image_feature_json.keys()])
    image_array_norm = array_norm(image_array)

    image_name = [index_frame for index_frame in image_feature_json.keys()]
    # result = np.dot(video_array,image_array.T)
    result = np.dot(image_array_norm, video_array_norm.T)  # 该方法只对列生效，且快于argmax,所以将其先后顺序颠倒一下
    result_index = np.where(result == np.max(result, axis=0))
    result_index = list(result_index[0])  # 元组，前面的array对应行数，后者对应列数，所以取前面
    # axis=0即列向,如果axis=1即横向
    output_path = constant.test_match_result_path

    result_all_pred_dict = {}
    for k, v in zip(video_name, result_index):
        result_all_pred_dict[k] = image_name[v]
    image_feature_json.close()
    video_feature_json.close()
    print('matrix_dot', result_all_pred_dict)
    with open(output_path, 'w+') as fp:
        json.dump(result_all_pred_dict, fp, ensure_ascii=False)
    print("time end :", time.time() - t_start)
    return result_all_pred_dict


def hzn_cos():
    a = np.arange(9).reshape((3, 3))

    time_start = time.time()

    result_all_pred = []
    count = 0
    result_all_pred_dict = {}

    image_fea_path = 'data/image_feature(1).h5'
    frame_fea_path = 'data/video_feature(1).h5'
    # image_fea_path = 'image_feature.h5'
    # frame_fea_path = 'video_feature.h5'

    image_feature_json = h5py.File(image_fea_path, 'r')
    video_feature_json = h5py.File(frame_fea_path, 'r')
    idx = 0
    for index_frame in video_feature_json.keys():
        idx += 1
        print(idx)
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
        # print(end_time - start_time)
        result_all_pred_dict[index_frame] = temp_index
        count += 1
        # if count % 10 == 0:
        #     print('cos_predict: %d' % count)
    output_path = constant.test_match_result_path
    image_feature_json.close()
    video_feature_json.close()
    print('the cos', result_all_pred_dict)
    with open(output_path, 'w+') as fp:
        json.dump(result_all_pred_dict, fp, ensure_ascii=False)
    print("pross done! time used ", time.time() - time_start)
    return result_all_pred_dict


def valid_accuracy():
    path = '/Users/lingwu/Downloads/validation_gt_round2.json'
    d = {}
    result_cos = hzn_cos()
    result_dot = matrix_dot()

    score_cos, score_dot = 0, 0
    score_cos_v2, score_dot_v2 = 0, 0
    with open(path) as fp:
        content = json.load(fp)
        for key, val in content.items():
            d[key] = val['item_id']
            # print(key, val.keys(), val) # 'image_anno', 'item_id', 'frame_index', 'instance_id_count', 'frame_anno']
    print(d)
    for k, v in result_cos.items():
        # score_dot_v2 score_cos_v2是验证非valid3的正确情况的
        if k in d:
            if v == d[k]:
                score_cos += 1
        # score_dot score_cos是验证valid3的正确情况的
        if k == v:
            score_cos_v2 += 1

    print('finish cos. ')
    for k, v in result_dot.items():
        # score_dot_v2 score_cos_v2是验证非valid3的正确情况的
        if k in d:
            if v == d[k]:
                score_dot += 1
        # score_dot score_cos是验证valid3的正确情况的
        if k == v:
            score_dot_v2 += 1
    result_cos = {}
    print(score_dot, score_cos, 'dot: {}/{} ;'.format(score_dot_v2, len(result_dot)),
          'cos: {}/{}'.format(score_cos_v2, len(result_cos)))


if __name__ == '__main__':
    # valid_accuracy()
    # # main_cosine_with_multiprocessing()
    #
    matrix_dot(image_fea_path=constant.test_path_head_save + 'image_feature.h5',
               frame_fea_path=constant.test_path_head_save + 'video_feature.h5')
    # todo 为什么只能列不能行

# def main_cosine_with_multiprocessing():
#     pass
# start_time = time.time()
# pool = multiprocessing.Pool(processes=5)  # 创建5个进程
# time_start = time.time()
# image_fea_path = 'data/image_feature.h5'
# frame_fea_path = 'data/video_feature.h5'
#
# image_feature_json = h5py.File(image_fea_path, 'r')
# video_feature_json = h5py.File(frame_fea_path, 'r')
# result_all_pred_dict = {}
#
# def calculate_cos(image_feature_json, frame_feature, temp_index):
#     print(222, index_frame)
#     # frame_feature = video_feature_json[index_frame].value
#     # temp_index = list(image_feature_json.keys())[0]
#     temp_value = 0
#     for index_image in image_feature_json.keys():
#         image_feature = image_feature_json[index_image].value
#         temp_cos = cos_sim(frame_feature, image_feature)
#         print(temp_cos)
#         if temp_cos > temp_value:
#             temp_value = temp_cos
#             temp_index = index_image
#     return temp_index
#     # result_all_pred_dict[index_frame] = temp_index
#
# # for i in range(10):
# #     print(i, time.time() - time_start)
# for index_frame in video_feature_json.keys():
#     print(index_frame)
#     frame_feature = video_feature_json[index_frame].value
#     temp_index = list(image_feature_json.keys())[0]
#     # temp_value = 0
#     cos = pool.apply_async(calculate_cos, (image_feature_json, frame_feature, temp_index,)).get()
#     # result_all_pred_dict[index_frame] = cos
#     # print(result_all_pred_dict)
#
# pool.close()
# pool.join()
#
# output_path = constant.test_match_result_path
# image_feature_json.close()
# video_feature_json.close()
#
# with open(output_path, 'w+') as fp:
#     json.dump(result_all_pred_dict, fp, ensure_ascii=False)
# print("pross done! time used ", time.time() - time_start)
