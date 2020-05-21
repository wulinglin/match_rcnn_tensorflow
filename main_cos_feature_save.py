import os
import h5py
import cv2
import time
import constant
import tensorflow as tf

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.box_utils import draw_box_in_img
from help_utils import tools


def detect(sess, real_test_imgname_list, h5_file, flatten_feature, img_batch, img_plac):
    # 1. preprocess img

    for i, a_img_name in enumerate(real_test_imgname_list):

        raw_img = cv2.imread(a_img_name + '/0.jpg')
        resized_img, flatten_feature_result = \
            sess.run(
                [img_batch, flatten_feature],
                feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
            )
        h5_file[a_img_name] = flatten_feature_result[0]


def get_mn_test_image_pair():
    test_video_path_head, test_image_path=constant.test_video_path_head, constant.test_image_path
    test_video_path_list, test_img_path_list = [],[]
    for video_path in os.listdir(test_video_path_head):
        video_path_ = test_video_path_head + video_path + '/' + '0.jpg'

        test_video_path_list.append(video_path_)
    for img_path in os.listdir(test_image_path):
        image_path_ = test_image_path + img_path + '/' + '0.jpg'
        test_img_path_list.append(image_path_)
    return test_video_path_list,test_img_path_list


def pca(x, dim=2):
    '''
        x:输入矩阵
        dim:降维之后的维度数
    '''
    with tf.name_scope("PCA"):

        m,n= tf.to_float(x.shape[0]),tf.to_int32(x.shape[1])
        assert not tf.assert_less(dim,n)
        mean = tf.reduce_mean(x,axis=1)
        # 去中心化
        x_new = x - tf.reshape(mean,(-1,1))
        # 无偏差的协方差矩阵
        cov = tf.matmul(x_new,x_new,transpose_a=True)/(m - 1)
        # 计算特征分解
        e,v = tf.linalg.eigh(cov,name="eigh")
        # 将特征值从大到小排序，选出前dim个的index
        e_index_sort = tf.math.top_k(e,sorted=True,k=dim)[1]
        # 提取前排序后dim个特征向量
        v_new = tf.gather(v,indices=e_index_sort)
        # 降维操作
        pca = tf.matmul(x_new,v_new,transpose_b=True)
    return pca


def main_match_test():

    test_video_path_list, test_img_path_list = get_mn_test_image_pair()

    test_video_path_list, test_img_path_list = test_video_path_list, test_img_path_list

    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)

    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not GBR
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0) # [1, None, None, 3]

    flatten_feature, _ = faster_rcnn.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch=None)
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = faster_rcnn.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        count = 0
        print(len(test_video_path_list))
        feature_index_list = []
        # test_video_path_list = test_video_path_list[:500]
        h5_file = h5py.File(constant.test_cos_frame_feature_path, 'w')
        for p1 in test_video_path_list:
            if '.DS_Store' in p1:
                continue
            p1_video_path = constant.test_video_path_head + p1.split("/")[-2]

            count += 1
            if count % 50 == 0:
                print('p1 match test', count)
            feature_index_list.append(p1_video_path)

        for i, a_img_name in enumerate(feature_index_list):
            raw_img = cv2.imread(a_img_name + '/40.jpg')
            resized_img, flatten_feature_result = \
                sess.run(
                    [img_batch, flatten_feature],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                )
            print(i)
            # a = sess.run(pca(flatten_feature_result))

            h5_file[a_img_name.split('/')[-1]] = flatten_feature_result[0]

        # detect(faster_rcnn, feature_index_list, h5_file, flatten_feature, img_batch, img_plac)

        h5_file.close()
        # for feature in video_feature_list:
        #     video_feature_json[feature[0]] = feature[1]
        # with open('video_feature.json', 'w+') as f:
        #     json.dump(video_feature_json, f)
        print('-'*30, len(test_img_path_list))
        count = 0
        feature_index_list = []
        h5_file = h5py.File(constant.test_cos_image_feature_path, 'w')

        # test_img_path_list = test_img_path_list[:500]
        for p2 in test_img_path_list:
            if '.DS_Store' in p2:
                continue
            p2_image_path = constant.test_image_path + p2.split("/")[-2]
            feature_index_list.append(p2_image_path)

            count += 1
            if count % 50 == 0:
                print('p2 match test', count)

        for i, a_img_name in enumerate(feature_index_list):
            raw_img = cv2.imread(a_img_name + '/0.jpg')
            resized_img, flatten_feature_result = \
                sess.run(
                    [img_batch, flatten_feature],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                )
            print(i)
            h5_file[a_img_name.split('/')[-1]] = flatten_feature_result[0]

        # detect(faster_rcnn, feature_index_list, h5_file, flatten_feature, img_batch, img_plac)
        h5_file.close()
        # for feature in video_feature_list:
        #     image_feature_json[feature[0]] = feature[1]
        # with open('image_feature.json', 'w+') as f:
        #     json.dump(image_feature_json, f)


if __name__ is '__main__':
    main_match_test()