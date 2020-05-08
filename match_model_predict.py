# -*- coding: utf-8 -*-
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import constant
from tools import data_utils

mode_name = "matchcnn"
mode_list = ["matchcnn"]
if mode_name not in mode_list:
    print("the current model has : ", mode_list)
    exit()

DL_MODEL_ROOT = "model_all"
MODEL_RESULT = "predict_result"

if not os.path.exists(MODEL_RESULT):
    os.makedirs(MODEL_RESULT)

feature1_name = ".rois_feature.npy"
feature2_name = ".fpn5_feature.npy"


class matchcnn_model():
    def __init__(self, graph, target_name):
        with graph.as_default():
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            sess_config.allow_soft_placement = True
            self.sess = tf.Session(config=sess_config, graph=graph)
            self.graph = graph
            MODEL_DIR = DL_MODEL_ROOT + "/" + target_name + "/"
            if not os.path.exists(MODEL_DIR):
                print(MODEL_DIR + " is not extis! please run dl_train_new at first!")
                # os.makedirs(MODEL_PATH)
            print(MODEL_DIR)
            checkpoint_file = tf.train.latest_checkpoint(MODEL_DIR)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)
            print("successfullly load mdoel form " + MODEL_DIR)

            # Get the placeholders from the graph by name
            self.img1_fea1_ph = graph.get_operation_by_name("video_fea1").outputs[0]
            self.img2_fea1_ph = graph.get_operation_by_name("img_fea1").outputs[0]

            self.img1_fea2_ph = graph.get_operation_by_name("video_fea2").outputs[0]
            self.img2_fea2_ph = graph.get_operation_by_name("img_fea2").outputs[0]

            self.is_train = graph.get_operation_by_name("is_train").outputs[0]
            self.keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
            self.prediction_proba = graph.get_operation_by_name("output_layer/softmax_proba").outputs[0]

    def predict(self, img1_fea1, img2_fea1, img1_fea2, img2_fea2):
        feed_dict = dict()
        feed_dict[self.img1_fea1_ph] = img1_fea1
        feed_dict[self.img2_fea1_ph] = img2_fea1
        feed_dict[self.img1_fea2_ph] = img1_fea2
        feed_dict[self.img2_fea2_ph] = img2_fea2
        feed_dict[self.is_train] = False
        feed_dict[self.keep_prob] = 1.0

        prediction_proba = self.sess.run(self.prediction_proba, feed_dict)
        return prediction_proba


def get_feature_info(df):
    video_feature_path = df['video_feature_path'].tolist()
    video_frame_name = df["video_frame_name"].tolist()
    img_feature_path = df["img_feature_path"].tolist()
    img_name = df["img_name"].tolist()
    # train_label = df["label"].tolist()

    frame_fea1_path = []
    image_fea1_path = []
    frame_fea2_path = []
    image_fea2_path = []
    # label_input = []
    # for v1, v2, i1, i2, label_tmp in zip(video_feature_path, video_frame_name, img_feature_path, img_name, train_label):
    for v1, v2, i1, i2 in zip(video_feature_path, video_frame_name, img_feature_path, img_name
                              ):
        frame_fea1_path.append(v1.strip() + "/" + v2 + feature1_name)
        image_fea1_path.append(i1.strip() + "/" + i2 + feature1_name)

        frame_fea2_path.append(v1.strip() + "/" + v2 + feature2_name)
        image_fea2_path.append(i1.strip() + "/" + i2 + feature2_name)

    return frame_fea1_path, image_fea1_path, frame_fea2_path, image_fea2_path


if __name__ == '__main__':

    time_start = time.time()
    graph = tf.Graph()
    model = matchcnn_model(graph, mode_name)

    test_df = pd.read_csv(constant.path_head_save+"pair_data_info_test.csv", encoding="utf8")
    frame_fea1_path, image_fea1_path, frame_fea2_path, image_fea2_path = get_feature_info(test_df)
    result_all_pred = []
    count = 0
    result_all_pred_dict = {}

    for f1, f2, i1, i2 in zip(frame_fea1_path, frame_fea2_path, image_fea1_path, image_fea2_path):
        video_id = data_utils.get_item_id_by_path(f1)
        item_id = data_utils.get_item_id_by_path(i1)
        frame_fea1 = np.load(f1)
        frame_fea2 = np.load(f2)
        image_fea1 = np.load(i1)
        image_fea2 = np.load(i2)
        prediction = model.predict([frame_fea1], [image_fea1], [frame_fea2], [image_fea2])

        print('prediction:', count,prediction[0][1])

        result_all_pred.append(prediction[0][1])
        if video_id in result_all_pred_dict:
            result_all_pred_dict[video_id]['item_ids'].append(item_id)
            result_all_pred_dict[video_id]['pred_scores'].append(prediction[0][1])
        else:
            result_all_pred_dict[video_id] = {'item_ids': [item_id],
                                              'pred_scores': [prediction[0][1]]}
        count += 1
        if count % 50 == 0:
            print("model has prossed number :" + str(count))

    result_dict = {}
    for key, val in result_all_pred_dict.items():
        result_dict[key] = val['item_ids'][val['pred_scores'].index(max(val['pred_scores']))]

    with open(constant.test_match_result_path, 'w+') as fp:
        json.dump(result_dict, fp, ensure_ascii=False)
    test_df["predict_label"] = result_all_pred

    pred_label = []
    for ii in result_all_pred:
        if ii > 0.5:
            pred_label.append(1)
        else:
            pred_label.append(0)
    # print(classification_report(test_df['label'].tolist(), pred_label))

    # print(u"混淆矩阵")
    # print(confusion_matrix(test_df['label'].tolist(), pred_label))  ## labels = ["0","1"]))

    # 保存结果
    test_df = test_df.sort_values(["predict_label"], ascending=False)
    test_df.to_csv(MODEL_RESULT + "/" + mode_name + "_predict.csv", index=False, encoding="utf8")
    print("pross done! time used ", time.time() - time_start)
