import json
import os

import pandas as pd
import cv2
from sklearn.utils import shuffle

# import constant
# from constant import video_path_head, image_path_head
import constant

pd.set_option('expand_frame_repr', False)
pd.set_option("display.max_rows", 100)


def get_annos_content(path):
    with open(path, 'r+') as fp:
        content = json.load(fp)
    return content


def get_mn_image_pair(video_path_head, image_path_head):
    import random
    postive_path_list = []
    negative_path_list = []
    all_video_cut_path = []
    count = 0
    for item_id in os.listdir(image_path_head):
        count += 1
        print('111', count)
        if not os.path.isdir(image_path_head + item_id):
            continue

        image_path = image_path_head + item_id + '/' + '0.jpg'
        for img_name in os.listdir(video_path_head + item_id):
            video_cut_img_path = video_path_head + item_id + '/' + img_name
            all_video_cut_path.append(video_cut_img_path)
            postive_path_list.append((video_cut_img_path, image_path))

    count = 0
    for item_id in os.listdir(image_path_head):
        if not os.path.isdir(image_path_head + item_id):
            continue
        count += 2
        print('111', count)
        image_path = image_path_head + item_id + '/' + '0.jpg'
        real_video_cut_img_path = video_path_head + item_id + '/' + '0.jpg'
        for i in range(2):  # todo
            video_cut_img_path = random.choice(all_video_cut_path)
            if video_cut_img_path != real_video_cut_img_path:
                negative_path_list.append((video_cut_img_path, image_path))
    postive_label_list = [1] * len(postive_path_list)
    negative_label_list = [0] * len(negative_path_list)
    return postive_path_list + negative_path_list, postive_label_list + negative_label_list


def get_mn_image_pair_v2(video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
                         path_head_save):
    import random
    postive_path_list = []
    negative_path_list = []
    count = 0
    image_path_list_groupby_cls = {}

    # 正样本构造
    for video_id in os.listdir(image_path_head):
        if '.DS_Store' in video_id:
            continue
        video_annos_content = get_annos_content(video_annos_path_head + '{}.json'.format(video_id))

        # 构造根据instance id 查找image的信息的字典
        image_info_dict = {}
        for image_name in os.listdir(image_annos_path_head + video_id):
            if '.DS_Store' in image_name:
                continue
            image_json_path = image_annos_path_head + video_id + '/' + image_name
            image_annos_content = get_annos_content(image_json_path)
            image_name_path = (image_path_head + video_id + '/' + image_name).replace('.json', '.jpg')
            image_annos_content['image_name_path'] = image_name_path
            # img_name = image_annos_content['img_name']
            for anno in image_annos_content['annotations']:
                # viewpoint = anno['viewpoint']
                # display = anno['display']
                label = anno['label']
                class_dict_index = constant.class_dict[label]
                if class_dict_index in image_path_list_groupby_cls:
                    image_path_list_groupby_cls[class_dict_index].append(image_name_path)
                else:
                    image_path_list_groupby_cls[class_dict_index] = [image_name_path]
                instance_id = anno['instance_id']
                if instance_id == 0:  # instance_id为0表示不具有匹配关系
                    continue
                if instance_id in image_info_dict:
                    image_info_dict[instance_id].append(image_annos_content)
                else:
                    image_info_dict[instance_id] = [image_annos_content]
        m, n = 0, 0
        for frame_info in video_annos_content['frames']:
            frame_index = frame_info['frame_index']
            video_cut_img_path = video_path_head + video_id + '/' + str(frame_index) + '.jpg'
            for anno in frame_info['annotations']:
                # viewpoint = anno['viewpoint']
                # display = anno['display']
                label = anno['label']
                instance_id = anno['instance_id']
                if instance_id == 0:  # instance_id为0表示不具有匹配关系
                    continue
                if instance_id not in image_info_dict:  # 有的video没有对应的商品
                    n += 1
                    continue
                m += 1
                class_dict_index = constant.class_dict[label]
                random_image_path = random.choice(image_info_dict[instance_id])
                postive_path_list.append(
                    (video_cut_img_path, random_image_path['image_name_path'], video_id, instance_id, label,
                     class_dict_index, 1)
                )
        # if n!=0:
    # 负样本构造（尽量同类别的构造）
    for video_id in os.listdir(image_path_head):
        if '.DS_Store' in video_id:
            continue
        # p = 0
        video_annos_content = get_annos_content(video_annos_path_head + '{}.json'.format(video_id))
        for frame_info in video_annos_content['frames']:
            frame_index = frame_info['frame_index']
            video_cut_img_path = video_path_head + video_id + '/' + str(frame_index) + '.jpg'
            for anno in frame_info['annotations']:  # todo 应该没有重复吧
                label = anno['label']
                instance_id = anno['instance_id']

                class_dict_index = constant.class_dict[label]
                for i in range(3):  # 正负样本1：2
                    random_img_path = random.choice(image_path_list_groupby_cls[class_dict_index])
                    if video_cut_img_path != random_img_path:
                        # p+=1
                        negative_path_list.append(
                            (video_cut_img_path, random_img_path, video_id, instance_id, label, class_dict_index, 0))

    df = pd.DataFrame(postive_path_list + negative_path_list,
                      columns=['video_cut_img_path', 'img_path', 'video_id', 'instance_id',
                               'cls_label', 'cls_label_index', 'label'])
    df['video_id'] = df['video_id'].astype(str)
    df['video_id'] = df['video_id'].apply(lambda x: x.zfill(6))
    df.to_csv(path_head_save + 'pair_data_info_train_v2.csv', index=False)


def get_sample_mn_image_pair(path_head_save, m, n):
    # 每个视频取m对正样本；每帧正样本取n对负样本
    df = pd.read_csv(path_head_save + 'pair_data_info_train_v2.csv')
    df = shuffle(df)
    df['video_id'] = df['video_id'].astype(str)
    df['video_id'] = df['video_id'].apply(lambda x: x.zfill(6))
    mn_image_pair_data_list = []
    for idx, df_group in df.groupby(by=['video_id']):
        m_, n_ = 0, 0
        for idx, row in df_group.iterrows():
            if row['label'] == 1 and m_ < m:
                # print(dict(row))
                mn_image_pair_data_list.append(dict(row))
                # 选取每帧正样本取n对负样本
                negtive_df_group = df_group[
                    (df_group['label'].isin([0, '0'])) & (df_group['video_cut_img_path'] == row['video_cut_img_path'])]
                # print(len(negtive_df_group),negtive_df_group.to_dict(orient='records'))
                if len(negtive_df_group) > 0:
                    mn_image_pair_data_list.extend(negtive_df_group.head(n).to_dict(orient='records'))
                    m_ += 1

    return mn_image_pair_data_list


def get_mn_test_image_pair(test_video_path_head, test_image_path):
    test_video_path_head, test_image_path = test_video_path_head, test_image_path
    test_video_path_list, test_img_path_list = [], []
    for video_path in os.listdir(test_video_path_head):
        video_path_ = test_video_path_head + video_path + '/' + '0.jpg'
        test_video_path_list.append(video_path_)
    for img_path in os.listdir(test_image_path):
        image_path_ = test_image_path + img_path + '/' + '0.jpg'
        test_img_path_list.append(image_path_)
    return test_video_path_list, test_img_path_list


def load_image(image_path):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    # image = skimage.io.imread(image_path)
    image = cv2.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def get_item_id_by_path(path):
    return path.split('/')[-2]


def read_npy_file(path):
    import numpy as np
    data = np.load(path)
    return data.astype(np.float32)


def find_last():
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        The path of the last checkpoint file
    """
    # Get directory names. Each directory corresponds to a model
    model_dir = ('./logs')
    dir_names = next(os.walk(model_dir))[1]
    key = "deepfashion2"
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        import errno
        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model directory under {}".format(model_dir))
    # Pick last directory
    dir_name = os.path.join(model_dir, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        import errno
        raise FileNotFoundError(
            errno.ENOENT, "Could not find weight files in {}".format(dir_name))
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return checkpoint
