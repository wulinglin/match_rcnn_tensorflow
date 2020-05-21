"""
欲利用SSD的网络结构输出，最后加一个全连接的网络层，把输出从n*n转换成一维。
所以现在构造一个分类数据集，以微调最后一个全连接网络层。
"""
import json
import os
import re

import pandas as pd

import constant
from tools.data_utils import get_annos_content


def dataset_prepare(mode, video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
                    path_head_save):
    """
    根据是少类还是多类确定构造分类数据集
    """
    df_list = []
    # 正样本构造
    for video_id in os.listdir(video_path_head):
        if '.DS_Store' in video_id:
            continue
        video_annos_content = get_annos_content(video_annos_path_head + '{}.json'.format(video_id))

        # # 构造根据instance id 查找image的信息的字典
        for image_name in os.listdir(image_annos_path_head + video_id):
            if '.DS_Store' in image_name:
                continue

            image_json_path = image_annos_path_head + video_id + '/' + image_name
            image_annos_content = get_annos_content(image_json_path)
            image_name_path = (image_path_head + video_id + '/' + image_name).replace('.json', '.jpg')
            if not os.path.exists(image_name_path):
                continue
            image_annos_content['image_name_path'] = image_name_path
            # img_name = image_annos_content['img_name']
            if len(image_annos_content['annotations']) > 0:
                anno = image_annos_content['annotations'][0]
                label = anno['label']
                box = anno['box']
                class_dict_index = constant.class_dict[label]
                # 类别不均衡，所以类别少的全部算进来，其他的只要一张图片
                if label not in constant.class_dict_rare:
                    df_list.append({'path': image_name_path, 'class': class_dict_index, 'box': box})
                    break
                else:
                    df_list.append({'path': image_name_path, 'class': class_dict_index, 'box': box})

        for frame_info in video_annos_content['frames']:
            frame_index = frame_info['frame_index']
            video_cut_img_path = video_path_head + video_id + '/' + str(frame_index) + '.jpg'
            if not os.path.exists(video_cut_img_path):
                continue
            if len(frame_info['annotations']) > 0:
                anno = frame_info['annotations'][0]
                label = anno['label']
                box = anno['box']
                class_dict_index = constant.class_dict[label]
                # 类别不均衡，所以类别少的全部算进来，其他的只要一张图片
                if label not in constant.class_dict_rare:
                    df_list.append({'path': video_cut_img_path, 'class': class_dict_index, 'box': box})
                    break
                else:
                    df_list.append({'path': video_cut_img_path, 'class': class_dict_index, 'box': box})

    pd.DataFrame(df_list).to_csv(path_head_save + 'live2vgg_{}.csv'.format(mode), index=False)
    return df_list


def dataset_prepare_v2(mode, video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
                       path_head_save):
    """
    构造匹配数据集（每个视频或者图片只选一张）
    """
    df_list = []
    # 正样本构造
    for video_id in os.listdir(video_path_head):
        if '.DS_Store' in video_id:
            continue
        video_annos_content = get_annos_content(video_annos_path_head + '{}.json'.format(video_id))

        # # 构造根据instance id 查找image的信息的字典
        for image_name in os.listdir(image_annos_path_head + video_id):
            if '.DS_Store' in image_name:
                continue

            image_json_path = image_annos_path_head + video_id + '/' + image_name
            image_annos_content = get_annos_content(image_json_path)
            image_name_path = (image_path_head + video_id + '/' + image_name).replace('.json', '.jpg')
            if not os.path.exists(image_name_path):
                continue
            image_annos_content['image_name_path'] = image_name_path
            # img_name = image_annos_content['img_name']
            if len(image_annos_content['annotations']) > 0:
                anno = image_annos_content['annotations'][0]
                label = anno['label']
                box = anno['box']
                class_dict_index = constant.class_dict[label]
                # 类别不均衡，所以类别少的全部算进来，其他的只要一张图片
                df_list.append({'path': image_name_path, 'class': class_dict_index, 'box': box})
                break  # 因为只构造一个

        for frame_info in video_annos_content['frames']:
            frame_index = frame_info['frame_index']
            video_cut_img_path = video_path_head + video_id + '/' + str(frame_index) + '.jpg'
            if not os.path.exists(video_cut_img_path):
                continue
            if len(frame_info['annotations']) > 0:
                anno = frame_info['annotations'][0]
                label = anno['label']
                box = anno['box']
                class_dict_index = constant.class_dict[label]
                # 类别不均衡，所以类别少的全部算进来，其他的只要一张图片
                df_list.append({'path': video_cut_img_path, 'class': class_dict_index, 'box': box})
                break  # 因为只构造一个

    pd.DataFrame(df_list).to_csv(path_head_save + 'live2vgg_{}.csv'.format(mode), index=False)
    return df_list

def dataset_prepare_v4(video_path_head, image_path_head, video_annos_path_head, image_annos_path_head
                       ):
    """
    构造匹配数据集（每个视频或者图片全部都选）
    """
    df_list = []
    # 正样本构造
    for video_id in os.listdir(video_path_head):
        video_img_dict_cur = {'video':[],'image':[]}

        if '.DS_Store' in video_id:
            continue
        video_annos_content = get_annos_content(video_annos_path_head + '{}.json'.format(video_id))

        # # 构造根据instance id 查找image的信息的字典
        for image_name in os.listdir(image_annos_path_head + video_id):
            if '.DS_Store' in image_name:
                continue

            image_json_path = image_annos_path_head + video_id + '/' + image_name
            image_annos_content = get_annos_content(image_json_path)
            image_name_path = (image_path_head + video_id + '/' + image_name).replace('.json', '.jpg')
            if not os.path.exists(image_name_path):
                continue
            image_annos_content['image_name_path'] = image_name_path
            # img_name = image_annos_content['img_name']
            if len(image_annos_content['annotations']) > 0:
                anno = image_annos_content['annotations'][0]
                label = anno['label']
                box = anno['box']
                class_dict_index = constant.class_dict[label]
                # 类别不均衡，所以类别少的全部算进来，其他的只要一张图片
                video_img_dict_cur['video'].append({'path': image_name_path, 'class': class_dict_index, 'box': box})
                # break  # 因为不只构造一个

        for frame_info in video_annos_content['frames']:
            frame_index = frame_info['frame_index']
            video_cut_img_path = video_path_head + video_id + '/' + str(frame_index) + '.jpg'
            if not os.path.exists(video_cut_img_path):
                continue
            if len(frame_info['annotations']) > 0:
                anno = frame_info['annotations'][0]
                label = anno['label']
                box = anno['box']
                class_dict_index = constant.class_dict[label]
                # 类别不均衡，所以类别少的全部算进来，其他的只要一张图片
                video_img_dict_cur['image'].append({'path': video_cut_img_path, 'class': class_dict_index, 'box': box})
                # break  # 因为不只构造一个
        df_list.append(video_img_dict_cur)

    return df_list

def get_result_image_cls():
    # constant.test_image_frame_annos_path
    from collections import Counter
    with open(constant.test_image_frame_annos_path) as fp:
        image_context = json.load(fp)

        image_cls_dict = {}
        for cls_name, cls in constant.class_dict.items():
            image_cls_dict[cls] = []
        image_cls_dict[0] = []

        for key, val in image_context.items():
            image_id = key[:6]
            class_id_list = []
            for each in val["result"]:
                class_id_list.append(each["class_id"])
            count = Counter(class_id_list)
            cls = count.most_common(1)[0][0]
            image_cls_dict[cls].append(image_id)

    with open(constant.test_video_frame_annos_path) as fp:
        video_context = json.load(fp)
        video_cls_dict = {}
        for key, val in video_context.items():
            video_id = key[:6]
            class_id_list = []
            for each in val["result"]:
                class_id_list.append(each["class_id"])
            count = Counter(class_id_list)
            cls = count.most_common(1)[0][0]
            video_cls_dict[video_id] = cls
    return image_cls_dict, video_cls_dict


def get_result_image_box():
    with open(constant.test_image_frame_annos_path) as fp:
        image_context = json.load(fp)
        image_box_dict = {}
        for key, val in image_context.items():
            image_id = key[:6]
            if image_id not in image_box_dict:  # 只取一帧的情况
                frame_index = re.findall('_([\d]+).jpg', key)[0]
                # val["result"] 有很多box和分类结果，是按置信度排序的，所以选取第一个就好了
                cls_and_box_dict = val["result"][0]
                box = cls_and_box_dict['bbox']
                cls = cls_and_box_dict['class_id']
                image_box_dict[image_id] = {'frame_index': frame_index, 'box': box, 'class': cls}

    with open(constant.test_video_frame_annos_path) as fp:
        video_context = json.load(fp)
        video_box_dict = {}
        for key, val in video_context.items():
            video_id = key[:6]
            if video_id not in video_box_dict:  # 只取一帧的情况
                frame_index = re.findall('_([\d]+).jpg', key)[0]
                if frame_index != '0':
                    # val["result"] 有很多box和分类结果，是按置信度排序的，所以选取第一个就好了
                    cls_and_box_dict = val["result"][0]
                    box = cls_and_box_dict['bbox']
                    cls = cls_and_box_dict['class_id']
                    video_box_dict[video_id] = {'frame_index': frame_index, 'box': box, 'class': cls}

    return image_box_dict, video_box_dict


def dataset_prepare_v3_for_test(test_video_path_head, test_image_path):
    image_box_dict, video_box_dict = get_result_image_box()
    df_list = []
    for video_path in os.listdir(test_video_path_head):
        if '.DS_Store' in video_path:
            continue
        frame_index = video_box_dict[video_path]['frame_index']
        video_cut_img_path = test_video_path_head + video_path + '/{}.jpg'.format(frame_index)

        df_list.append({'path': video_cut_img_path, 'class': video_box_dict[video_path]['class'],
                        'box': video_box_dict[video_path]['box']})

    for img_path in os.listdir(test_image_path):
        if '.DS_Store' in img_path:
            continue
        frame_index = image_box_dict[img_path]['frame_index']
        image_name_path = test_image_path + img_path + '/{}.jpg'.format(frame_index)

        df_list.append({'path': image_name_path, 'class': image_box_dict[img_path]['class'],
                        'box': image_box_dict[img_path]['box']})

    return df_list
