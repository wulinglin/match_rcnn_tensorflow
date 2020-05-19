"""
欲利用SSD的网络结构输出，最后加一个全连接的网络层，把输出从n*n转换成一维。
所以现在构造一个分类数据集，以微调最后一个全连接网络层。
"""
import os

import pandas as pd

import constant
from tools.data_utils import get_annos_content


def dataset_prepare(mode, video_path_head, image_path_head, video_annos_path_head, image_annos_path_head,
                    path_head_save):
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
