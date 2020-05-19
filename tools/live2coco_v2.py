# -*- coding: utf-8 -*-

import json
import os

import numpy as np
from PIL import Image

# from constant import video_path_head, video_annos_path_head, annos_save_path, path_head
from constant import class_dict


def live2coco_main(video_path_head, video_annos_path_head, annos_save_path, path_head_save):
    dataset = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    for k, v in class_dict.items():
        dataset['categories'].append({
            'id': v,
            'name': k,
            'supercategory': "clothes",
            'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                          '18',
                          '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
                          '34',
                          '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
                          '50',
                          '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65',
                          '66',
                          '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81',
                          '82',
                          '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97',
                          '98',
                          '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111',
                          '112',
                          '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125',
                          '126',
                          '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139',
                          '140',
                          '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153',
                          '154',
                          '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167',
                          '168',
                          '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181',
                          '182',
                          '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195',
                          '196',
                          '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209',
                          '210',
                          '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223',
                          '224',
                          '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237',
                          '238',
                          '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251',
                          '252',
                          '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265',
                          '266',
                          '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279',
                          '280',
                          '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293',
                          '294'],
            'skeleton': []
        })

    num_images = 4525

    img_path_list = []
    img_anns_list = []

    # image_path_head = '../../Live_demo_20200117/image/'
    # image_annos_path_head = '../../Live_demo_20200117/image_annotation/'
    image_path_head = video_path_head
    image_annos_path_head = video_annos_path_head

    for item_id in os.listdir(image_path_head):
        if not os.path.isdir(image_path_head + item_id):
            continue
        img_path_list.append(image_path_head + item_id + '/' + '0.jpg')
        img_anns_list.append(image_annos_path_head + item_id + '/' + '0.json')

    sub_index = 0  # the index of ground truth instance
    import cv2

    def show_image(img):
        cv2.imshow('test', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    num = -1
    m = 0
    for p in os.listdir(annos_save_path):
        num += 1
        json_name = annos_save_path + p
        item_id = p.strip('.json')
        # image_name = image_path_head + '{}/0.jpg'.format(item_id) if len(
        #     item_id) == 6 else video_path_head + '{}/0.jpg'.format(item_id[-6:])
        image_name = video_path_head + '{}/0.jpg'.format(item_id) if len(
            item_id) == 6 else image_path_head + '{}/0.jpg'.format(item_id[-6:])
        if not os.path.exists(image_name):
            m += 1
            # print(m, '{} does not exists! '.format(image_name))
            continue
        if True:
            imag = Image.open(image_name)
            width, height = imag.size
            with open(json_name, 'r') as f:
                print(json_name)

                temp = json.loads(f.read())
                pair_id = temp['pair_id']

                dataset['images'].append({
                    'coco_url': '',
                    'date_captured': '',
                    'file_name': item_id + '/0.jpg',
                    'flickr_url': '',
                    'id': num,
                    'license': 0,
                    'width': width,
                    'height': height
                })
                for i in temp:
                    if i == 'source' or i == 'pair_id':
                        continue
                    else:
                        sub_index = sub_index + 1
                        points = np.zeros(294 * 3)
                        box = temp[i]['bounding_box']
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        x_1 = box[0]
                        y_1 = box[1]
                        # bbox = [x_1, y_1, w, h]
                        bbox = [y_1, x_1, box[3], box[2]]
                        cat = temp[i]['category_id']
                        style = temp[i]['style']

                        dataset['annotations'].append({
                            'area': w * h,
                            'bbox': bbox,
                            'category_id': cat,
                            'id': sub_index,
                            'pair_id': pair_id,
                            'image_id': num,
                            'iscrowd': 0,
                            'style': style,
                            'num_keypoints': [],
                            'keypoints': points.tolist(),
                            'segmentation': [],
                        })
    print(m,'does not exists.')
    json_name = path_head_save + 'train.json'
    with open(json_name, 'w') as f:
        json.dump(dataset, f)
