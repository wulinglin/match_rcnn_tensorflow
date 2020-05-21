# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from lib.configs import cfgs

if cfgs.DATASET_NAME == 'ship':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'ship': 1
    }
elif cfgs.DATASET_NAME == 'FDDB':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'face': 1
    }
elif cfgs.DATASET_NAME == 'icdar':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'text': 1
    }
elif cfgs.DATASET_NAME.startswith('DOTA'):
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'roundabout': 1,
        'tennis-court': 2,
        'swimming-pool': 3,
        'storage-tank': 4,
        'soccer-ball-field': 5,
        'small-vehicle': 6,
        'ship': 7,
        'plane': 8,
        'large-vehicle': 9,
        'helicopter': 10,
        'harbor': 11,
        'ground-track-field': 12,
        'bridge': 13,
        'basketball-court': 14,
        'baseball-diamond': 15
    }
elif cfgs.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
elif cfgs.DATASET_NAME == 'tb':
    NAME_LABEL_MAP = {
        'duanwaitao': 1,
        'gufeng': 2, 'guzhuang': 2,
        'duanku': 3,
        'duanxiushangyi': 4, 'duanxiuTop': 4,
        'changbanshenqun': 5,
        'beidaiku': 6,
        'changxiushangyi': 7, 'changxiuTop': 7,
        'changxiulianyiqun': 8,
        'duanmajia': 9,
        'duanqun': 10,
        'beixinshangyi': 11,
        'duanxiulianyiqun': 12,
        'changxiuchenshan': 13,
        'zhongdengbanshenqun': 14,
        'wuxiushangyi': 15,
        'changwaitao': 16, 'changkuanwaitao': 16,
        'wuxiulianyiqun': 17,
        'liantiyi': 18,
        'changmajia': 19,
        'changku': 20,
        'diaodaishangyi': 21,
        'zhongku': 22,
        'duanxiuchenshan': 23,
    }
    # NAME_LABEL_MAP = {
    #     '短外套': 1,
    #     '古风': 2, '古装': 2,
    #     '短裤': 3,
    #     '短袖上衣': 4, '短袖Top': 4,
    #     '长半身裙': 5,
    #     '背带裤': 6,
    #     '长袖上衣': 7, '长袖Top': 7,
    #     '长袖连衣裙': 8,
    #     '短马甲': 9,
    #     '短裙': 10,
    #     '背心上衣': 11,
    #     '短袖连衣裙': 12,
    #     '长袖衬衫': 13,
    #     '中等半身裙': 14,
    #     '无袖上衣': 15,
    #     '长外套': 16, '长款外套': 16,
    #     '无袖连衣裙': 17,
    #     '连体衣': 18,
    #     '长马甲': 19,
    #     '长裤': 20,
    #     '吊带上衣': 21,
    #     '中裤': 22,
    #     '短袖衬衫': 23,
    # }
else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEL_NAME_MAP = get_label_name_map()