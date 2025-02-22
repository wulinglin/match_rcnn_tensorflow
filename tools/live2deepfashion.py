import collections
import json
import os

path_head = '../../train_part_1/'

image_path_head = path_head + 'image/'
image_annos_path_head = path_head + 'image_annotation/'

video_path_head = path_head + 'video_cut/'
video_annos_path_head = path_head + 'video_annotation/'

annos_save_path = path_head + 'annos/'
if not os.path.exists(annos_save_path):
    os.makedirs(annos_save_path)

class_dict = collections.OrderedDict({
    '短外套': 1,
    '古风': 2, '古装': 2,
    '短裤': 3,
    '短袖上衣': 4, '短袖Top': 4,
    '长半身裙': 5,
    '背带裤': 6,
    '长袖上衣': 7, '长袖Top': 7,
    '长袖连衣裙': 8,
    '短马甲': 9,
    '短裙': 10,
    '背心上衣': 11,
    '短袖连衣裙': 12,
    '长袖衬衫': 13,
    '中等半身裙': 14,
    '无袖上衣': 15,
    '长外套': 16, '长款外套': 16,
    '无袖连衣裙': 17,
    '连体衣': 18,
    '长马甲': 19,
    '长裤': 20,
    '吊带上衣': 21,
    '中裤': 22,
    '短袖衬衫': 23,
})


def process_cat(cat):
    if '（' in cat:
        return cat[:cat.index('（')]
    return cat

m,n = 0,0
#
# for item_id in os.listdir(image_path_head):
#     m+=1
#     if not os.path.isdir(image_path_head + item_id):
#         continue
#     save_path = ''
#     image_path = image_path_head + item_id + '/' + '0.jpg'
#     annos_path = image_annos_path_head + item_id + '/' + '0.json'
#     with open(annos_path, 'r') as f:
#         temp = json.loads(f.read())
#         if len(temp['annotations']) > 0:  # todo 为0怎么办
#             n+=1
#             cat = process_cat(temp['annotations'][0]['label'])
#             cat_id = class_dict[cat]
#             temp_dict = {"source": "shop",
#                          "pair_id": int(item_id.lstrip('0')),
#                          "item1": {
#                              "segmentation": [],
#                              "scale": 1,  # 此处无用1表示，deepfashion中表示一个数字，其中1代表小比例尺，2代表中等比例尺，3代表大比例尺
#                              "viewpoint": temp['annotations'][0]['viewpoint'],  # 一个数字，其中1表示不磨损，2表示正面视点，3表示侧面或背面视点。
#                              "zoom_in": 1,  # 此处无用1表示，deepfashion中表示一个数字，其中1表示无放大，2表示中等放大，3表示lagre放大.
#                              "landmarks": [],
#                              "style": temp['annotations'][0]['display'],
#                              # 样式编号不同的服装具有不同的样式，例如颜色，印刷和徽标。这里没法对应直播中数据，所以用直播中display代替，表示衣服的展示方式，分为由主播或者模特进行试穿和纯商品展示
#                              "bounding_box": temp['annotations'][0]['box'],
#                              "category_id": cat_id,
#                              "occlusion": 1,  # 此处无用1表示，一个数字，其中1表示轻微遮挡（包括无遮挡），2表示中度遮挡，3表示重度遮挡。
#                              "category_name": cat
#                          }}
#             with open(annos_save_path + '{}.json'.format(item_id), 'w+') as f:
#                 json.dump(temp_dict, f)

for item_id in os.listdir(video_path_head):
    m+=1
    if not os.path.isdir(video_path_head + item_id):
        continue
    save_path = ''
    video_path = video_path_head + item_id + '/' + '0.jpg'
    annos_path = video_annos_path_head + item_id + '.json'
    with open(annos_path, 'r') as f:
        temp = json.loads(f.read())
        temp = temp['frames'][0]
        if len(temp['annotations']) > 0:  # todo 为0怎么办
            n+=1
            cat = process_cat(temp['annotations'][0]['label'])
            cat_id = class_dict[cat]
            temp_dict = {"source": "user",
                         "pair_id": int(item_id.lstrip('0')),
                         "item1": {
                             "segmentation": [],
                             "scale": 1,  # 此处无用1表示，deepfashion中表示一个数字，其中1代表小比例尺，2代表中等比例尺，3代表大比例尺
                             "viewpoint": temp['annotations'][0]['viewpoint'],  # 一个数字，其中1表示不磨损，2表示正面视点，3表示侧面或背面视点。
                             "zoom_in": 1,  # 此处无用1表示，deepfashion中表示一个数字，其中1表示无放大，2表示中等放大，3表示lagre放大.
                             "landmarks": [],
                             "style": temp['annotations'][0]['display'],
                             # 样式编号不同的服装具有不同的样式，例如颜色，印刷和徽标。这里没法对应直播中数据，所以用直播中display代替，表示衣服的展示方式，分为由主播或者模特进行试穿和纯商品展示
                             "bounding_box": temp['annotations'][0]['box'],
                             "category_id": cat_id,
                             "occlusion": 1,  # 此处无用1表示，一个数字，其中1表示轻微遮挡（包括无遮挡），2表示中度遮挡，3表示重度遮挡。
                             "category_name": cat
                         }}
            with open(annos_save_path + '' + '{}.json'.format(item_id), 'w+') as f:
                json.dump(temp_dict, f)


print('总条数',m,'有效条数',n)