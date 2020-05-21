# translate coco_json to xml
# 使用时仅需修改21、22、24行路径文件
import os
import time
import json
import pandas as pd
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO


def trans_id(category_id):
    names = []
    namesid = []
    for i in range(0, len(cats)):
        names.append(cats[i]['name'])
        namesid.append(cats[i]['id'])
        # print('id:{1}\t {0}'.format(names[i], namesid[i]))
    index = namesid.index(category_id)
    return index


# root = '/home/***/datasets/COCO/coco2017/'  # 你下载的 COCO 数据集所在目录
# dataType = 'train2017'
# anno = '{}/annotations/instances_{}.json'.format(root, dataType)
# xml_dir = '{}/xml/{}_xml'.format(root, dataType)
# dir_index = 1

for dir_index in range(1, 2):
    anno = '/data/wl_data/myspace/validation_dataset_part%d/train.json'%dir_index
    xml_dir = '/data/wl_data/myspace/validation_dataset_part%d/test_xml/'%dir_index

    coco = COCO(anno)  # 读文件
    cats = coco.loadCats(coco.getCatIds())  # 这里loadCats就是coco提供的接口，获取类别

    # Create anno dir
    # dttm = time.strftime("%Y%m%d%H%M%S", time.localtime())
    # if os.path.exists(xml_dir):
    #     os.rename(xml_dir, xml_dir + dttm)
    # os.mkdir(xml_dir)

    with open(anno, 'r') as load_f:
        f = json.load(load_f)

    imgs = f['images']

    df_cate = pd.DataFrame(f['categories'])
    df_cate_sort = df_cate.sort_values(["id"], ascending=True)
    categories = list(df_cate_sort['name'])
    print('categories = ', categories)
    df_anno = pd.DataFrame(f['annotations'])

    for i in tqdm(range(len(imgs))):
        xml_content = []
        file_name = imgs[i]['file_name']
        # if file_name == '0000041/0.jpg':
        #     print('xxx')
        if len(file_name) == 13:
            # image = cv2.imread('/data/wl_data/live_data/validation_dataset_part%d/image/'%dir_index + file_name[1:])
            continue
        else:
            image = cv2.imread('/data/wl_data/myspace/validation_dataset_part%d/video_cut/'%dir_index + file_name)
        height = image.shape[0]
        width = image.shape[1]
        img_id = imgs[i]['id']

        xml_content.append("<annotation>")
        xml_content.append("	<folder>VOC2007</folder>")
        xml_content.append("	<filename>" + file_name + "</filename>")
        xml_content.append("	<size>")
        xml_content.append("		<width>" + str(width) + "</width>")
        xml_content.append("		<height>" + str(height) + "</height>")
        xml_content.append("	</size>")
        xml_content.append("	<segmented>0</segmented>")
        # 通过img_id找到annotations
        annos = df_anno[df_anno["image_id"].isin([img_id])]

        for index, row in annos.iterrows():
            bbox = row["bbox"]
            # if int(bbox[0]) > int(bbox[2]) or int(bbox[1]) > int(bbox[3]):
            #     print(index)
            # if int(bbox[2]) > width or int(bbox[3]) > height:
            #     print(index)
            category_id = row["category_id"]
            cate_name = categories[trans_id(category_id)]

            # add new object
            xml_content.append("	<object>")
            xml_content.append("		<name>" + cate_name + "</name>")
            xml_content.append("		<pose>Unspecified</pose>")
            xml_content.append("		<truncated>0</truncated>")
            xml_content.append("		<difficult>0</difficult>")
            xml_content.append("		<bndbox>")
            xml_content.append("			<xmin>" + str(int(bbox[1])) + "</xmin>")
            xml_content.append("			<ymin>" + str(int(bbox[0])) + "</ymin>")
            xml_content.append("			<xmax>" + str(int(bbox[3])) + "</xmax>")
            xml_content.append("			<ymax>" + str(int(bbox[2])) + "</ymax>")
            xml_content.append("		</bndbox>")
            xml_content.append("	</object>")
        xml_content.append("</annotation>")

        x = xml_content
        xml_content = [x[i] for i in range(0, len(x)) if x[i] != "\n"]

        index_id = file_name.split('/', 2)[0]
        os.mkdir(os.path.join(xml_dir, index_id))
        ### list存入文件
        xml_path = os.path.join(xml_dir, file_name.replace('.jpg', '.xml'))
        with open(xml_path, 'w+', encoding="utf8") as f:
            f.write('\n'.join(xml_content))
        xml_content[:] = []


# for dir_index in range(1, 7):
#     anno = '/data/wl_data/myspace/train_dataset_part%d/train.json'%dir_index
#     xml_dir = 'train_xml_%d/'%dir_index
#
#     coco = COCO(anno)  # 读文件
#     cats = coco.loadCats(coco.getCatIds())  # 这里loadCats就是coco提供的接口，获取类别
#
#     # Create anno dir
#     # dttm = time.strftime("%Y%m%d%H%M%S", time.localtime())
#     # if os.path.exists(xml_dir):
#     #     os.rename(xml_dir, xml_dir + dttm)
#     # os.mkdir(xml_dir)
#
#     with open(anno, 'r') as load_f:
#         f = json.load(load_f)
#
#     imgs = f['images']
#
#     df_cate = pd.DataFrame(f['categories'])
#     df_cate_sort = df_cate.sort_values(["id"], ascending=True)
#     categories = list(df_cate_sort['name'])
#     print('categories = ', categories)
#     df_anno = pd.DataFrame(f['annotations'])
#
#     for i in tqdm(range(len(imgs))):
#         xml_content = []
#         file_name = imgs[i]['file_name']
#         # if file_name == '0000041/0.jpg':
#         #     print('xxx')
#         if len(file_name) == 13:
#             image = cv2.imread('/data/wl_data/live_data/train_dataset_part%d/image/'%dir_index + file_name[1:])
#         else:
#             image = cv2.imread('/data/wl_data/myspace/train_dataset_part%d/video_cut/'%dir_index + file_name)
#         height = image.shape[0]
#         width = image.shape[1]
#         img_id = imgs[i]['id']
#
#         xml_content.append("<annotation>")
#         xml_content.append("	<folder>VOC2007</folder>")
#         xml_content.append("	<filename>" + file_name + "</filename>")
#         xml_content.append("	<size>")
#         xml_content.append("		<width>" + str(width) + "</width>")
#         xml_content.append("		<height>" + str(height) + "</height>")
#         xml_content.append("	</size>")
#         xml_content.append("	<segmented>0</segmented>")
#         # 通过img_id找到annotations
#         annos = df_anno[df_anno["image_id"].isin([img_id])]
#
#         for index, row in annos.iterrows():
#             bbox = row["bbox"]
#             # if int(bbox[0]) > int(bbox[2]) or int(bbox[1]) > int(bbox[3]):
#             #     print(index)
#             # if int(bbox[2]) > width or int(bbox[3]) > height:
#             #     print(index)
#             category_id = row["category_id"]
#             cate_name = categories[trans_id(category_id)]
#
#             # add new object
#             xml_content.append("	<object>")
#             xml_content.append("		<name>" + cate_name + "</name>")
#             xml_content.append("		<pose>Unspecified</pose>")
#             xml_content.append("		<truncated>0</truncated>")
#             xml_content.append("		<difficult>0</difficult>")
#             xml_content.append("		<bndbox>")
#             xml_content.append("			<xmin>" + str(int(bbox[1])) + "</xmin>")
#             xml_content.append("			<ymin>" + str(int(bbox[0])) + "</ymin>")
#             xml_content.append("			<xmax>" + str(int(bbox[3])) + "</xmax>")
#             xml_content.append("			<ymax>" + str(int(bbox[2])) + "</ymax>")
#             xml_content.append("		</bndbox>")
#             xml_content.append("	</object>")
#         xml_content.append("</annotation>")
#
#         x = xml_content
#         xml_content = [x[i] for i in range(0, len(x)) if x[i] != "\n"]
#
#         index_id = file_name.split('/', 2)[0]
#         os.mkdir(os.path.join(xml_dir, index_id))
#         ### list存入文件
#         xml_path = os.path.join(xml_dir, file_name.replace('.jpg', '.xml'))
#         with open(xml_path, 'w+', encoding="utf8") as f:
#             f.write('\n'.join(xml_content))
#         xml_content[:] = []
