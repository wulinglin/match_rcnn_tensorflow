# translate coco_json to xml
# 使用时仅需修改21、22、24行路径文件
import os
import time
import json
import pandas as pd
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

anno = '/data/wl_data/live_data/train_dataset_part1/train.json'
xml_dir = 'train_xml/'

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
    height = imgs[i]['height']
    img_id = imgs[i]['id']
    width = imgs[i]['width']

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
        category_id = row["category_id"]
        cate_name = categories[trans_id(category_id)]

        # add new object
        xml_content.append("	<object>")
        xml_content.append("		<name>" + cate_name + "</name>")
        xml_content.append("		<pose>Unspecified</pose>")
        xml_content.append("		<truncated>0</truncated>")
        xml_content.append("		<difficult>0</difficult>")
        xml_content.append("		<bndbox>")
        xml_content.append("			<xmin>" + str(int(bbox[0])) + "</xmin>")
        xml_content.append("			<ymin>" + str(int(bbox[1])) + "</ymin>")
        xml_content.append("			<xmax>" + str(int(bbox[0] + bbox[2])) + "</xmax>")
        xml_content.append("			<ymax>" + str(int(bbox[1] + bbox[3])) + "</ymax>")
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
