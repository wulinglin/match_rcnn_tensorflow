import os
import json
import numpy as np


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


annos_dir = '../../annos'
result_json = '../../test_detect/part_1/video_boxes.json'

with open(result_json, 'r') as f:
    results = json.loads(f.read())

keys_list = results.keys()

# for key in keys_list:
#     if len(results[key]['result']) == 0:
#         print(key)
# exit()
predict_true = 0
predict_false = 0
sum_iou = 0
dict_predict = {i:0 for i in range(24)}
dict_all = {i:0 for i in range(24)}
dict_TP = {i:0 for i in range(24)}
dict_FN = {i:0 for i in range(24)}
dict_FP = {i:0 for i in range(24)}

confusion_matrix = np.zeros((24, 24), dtype=np.int)

json_list = os.listdir(annos_dir)
for json_name in json_list:
    index_ = json_name.split('.', 2)[0] + '_0.jpg'
    with open(os.path.join(annos_dir, json_name), 'r') as f:
        anno = json.loads(f.read())
    class_id = anno['item1']['category_id']
    bbox = anno['item1']['bounding_box']
    try:
        predict_class = results[index_]['result'][0]['class_id']
        predict_bbox = results[index_]['result'][0]['bbox']
        predict_bbox = [predict_bbox[1], predict_bbox[0], predict_bbox[3], predict_bbox[2]]

        iou = compute_iou(bbox, predict_bbox)
        dict_all[class_id] = dict_all[class_id] + 1
        if class_id == predict_class:
            predict_true += 1
            dict_predict[class_id] = dict_predict[class_id] + 1
            dict_TP[class_id] = dict_TP[class_id] + 1

        else:
            dict_FN[class_id] = dict_FN[class_id] + 1
            dict_FP[predict_class] = dict_FP[predict_class] + 1
            predict_false += 1
        if len(results[index_]['result']) != 0:
            sum_iou += iou
    except:
        continue
    confusion_matrix[class_id][predict_class] += 1

print('predict_true: %d'%predict_true)
print('predict_false: %d'%predict_false)
print('mean_iou: %f'%(sum_iou/(predict_false + predict_true)))

dict_probability = {}
for i in range(1, 24):
    dict_probability[i] = dict_predict[i] / dict_all[i]
# print('each_class_probability', dict_probability)

for i in range(1, len(confusion_matrix)):
    row_sum, colum_sum = sum(confusion_matrix[i]), sum(confusion_matrix[r, i] for r in range(len(confusion_matrix)))
    try:
        print('class %d, presion: %s' % (i, confusion_matrix[i][i] / float(colum_sum)))
        print('class %d, recall: %s' % (i, confusion_matrix[i][i] / float(row_sum)))
    except:
        print('error')
print('finished')
