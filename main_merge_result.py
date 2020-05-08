import json

from constant import test_video_frame_annos_path, test_image_frame_annos_path, test_match_result_path


def read_json_text_by_path(path):
    with open(path, 'r') as f:
        json_text = json.load(f)
        return json_text


video_boxes = read_json_text_by_path(test_video_frame_annos_path)
image_boxes = read_json_text_by_path(test_image_frame_annos_path)
match_result = read_json_text_by_path(test_match_result_path)
result = {}
for video_id, item_id in match_result.items():
    if not '{}_{}.jpg'.format(item_id, '0') in image_boxes.keys():
        continue
    if not '{}_{}.jpg'.format(item_id, '0') in video_boxes.keys():
        continue
    try:
        item_box = image_boxes['{}_{}.jpg'.format(item_id, '0')]["result"][0]["bbox"]
    except:
        item_box = []
    try:
        frame_box = video_boxes['{}_{}.jpg'.format(video_id, '0')]["result"][0]["bbox"]
    except:
        frame_box = []
    result[video_id] = {'item_id': item_id,
                        'frame_index': 0,
                        # 'frame_index':video_boxes['{}_{}.jpg'.format(video_id,'0.jpg')], #["result"][0][]
                        'result': [
                            {'img_name': '0',
                             'item_box': item_box,
                             'frame_box': frame_box,
                             }
                        ]
                        }

for key, val in video_boxes.items():
    video_id = key[:6]
    if video_id not in match_result.keys():
        item_id = '999999'
    else:
        item_id = match_result[video_id]
    try:
        item_box = image_boxes['{}_{}.jpg'.format(item_id, '0')]["result"][0]["bbox"]
    except:
        item_box = []
    try:
        frame_box = video_boxes['{}_{}.jpg'.format(video_id, '0')]["result"][0]["bbox"]
    except:
        frame_box = []

    result[video_id] = {'item_id': item_id,
                        'frame_index': 0,
                        # 'frame_index':video_boxes['{}_{}.jpg'.format(video_id,'0.jpg')], #["result"][0][]
                        'result': [
                            {'img_name': '0',
                             'item_box': item_box,
                             'frame_box': frame_box,
                             }
                        ]
                        }

with open('result.json', 'w+') as f:
    json.dump(result, f)
print(result)