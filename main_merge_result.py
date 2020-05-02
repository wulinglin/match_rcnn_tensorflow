import json

test_video_frame_annos_path = 'video_boxes.json'
test_image_frame_annos_path = 'image_boxes.json'
test_match_result_path = 'match_result.json'


def read_json_text_by_path(path):
    with open(path, 'r') as f:
        json_text = json.load(f)
        return json_text


video_boxes = read_json_text_by_path(test_video_frame_annos_path)
image_boxes = read_json_text_by_path(test_image_frame_annos_path)
match_result = read_json_text_by_path(test_match_result_path)
result = {}
for video_id, item_id in match_result.items():
    result[video_id] = {'item_id': item_id,
                        'frame_index': 0,
                        # 'frame_index':video_boxes['{}_{}.jpg'.format(video_id,'0.jpg')], #["result"][0][]
                        'result': [
                            {'img_name': '0',
                             'item_box': image_boxes['{}_{}.jpg'.format(item_id, '0')]["result"][0]["bbox"],
                             'frame_box': video_boxes['{}_{}.jpg'.format(video_id, '0')]["result"][0]["bbox"],
                             }
                        ]}

with open('result.json', 'w+') as f:
    json.dump(result, f)
