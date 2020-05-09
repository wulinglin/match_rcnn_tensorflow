# path_head = '../Live_demo_20200117/'
# path_head_save = '../Live_demo_20200117/'

# path_head = '/data/wl_data/live_data/train_dataset_part1/'
# path_head_save = '/data/wl_data/live_data/train_dataset_part1/'
# path_head = '/home/hzn/match-rcnn/train_dataset_part1/'
# path_head_save = '/home/hzn/match-rcnn/train_dataset_part1/'
path_head = '/tcdata/train_dataset_part1/'
path_head_save = '/tcdata/train_dataset_part1/'

image_path_head = path_head + 'image/'
image_annos_path_head = path_head + 'image_annotation/'

video_path_raw = path_head + 'video/'
video_path_head = path_head_save + 'video_cut/'
video_annos_path_head = path_head + 'video_annotation/'

annos_save_path = path_head_save + 'annos/'

train_img_dir = path_head_save + "video_cut"
train_json_path = path_head_save + "train.json"
valid_img_dir = path_head_save + "video_cut"
valid_json_path = path_head_save + "train.json"

test_path_head = '/tcdata/test_dataset_3w/'
test_path_head_save = '/myspace/test_dataset_3w/'

# test_path_head = '../live_demo_test/'
test_video_path_raw = test_path_head + 'video/'
# test_video_path_head = test_path_head_save + 'video_cut/'
# test_image_path = test_path_head + 'image/'
test_video_path_head = path_head_save + 'video_cut/'
test_image_path = path_head + 'image/'

train_data_all_path = path_head_save + '/train_data_all/'
train_data_all_path_no_pair = path_head_save + '/train_data_all_nopair/'
test_video_frame_annos_path = path_head_save + 'video_boxes.json'
test_image_frame_annos_path = path_head_save + 'image_boxes.json'
test_match_result_path = path_head_save + 'match_result.json'
