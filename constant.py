# path_head = '../Live_demo_20200117/'
# path_head_save = '../Live_demo_20200117/'

train_json_path = []
train_img_dir = []
valid_json_path = []
valid_img_dir = []

path_head = '/data/wl_data/live_data/validation_dataset_part3/'
path_head_save = '/data/wl_data/myspace/train_dataset_part1/'
for i in range(1, 7):
    path_head = '/data/wl_data/live_data/train_dataset_part%d/'%i
    path_head_save = '/data/wl_data/myspace/train_dataset_part%d/'%i
#path_head_save = '/data/wl_data/live_data/train_dataset_part1/'
# path_head = '/home/hzn/match-rcnn/train_dataset_part1/'
# path_head_save = '/home/hzn/match-rcnn/train_dataset_part1/'
#path_head = '/tcdata_train/train_dataset_part1/'
#path_head_save = './test_detect/part_1/'

    image_path_head = path_head + 'image/'
    image_annos_path_head = path_head + 'image_annotation/'

    video_path_raw = path_head + 'video/'
    video_path_head = path_head_save + 'video_cut/'
    video_annos_path_head = path_head + 'video_annotation/'

    annos_save_path = path_head_save + 'annos/'

    train_img_dir.append(path_head_save + "video_cut")
    train_json_path.append(path_head_save + "train.json")
    valid_img_dir.append(path_head_save + "video_cut")
    valid_json_path.append(path_head_save + "train.json")

# test_path_head = '/tcdata/test_dataset_part5/'
# test_path_head_save = '/myspace/test_dataset_part5/'

# test_path_head = '/data/wl_data/myspace/validation_dataset_part3/'
test_path_head = '/data/wl_data/live_data/validation_dataset_part3/'
# test_path_head = '/tcdata/test_dataset_part5/'
# test_path_head_save = '/data/wl_data/myspace/validation_dataset_part2/'
test_path_head_save = '/data/wl_data/myspace/validation_dataset_part3/'
# test_path_head_save = '/home/wuling/myspace/validation_dataset_part3/'

# test_path_head = '../live_demo_test/'
test_video_path_raw = test_path_head + 'video/'
# test_video_path_head = test_path_head_save + 'video_cut/'
# test_image_path = test_path_head + 'image/'
test_video_path_head = test_path_head_save + 'video_cut/'
test_image_path = test_path_head + 'image/'

train_data_all_path = path_head_save + '/train_data_all/'
train_data_all_path_no_pair = path_head_save + '/train_data_all_nopair/'
test_video_frame_annos_path = './' + 'video_boxes.json'
test_image_frame_annos_path = './' + 'image_boxes.json'
test_match_result_path = './' + 'match_result.json'

test_cos_image_feature_path = './' + 'image_feature.h5'
test_cos_frame_feature_path = './' + 'video_feature.h5'
