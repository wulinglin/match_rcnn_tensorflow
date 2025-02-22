import json


def json_data_view():
    train_img_dir = "/Users/lingwu/data/deepfashion/train/image"
    train_json_path = "/home/hzn/match-rcnn/Live_demo_20200117/train.json"
    valid_img_dir = "/Users/lingwu/data/deepfashion/validation/image"
    valid_json_path = "/Users/lingwu/PycharmProjects/match_rcnn/tools/valid.json"
    # with open(valid_json_path) as f:
    with open(train_json_path) as f:
        json_content = json.load(fp=f)

    print(json_content.keys())
    # info": {}, "licenses": [], dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    print(json_content['info'])
    print(json_content['licenses'])
    print('--------images----------', len(json_content['images']))
    for i in json_content['images'][:1]:
        print(i.keys())
        print('images', i)
    print('--------annotations----------', len(json_content['annotations']))
    for i in json_content['annotations'][:1]:
        print('annotations', i.keys())
        print(i)
    print('--------categories----------', len(json_content['categories']))
    for i in json_content['categories'][:1]:
        print(i.keys())
        print(i)


import cv2  # 导入opencv模块
import os
import time


def video_split(video_path, save_path):
    '''
    对视频文件切割成帧
    '''
    '''
    @param video_path:视频路径
    @param save_path:保存切分后帧的路径
    '''
    vc = cv2.VideoCapture(video_path)
    # 一帧一帧的分割 需要几帧写几
    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        # 每秒提取5帧图片
        if c % 40 == 0:
            cv2.imwrite(save_path + "/" + str(c) + '.jpg', frame)
            cv2.waitKey(1)
        c = c + 1


def video_split_and_save(data_dir, save_dir):
    # data_dir = "/Users/lingwu/data/Live_demo_20200117/video/"  # 视频数据主目录
    #
    # save_dir = "/Users/lingwu/data/Live_demo_20200117/video_cut/"  # 帧文件保存目录

    start_time = time.time()
    for parents, dirs, filenames in os.walk(data_dir):
        print(parents, dirs, filenames)
        # if parents == DATA_DIR:
        #     continue

        print("正在处理文件夹", parents)
        # path = parents.replace("/", "//")
        path = parents
        f = parents.split("/")[1]
        save_path = save_dir + "/"
        # 对每视频数据进行遍历
        for file in filenames:
            print(file)
            file_name = file.split(".")[0]
            save_path_ = save_path + "/" + file_name
            if not os.path.isdir(save_path_):
                os.makedirs(save_path_)
            video_path = path + "/" + file
            video_split(video_path, save_path_)

    end_time = time.time()
    print("Cost time", start_time - end_time)


if __name__ == "__main__":
    data_dir = "/data/wl_data/live_data/train_dataset_part1/video"  # 视频数据主目录
    save_dir = "../../train_part_1/video_cut"  # 帧文件保存目录
    video_split_and_save(data_dir, save_dir)

    # json_data_view()
