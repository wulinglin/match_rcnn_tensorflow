import json
import os
import time

import cv2  # 导入opencv模块


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
        # 每隔40帧保存一下
        if c % 40 == 0:
            cv2.imwrite(save_path + "/" + str(c) + '.jpg', frame)
            cv2.waitKey(1)
        c = c + 1


def video_split_and_save(data_dir, save_dir):
    # data_dir = "/Users/lingwu/data/Live_demo_20200117/video/"  # 视频数据主目录
    #
    # save_dir = "/Users/lingwu/data/Live_demo_20200117/video_cut/"  # 帧文件保存目录

    start_time = time.time()
    count = 0
    for parents, dirs, filenames in os.walk(data_dir):
        # if parents == DATA_DIR:
        #     continue

        # path = parents.replace("/", "//")
        path = parents
        f = parents.split("/")[1]
        save_path = save_dir + "/"
        # 对每视频数据进行遍历
        for file in filenames:
           # count+=1
            #if count>100:
             #   break
            file_name = file.split(".")[0]
            save_path_ = save_path + "/" + file_name
            if not os.path.exists(save_path_):
                os.makedirs(save_path_)
            video_path = path + "/" + file
            video_split(video_path, save_path_)
    end_time = time.time()
    print("Cost time", start_time - end_time)


def video_split_and_save_multiprocess(data_dir, save_dir):
    import multiprocessing
    import time
    start_time = time.time()
    pool = multiprocessing.Pool(processes=5)  # 创建5个进程

    for parents, dirs, filenames in os.walk(data_dir):
        path = parents
        f = parents.split("/")[1]
        save_path = save_dir + "/"
        # 对每视频数据进行遍历
        for file in filenames:
            file_name = file.split(".")[0]
            save_path_ = save_path + "/" + file_name
            if not os.path.exists(save_path_):
                os.makedirs(save_path_)
            video_path = path + "/" + file
            pool.apply_async(video_split, (video_path, save_path_))
        print('test')
        pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
        pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
        print("Sub-process all done.")

    end_time = time.time()
    print("Cost time", start_time - end_time)


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


# if __name__ == "__main__":
#     from constant import video_path_head, video_path_raw
#
#     data_dir = video_path_raw  # 视频数据主目录
#     save_dir = video_path_head  # 帧文件保存目录
#     video_split_and_save(data_dir, save_dir)
#
#     # json_data_view()
