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

        print("正在处理文件夹", parents)
        # path = parents.replace("/", "//")
        path = parents
        f = parents.split("/")[1]
        save_path = save_dir + "/"
        # 对每视频数据进行遍历
        len_f = len(filenames)
        count_f = 0
        for file in filenames:
            count += 1
            count_f += 1
            # if count > 100:  # todo
            #     break
            if count_f%50==0:
                print(count_f, len_f)
            file_name = file.split(".")[0]
            save_path_ = save_path + "/" + file_name
            if not os.path.isdir(save_path_):
                os.makedirs(save_path_)
            video_path = path + "/" + file
            video_split(video_path, save_path_)

    end_time = time.time()
    print("Cost time", end_time - start_time)


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


def thread_video_split(data_dir, save_dir):
    import threadpool
    start_time = time.time()

    def get_dir(data_dir, save_dir):
        video_path_and_save_path_list = []
        count = 0
        for parents, dirs, filenames in os.walk(data_dir):
            print("正在处理文件夹", parents)
            # path = parents.replace("/", "//")
            path = parents
            f = parents.split("/")[1]
            save_path = save_dir + "/"
            # 对每视频数据进行遍历
            len_f = len(filenames)
            count_f = 0
            for file in filenames:
                count += 1
                count_f += 1
                file_name = file.split(".")[0]
                save_path_ = save_path + "/" + file_name
                video_path = path + "/" + file

                video_path_and_save_path_list.append((video_path, save_path_))
        print(video_path_and_save_path_list)
        return video_path_and_save_path_list

    def run(video_path_and_save_path_list):
        """
        主函数
        """
        for video_path, save_path_ in video_path_and_save_path_list:
            if not os.path.isdir(save_path_):
                os.makedirs(save_path_)
            video_split(video_path, save_path_)

    # 参数列表
    args = get_dir(data_dir, save_dir)
    end_time = time.time()
    print("Cost time get_dir", start_time - end_time)
    print('start threading....!')
    # 使用多线程启动
    pool = threadpool.ThreadPool(10)
    requests = threadpool.makeRequests(run, [args])
    [pool.putRequest(req) for req in requests]
    pool.wait()

    end_time = time.time()
    print("Cost time thread cut", start_time - end_time)

# if __name__ == "__main__":
#     from constant import video_path_head, video_path_raw
#
#     data_dir = video_path_raw  # 视频数据主目录
#     save_dir = video_path_head  # 帧文件保存目录
#     video_split_and_save(data_dir, save_dir)
#
#     # json_data_view()
