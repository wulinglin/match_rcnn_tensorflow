#!-*-coding:utf-8-*-
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import cv2


def cut_video(videoPath, svPath, videoname):
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                name = videoname + "_" + "".join(["0" for i in range(4 - len(str(numFrame)))]) + str(numFrame)
                newPath = svPath + "/" + name + ".jpg"
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
                numFrame += 1
        else:
            break
    return "Finish"


def cut_video_with_multiprocessing(folder_video, thead_pool_size):
    # folder_video = sys.argv[1]  # 视频文件夹
    # thead_pool_size = int(sys.argv[2])

    folder_video = folder_video.rstrip() + "/"
    if os.path.exists(folder_video) and os.path.isdir(folder_video):
        thread_list = []
        thread_executor = ThreadPoolExecutor(thead_pool_size + 5)
        videos = [f.strip() for f in os.listdir(folder_video) if f.split(".")[-1] == "mp4"]
        for v in videos:
            videoPath = folder_video + v
            folder_image = "video_cut/" + v
            if not os.path.exists(folder_image):
                os.makedirs(folder_image)
            t = thread_executor.submit(cut_video, videoPath, folder_image, v)
            thread_list.append(t)
            if len(thread_list) > thead_pool_size:
                wait(thread_list, return_when=ALL_COMPLETED)
                thread_list = []
        wait(thread_list, return_when=ALL_COMPLETED)
        thread_list = []
    else:
        print("folder_video not exists or folder_image not exists")
