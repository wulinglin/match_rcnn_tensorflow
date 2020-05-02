import os

import skimage

import constant
from constant import video_path_head, image_path_head


def get_mn_image_pair():
    import random
    postive_path_list = []
    negative_path_list = []
    all_video_cut_path = []
    for item_id in os.listdir(image_path_head):
        if not os.path.isdir(image_path_head + item_id):
            continue
        image_path = image_path_head + item_id + '/' + '0.jpg'
        video_cut_img_path = video_path_head + item_id + '/' + '0.jpg'
        all_video_cut_path.append(video_cut_img_path)
        postive_path_list.append((video_cut_img_path, image_path))

    for item_id in os.listdir(image_path_head):
        if not os.path.isdir(image_path_head + item_id):
            continue
        image_path = image_path_head + item_id + '/' + '0.jpg'
        real_video_cut_img_path = video_path_head + item_id + '/' + '0.jpg'
        for i in range(1): # todo
            video_cut_img_path = random.choice(all_video_cut_path)
            if video_cut_img_path != real_video_cut_img_path:
                negative_path_list.append((video_cut_img_path, image_path))
    postive_label_list = [1] * len(postive_path_list)
    negative_label_list = [0] * len(negative_path_list)
    return postive_path_list + negative_path_list, postive_label_list + negative_label_list


def get_mn_test_image_pair():
    test_video_path_head, test_image_path=constant.test_video_path_head, constant.test_image_path
    test_video_path_list, test_img_path_list = [],[]
    for video_path in os.listdir(test_video_path_head):
        video_path_ = test_video_path_head + video_path + '/' + '0.jpg'
        test_video_path_list.append(video_path_)
    for img_path in os.listdir(image_path_head):
        image_path_ = test_image_path + img_path + '/' + '0.jpg'
        test_img_path_list.append(image_path_)
    return test_video_path_list,test_img_path_list


def load_image(image_path):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def get_item_id_by_path(path):
    return path.split('/')[-2]


def read_npy_file(path):
    import numpy as np
    data = np.load(path)
    return data.astype(np.float32)


def find_last():
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        The path of the last checkpoint file
    """
    # Get directory names. Each directory corresponds to a model
    model_dir=('./logs')
    dir_names = next(os.walk(model_dir))[1]
    key = "deepfashion2"
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        import errno
        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model directory under {}".format(model_dir))
    # Pick last directory
    dir_name = os.path.join(model_dir, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        import errno
        raise FileNotFoundError(
            errno.ENOENT, "Could not find weight files in {}".format(dir_name))
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return checkpoint