# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:15:50 2019

@author: loktarxiao
"""

import sys

import constant
from lib.get_rcnn_feature_nopair import Get_RCNN_Feature
from lib.model_new import load_image_gt_v2
from tools.data_utils import get_mn_test_image_pair

##from lib.model_mn_v2 import MatchRCNN

sys.dont_write_bytecode = True

import os
import numpy as np

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from lib.config import Config
from lib import utils
import pandas as pd


class DeepFashion2Config(Config):
    """Configuration for training on DeepFashion2.
    Derives from the base Config class and overrides values specific
    to the DeepFashion2 dataset.
    """
    # Give the configuration a recognizable name
    NAME = "deepfashion2"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 23  # COCO has 80 classes

    USE_MINI_MASK = True

    train_img_dir = constant.train_img_dir
    train_json_path = constant.train_json_path
    valid_img_dir = constant.valid_img_dir
    valid_json_path = constant.valid_json_path


############################################################
#  Dataset
############################################################

class DeepFashion2Dataset(utils.Dataset):
    def load_coco(self, image_dir, json_path, class_ids=None,
                  class_map=None, return_coco=False):
        """Load the DeepFashion2 dataset.
        """
        print(json_path)
        coco = COCO(json_path)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("deepfashion2", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "deepfashion2", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        # todo 为什么这里可以，而下面train_dataset不能.dataset看数据，那么要怎么看数据
        # print(coco.dataset.keys())
        # print(len(coco.dataset['images']), coco.dataset['images'][:2])
        # print(len(coco.dataset['annotations']), coco.dataset['annotations'][:1])
        # print(len(coco.dataset['categories']), coco.dataset['categories'][:1])

        if return_coco:
            return coco

    def load_keypoint(self, image_id):
        """
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion2":
            return super(DeepFashion2Dataset, self).load_mask(image_id)

        instance_keypoints = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))
            if class_id:
                keypoint = annotation['keypoints']

                instance_keypoints.append(keypoint)
                class_ids.append(class_id)

        keypoints = np.stack(instance_keypoints, axis=1)
        class_ids = np.array(class_ids, dtype=np.int32)
        return keypoints, class_ids

    def load_bbox(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        # if image_info["source"] != "deepfashion2":
        #     return super(DeepFashion2Dataset, self).load_mask(image_id)

        bbox_list = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))
            if class_id:
                bbox = annotation['bbox']

                bbox_list.append(bbox)
                class_ids.append(class_id)

        # Pack instance masks into an array
        # if class_ids:
        # mask = np.stack(instance_masks, axis=2).astype(np.bool)
        bbox_array = np.array(bbox_list, dtype=np.int32)
        class_ids = np.array(class_ids, dtype=np.int32)
        return class_ids, bbox_array
        # else:
        #     # Call super class to return an empty mask
        #     return super(DeepFashion2Dataset, self).load_mask(image_id)

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion2":
            return super(DeepFashion2Dataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return class_ids
        else:
            # Call super class to return an empty mask
            return super(DeepFashion2Dataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        super(DeepFashion2Dataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def main_match_train(mode, config, model_dir=None):
    from tools.data_utils import get_mn_image_pair
    #
    # dataset_train = DeepFashion2Dataset()
    # dataset_train.load_coco(config.train_img_dir, config.train_json_path)
    # dataset_train.prepare()

    img_path_list, label_list = get_mn_image_pair()

    rcnn_model = Get_RCNN_Feature(mode, config, model_dir)

    raw_img_path = []
    img_feature_path = []
    img_name = []

    raw_video_frame_path = []
    video_feature_path = []
    video_frame_name = []
    label_all = []
    count = 0
    # raw_video_frame_path,video_feature_path,video_frame_name,raw_img_path,img_feature_path,img_name,label
    # ../Live_demo_20200117/video_cut/000002/0.jpg,train_data_feature/video_feature/000002,0.jpg,../Live_demo_20200117/image/000002/0.jpg,train_data_feature/image_feature/000002,0.jpg,1
    # ../Live_demo_20200117/video_cut/000001/0.jpg,train_data_feature/video_feature/000001,0.jpg,../Live_demo_20200117/image/000001/0.jpg,train_data_feature/image_feature/000001,0.jpg,1
    # ../Live_demo_20200117/video_cut/000002/0.jpg,train_data_feature/video_feature/000002,0.jpg,../Live_demo_20200117/image/000001/0.jpg,train_data_feature/image_feature/000001,0.jpg,0
    for p1, p2 in img_path_list:
        raw_video_frame_path.append(p1)
        p1_video_path = p1.split("/")[-2]
        p1_frame_name = p1.split("/")[-1]
        video_frame_name.append(p1_frame_name)
        video_feature_path.append(
            constant.path_head_save + "{}_data_feature/video_feature/".format(mode) + p1_video_path)

        raw_img_path.append(p2)
        p2_image_path = p2.split("/")[-2]
        p2_image_name = p2.split("/")[-1]
        img_name.append(p2_image_name)
        img_feature_path.append(constant.path_head_save + "{}_data_feature/image_feature/".format(mode) + p2_image_path)
        if mode not in ['inference', 'test', 'predict']:
            label_all.append(label_list[count])
        count += 1
        if count % 10 == 0:
            print('1111 match train', count)

    df_info = pd.DataFrame({"raw_video_frame_path": raw_video_frame_path, "video_feature_path": video_feature_path,
                            "video_frame_name": video_frame_name, \
                            "raw_img_path": raw_img_path, "img_feature_path": img_feature_path, "img_name": img_name,
                            "label": label_all})
    df_info.to_csv(constant.path_head_save + "pair_data_info_{}.csv".format(mode), index=False, encoding="utf8")

    count = 0
    for p1, p1_feature_path, p1_name, p2, p2_feature_path, p2_name in zip(raw_video_frame_path, video_feature_path,
                                                                          video_frame_name, \
                                                                          raw_img_path, img_feature_path, img_name):

        count += 1
        if count % 10 == 0:
            print('2222 match train', count)

        image_1 = load_image_gt_v2(p1, config)
        rcnn_model.save_dataset(image_1, p1_feature_path, p1_name)

        image_2 = load_image_gt_v2(p2, config)
        rcnn_model.save_dataset(image_2, p2_feature_path, p2_name)


def main_match_test(mode, config, model_dir=None):
    # dataset_train = DeepFashion2Dataset()
    # dataset_train.load_coco(config.train_img_dir, config.train_json_path)
    # dataset_train.prepare()

    test_video_path_list, test_img_path_list = get_mn_test_image_pair()

    # todo
    test_video_path_list, test_img_path_list = test_video_path_list, test_img_path_list
    images = []

    rcnn_model = Get_RCNN_Feature(mode, config, model_dir)

    # raw_img_path = []
    # img_feature_path = []
    # img_name = []
    #
    # raw_video_frame_path = []
    # video_feature_path = []
    # video_frame_name = []
    # label_all = []
    # raw_video_frame_path,video_feature_path,video_frame_name,raw_img_path,img_feature_path,img_name,label
    # ../Live_demo_20200117/video_cut/000002/0.jpg,train_data_feature/video_feature/000002,0.jpg,../Live_demo_20200117/image/000002/0.jpg,train_data_feature/image_feature/000002,0.jpg,1
    # ../Live_demo_20200117/video_cut/000001/0.jpg,train_data_feature/video_feature/000001,0.jpg,../Live_demo_20200117/image/000001/0.jpg,train_data_feature/image_feature/000001,0.jpg,1
    # ../Live_demo_20200117/video_cut/000002/0.jpg,train_data_feature/video_feature/000002,0.jpg,../Live_demo_20200117/image/000001/0.jpg,train_data_feature/image_feature/000001,0.jpg,0
    count = 0
    print(len(test_video_path_list))
    drop_list = []
    for p1 in test_video_path_list:
        if '.DS_Store' in p1:
            continue
        # raw_video_frame_path.append(p1)
        p1_video_path = p1.split("/")[-2]
        p1_frame_name = p1.split("/")[-1]
        # video_frame_name.append(p1_frame_name)
        # video_feature_path.append(
        #     constant.path_head_save + "{}_data_feature/video_feature/".format(mode) + p1_video_path)
        # try:
        image_1 = load_image_gt_v2(p1, config)
        count += 1
        # except:
            # drop_list.append(p1)
            # continue
        if count % 10 == 0:
            print('3333 match test', count)
        p1_feature_path = constant.path_head_save + "{}_data_feature/video_feature/".format(mode) + p1_video_path
        rcnn_model.save_dataset(image_1, p1_feature_path, p1_frame_name)

    print('-'*30,len(test_img_path_list))
    count = 0
    for p2 in test_img_path_list:
        if '.DS_Store' in p2:
            continue
        # raw_img_path.append(p2)
        p2_image_path = p2.split("/")[-2]
        p2_image_name = p2.split("/")[-1]
        # img_name.append(p2_image_name)
        # img_feature_path.append(constant.path_head_save + "{}_data_feature/image_feature/".format(mode) + p2_image_path)
        p2_feature_path = constant.path_head_save + "{}_data_feature/image_feature/".format(mode) + p2_image_path
        # try:
        image_2 = load_image_gt_v2(p2, config)
        count += 1
        if count % 10 == 0:
            print('3333 match test', count)
        rcnn_model.save_dataset(image_2, p2_feature_path, p2_image_name)
        # except:
            # drop_list.append(p2)
            # continue

    raw_img_path = []
    img_feature_path = []
    img_name = []

    raw_video_frame_path = []
    video_feature_path = []
    video_frame_name = []
    count=0
    for p1 in test_video_path_list:

        if '.DS_Store' in p1 or p1 in drop_list:
            continue
        for p2 in test_img_path_list:
            if '.DS_Store' in p2 or p2 in drop_list:
                continue
            raw_video_frame_path.append(p1)
            p1_video_path = p1.split("/")[-2]
            p1_frame_name = p1.split("/")[-1]
            video_frame_name.append(p1_frame_name)
            video_feature_path.append(constant.path_head_save +"{}_data_feature/video_feature/".format(mode) + p1_video_path)

            raw_img_path.append(p2)
            p2_image_path = p2.split("/")[-2]
            p2_image_name = p2.split("/")[-1]
            img_name.append(p2_image_name)
            img_feature_path.append(constant.path_head_save +"{}_data_feature/image_feature/".format(mode) + p2_image_path)

            count += 1
            if count % 10 == 0:
                print('3333 match test', count)
    print(len(raw_video_frame_path),len(video_feature_path),len(video_frame_name),
          len(raw_img_path),len(img_feature_path),len(img_name))
    df_info = pd.DataFrame({"raw_video_frame_path": raw_video_frame_path, "video_feature_path": video_feature_path,
                            "video_frame_name": video_frame_name, \
                            "raw_img_path": raw_img_path, "img_feature_path": img_feature_path, "img_name": img_name,
                            })

    df_info.to_csv(constant.path_head_save + "pair_data_info_{}.csv".format(mode), index=False, encoding="utf8")


def train(model, config):
    """
    """
    dataset_train = DeepFashion2Dataset()
    dataset_train.load_coco(config.train_img_dir, config.train_json_path)
    dataset_train.prepare()

    dataset_valid = DeepFashion2Dataset()
    dataset_valid.load_coco(config.valid_img_dir, config.valid_json_path)
    dataset_valid.prepare()

    model.train(dataset_train, dataset_valid,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath("./")
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_deepfashion2_0003.h5")

    # model_dir = './mask_rcnn_deepfashion2_0001.h5'

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Match R-CNN for DeepFashion.')
    parser.add_argument("--command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    """
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    """

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DeepFashion2Config()
    else:
        class InferenceConfig(DeepFashion2Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    # config.display()
    from tools.data_utils import find_last

    model_dir = find_last()
    if args.command == 'train':
        main_match_train(mode=args.command, config=config, model_dir=model_dir)
    elif args.command == 'test':
        main_match_test(mode=args.command, config=config, model_dir=model_dir)
    else:
        raise Exception('args command error!')
