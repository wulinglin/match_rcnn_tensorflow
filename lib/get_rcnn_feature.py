import os

import numpy as np
from keras.models import Model

import constant
from lib.model_new import MaskRCNN


class Get_RCNN_Feature:
    def __init__(self, mode, config, model_dir=None):
        self.config = config
        ###self.config = Config()
        self.model_mask = MaskRCNN("inference", config, model_dir)
        ##self.mine_model = self.model_mask.build("inference", config)
        self.model_mask.load_weights(model_dir, by_name=True)
        self.model_mask_keras = self.model_mask.keras_model
        self.output = Model(inputs=self.model_mask_keras.inputs, output= \
            [self.model_mask_keras.get_layer('fpn_p5').output, \
             self.model_mask_keras.get_layer('roi_align_classifier').output, \
             self.model_mask_keras.get_layer('mrcnn_class').output])

    def save_dataset(self, images, labels, data_path):
        ###print("img:", len(images) )
        ###print("img_label:", len(labels) )

        images_concat = []
        img_data = []
        img_fpn_data = []
        count = 0
        for img1, img2 in images:
            images_concat.extend([img1, img2])

        for img1 in images_concat:
            images_tmp = [img1]

            # Mold inputs to format expected by the neural network
            molded_images, image_metas, windows = self.model_mask.mold_inputs(images_tmp)

            # Validate image sizes
            # All images in a batch MUST be of the same size
            image_shape = molded_images[0].shape
            for g in molded_images[1:]:
                assert g.shape == image_shape, \
                    "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

            # Anchors
            anchors = self.model_mask.get_anchors(image_shape)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
            fpn_p5, output_rois, output_rois_score = self.output.predict([molded_images, image_metas, anchors],
                                                                         verbose=0)

            ###print("len ", len(output_rois) ) 
            print("fpn_p5.shape:", fpn_p5.shape)
            print("output_rois.shape:", output_rois.shape)
            print("output_rois_score.shape:", output_rois_score.shape)
            label_count = output_rois_score.shape[-1]
            label_dict = {}
            for ii in range(1, label_count):  # 因为0是背景类
                tmp = output_rois_score[0][:, ii]
                ##print("tmp_shape:", tmp.shape)
                tmp_idx = np.argsort(-tmp)[0:2]  ###np.argmax(tmp)[:-1]
                tmp_score = tmp[np.argsort(-tmp)[0:2]]
                label_dict[ii] = [(tmp_score[0], tmp_idx[0]), (tmp_score[1], tmp_idx[1])]
                # print( "sort:", tmp[np.argsort(-tmp)[0:2]] )
            tmp_rois_score = sorted(label_dict.items(), key=lambda x: x[1][0][0], reverse=True)
            ##print("aaaaa:", tmp_rois_score)
            tmp_rois = np.zeros((4, 7, 7, 256))
            # if tmp_rois_score[0][1][0][0] > 0.5:
            tmp_rois[0] = output_rois[0][tmp_rois_score[0][1][0][1]]
            if tmp_rois_score[0][1][1][0] > 0.5:
                tmp_rois[1] = output_rois[0][tmp_rois_score[0][1][1][1]]
            ##
            if tmp_rois_score[1][1][0][0] > 0.5:
                tmp_rois[2] = output_rois[0][tmp_rois_score[1][1][0][1]]
            if tmp_rois_score[1][1][1][0] > 0.5:
                tmp_rois[3] = output_rois[0][tmp_rois_score[1][1][1][1]]

            d1 = np.hstack([tmp_rois[0], tmp_rois[1]])
            d2 = np.hstack([tmp_rois[2], tmp_rois[3]])
            image_data_select = np.vstack([d1, d2])
            print("image_select_shape:", image_data_select.shape)
            img_data.append(image_data_select)

            # np.armax(output_rois[1] , axis = 2)
            img_fpn_data.append(fpn_p5[0])

            ##print( "--------------", np.isnan(image_data_select).sum() ) 
            ##print( "--------------", np.isnan( fpn_p5[0]).sum() ) 
            count += 1
            if count % 2 == 0:
                img_data_npy = np.stack([img_data[0], img_data[1]], axis=0)
                img_fpn_data_npy = np.stack([img_fpn_data[0], img_fpn_data[1]], axis=0)
                abs_data_path = constant.train_data_all_path + "label_" + str(labels[0]) + "/" + data_path + "/"

                if not os.path.exists(abs_data_path):
                    os.makedirs(abs_data_path)
                np.save(abs_data_path + "rois_feature.npy", img_data_npy)
                np.save(abs_data_path + "fpn5_feature.npy", img_fpn_data_npy)
                ##img_data = []
                ##img_fpn_data = []

        '''    
        img1_fpn_data = []
        img2_fpn_data = []
        img1_data = []
        img2_data = []
        for i in range( int(len(img_data) / 2) ) :
            img1_data.append( img_data[2*i])
            img2_data.append( img_data[2*i+1])

            img1_fpn_data.append( img_fpn_data[2*i] )
            img2_fpn_data.append( img_fpn_data[2*i+1] )
        
        ##dataset = tf.data.Dataset.from_tensor_slices((images_concat, labels))
        ###dataset = dataset.shuffle(buffer_size=1000)
        '''
        ##return img1_data, img2_data, img1_fpn_data, img2_fpn_data, labels
