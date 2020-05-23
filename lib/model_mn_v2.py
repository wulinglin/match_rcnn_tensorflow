import keras
import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D, concatenate
from keras.models import load_model, Model

from lib import utils
from lib.model_new import MaskRCNN
from tools.data_utils import load_image, get_item_id_by_path


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class MatchRCNN:
    def __init__(self, mode, config, model_dir=None):
        self.config = config
        ###self.config = Config()
        self.model_mask = MaskRCNN(mode, config, model_dir)
        self.model_mask.load_weights(model_dir, by_name=True)
        self.model_mask_keras = self.model_mask.keras_model
        self.output = Model(inputs=self.model_mask_keras.inputs, output= \
            [self.model_mask_keras.get_layer('fpn_p5').output, \
             self.model_mask_keras.get_layer('roi_align_classifier').output, \
             self.model_mask_keras.get_layer('mrcnn_class_logits').output])

    def match_dataset(self, images, labels):
        print("img:", len(images))
        print("img_label:", len(labels))

        images_concat = []
        img_data = []
        img_fpn_data = []
        for img1, img2 in images:
            images_concat.extend([img1, img2])
        # img1_data = []
        # img2_data = []
        for img1 in images_concat:
            # Mold inputs to format expected by the neural network
            ###molded_images, image_metas, windows = self.model_mask.mold_inputs([img1])
            # print( self.config.BATCH_SIZE )
            ##print( self.config.BATCH_SIZE )
            images_tmp = [img1]
            ###images_tmp = [img1, img2 ]
            ##self.config.BATCH_SIZE = 2
            ###assert len( images_tmp) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"
            log("Processing {} images".format(len(images_tmp)))
            ##for image in images_tmp:
            ##    log("image", image)

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
            ###print("molded_images", molded_images)
            ##print("image_metas", image_metas)
            ###print("anchors", anchors)
            # Run object detection
            fpn_p5, output_rois, output_rois_score = self.output.predict([molded_images, image_metas, anchors],
                                                                         verbose=0)

            ###print("len ", len(output_rois) ) 
            # print("fpn_p5.shape:", fpn_p5.shape)
            # print("output_rois.shape:", output_rois.shape)
            # print("output_rois_score.shape:", output_rois_score.shape)
            # print("output_rois_score_max.shape:", np.max(output_rois_score, axis=2).shape)
            output_rois_score_max = np.max(output_rois_score, axis=2)
            output_rois_score_maxindex = np.argmax(output_rois_score_max, axis=1)
            print(output_rois_score_maxindex.shape)

            # np.armax(output_rois[1] , axis = 2)
            img_fpn_data.append(fpn_p5[0])
            img_data.append(output_rois[0][output_rois_score_maxindex[0]])
            ##img2_data.append( output_rois[1] )

            # output_rois
            # images_concat.append( output_rois )

            ##break
            # print(img1.shape)
            # images_concat.extend([img1 , img2])
        # # print(images_concat.shape)
        # # mask_images = self.output.predict(images_concat)
        # for img in images_concat:
        #
        #     mask_images = self.output.predict([img])
        #     print(mask_images)
        img1_fpn_data = []
        img2_fpn_data = []
        img1_data = []
        img2_data = []
        for i in range(int(len(img_data) / 2)):
            img1_data.append(img_data[2 * i])
            img2_data.append(img_data[2 * i + 1])

            img1_fpn_data.append(img_fpn_data[2 * i])
            img2_fpn_data.append(img_fpn_data[2 * i + 1])

        print('hhhhhhh')
        print(len(img1_data))
        print(len(img2_data))

        ##dataset = tf.data.Dataset.from_tensor_slices((images_concat, labels))
        ###dataset = dataset.shuffle(buffer_size=1000)
        return img1_data, img2_data, img1_fpn_data, img2_fpn_data, labels  ###dataset

    def match_base(self, image_tensor1):
        conv1 = Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same')(image_tensor1)
        conv2 = Conv2D(256, (2, 2), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(256, (2, 2), activation='relu', padding='same')(conv2)
        pooling1 = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv3)

        conv4 = Conv2D(512, (2, 2), activation='relu', padding='same')(pooling1)
        conv5 = Conv2D(512, (2, 2), activation='relu', padding='same')(conv4)
        conv6 = Conv2D(512, (2, 2), activation='relu', padding='same')(conv5)

        flat = Flatten()(conv6)
        net_base = Dense(1024, activation='relu')(flat)
        return net_base

    def match_base_fpn(self, image_tensor1_fpn):
        conv1 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(image_tensor1_fpn)
        conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv1)
        ##conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
        pooling1 = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv2)

        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pooling1)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        pooling2 = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv6)

        conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(pooling2)
        conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
        conv9 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv8)
        pooling3 = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv9)

        flat = Flatten()(pooling3)
        net_base_fpn = Dense(512, activation='relu')(flat)
        return net_base_fpn

    def siamese(self):
        ##dataset = self.match_dataset(images, labels)   
        input_tensor = keras.Input(shape=[7, 7, 256])
        base_model = Model(input_tensor, self.match_base(input_tensor))
        input_im1 = keras.Input(shape=[7, 7, 256])
        input_im2 = keras.Input(shape=[7, 7, 256])
        out_im1 = base_model(input_im1)
        out_im2 = base_model(input_im2)

        input_tensor_fpn = keras.Input(shape=[32, 32, 256])
        base_model_fpn = Model(input_tensor_fpn, self.match_base_fpn(input_tensor_fpn))
        input_im1_fpn = keras.Input(shape=[32, 32, 256])
        input_im2_fpn = keras.Input(shape=[32, 32, 256])
        out_im1_fpn = base_model_fpn(input_im1_fpn)
        out_im2_fpn = base_model_fpn(input_im2_fpn)

        out1_merge = concatenate([out_im1, out_im1_fpn], axis=1)  # merge([out_im1, out_im1_fpn], mode='contact')
        out2_merge = concatenate([out_im2, out_im2_fpn], axis=1)  ##merge([out_im2, out_im2_fpn], mode='contact')

        out1 = Dense(1024, activation='relu')(out1_merge)
        out2 = Dense(1024, activation='relu')(out2_merge)
        # K.concatenate([a , b] , axis=0)

        diff = keras.layers.Subtract()([out1, out2])
        mul = keras.layers.Multiply()([out1, out2])

        out_merge = concatenate([diff, mul], axis=1)
        out = Dense(512, activation='relu')(out_merge)
        out = Dense(1024, activation='relu')(out)
        out = Dense(1, activation='sigmoid')(out)
        model = Model([input_im1, input_im2, input_im1_fpn, input_im2_fpn], out)
        return model

    def build_match_v2(self, img1_rois_data, img2_rois_data, img1_fpn_data, img2_fpn_data, labels):

        model = self.siamese()
        # img1_rois, img2_rois, img1_fpn1, img2_fpn2, labels = self.match_dataset(images, labels)

        model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss=keras.losses.sparse_categorical_crossentropy,  ##keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

        print("************", labels)
        model.fit([img1_rois_data, img2_rois_data, img1_fpn_data, img2_fpn_data], labels, epochs=50,
                  batch_size=8)  # steps_per_epoch=4442 训练配置，仅供参考
        model.save('mn.h5')

    '''
    def build_match_v2(self, images, labels):
        dataset = self.match_dataset(images, labels)
        inputs = keras.Input(shape=[150, 150, 3])
        conv1 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pooling5 = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv4)
        flat = Flatten()(pooling5)
        fc1 = Dense(1024, activation='relu')(flat)
        prediction = keras.layers.Dense(1, activation=keras.activations.softmax, use_bias=True)(fc1)# 基于Model方法构建模型
        model = Model(inputs=dataset, outputs=prediction)
        # 编译模型
        model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        model.fit(dataset, epochs=5, steps_per_epoch=4442)# 训练配置，仅供参考
        model.save('mn.h5')
    '''

    def image_prepare(self, path):
        image = load_image(path)
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            min_scale=self.config.IMAGE_MIN_SCALE,
            max_dim=self.config.IMAGE_MAX_DIM,
            mode=self.config.IMAGE_RESIZE_MODE)
        return image

    def predict(self, test_path_pair_dict, match_model_dir):
        """
        和一般的模型不同之处在于，正常是对一张图片做分类预测，这里是输入两个图片做预测
        :return:
        """

        model = load_model(match_model_dir)
        result = {}
        for video_path, img_path_list in test_path_pair_dict:
            video_id = get_item_id_by_path(video_path)
            images = []
            pred_list = []
            for img_path in img_path_list:
                img1, img2 = self.image_prepare(video_path), self.image_prepare(img_path)
                images.append((img1, img2))
                img1_rois, img2_rois, img1_fpn1, img2_fpn2, labels = self.match_dataset(images, [])
                pred = model.predict([img1_rois, img2_rois, img1_fpn1, img2_fpn2])
                # todo
                pred_list.append(pred)
            img_path_pred = pred_list.index(max(pred_list))
            item_id = get_item_id_by_path(img_path_pred)
            result[video_id]=item_id
        return result
    #
    # model = self.siamese()
    #     img1_rois, img2_rois, img1_fpn1, img2_fpn2, labels = self.match_dataset(images, labels)
    #
    #     model = load_model(match_model_dir)
    #     # 把图片数组联合在一起
    #     x = np.concatenate([x1, x2])
    #     load_model(match_model_dir)
    #     # model = ResNet50(weights='imagenet')
    #     y = model.predict(x)
    #     print('Predicted:', decode_predictions(y, top=3))
