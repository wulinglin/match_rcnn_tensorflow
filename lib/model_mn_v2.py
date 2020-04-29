import keras
import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D
from keras.models import load_model, Model
from keras_applications.imagenet_utils import decode_predictions

from lib.model_new import MaskRCNN


class MatchRCNN:
    def __init__(self, mode, config, model_dir=None):
        # todo mold_inputs对么？
        if model_dir:
            self.model_mask = load_model(model_dir)
        else:
            self.model_mask = MaskRCNN(mode, config, model_dir)
        self.output = Model(inputs=self.model_mask.mold_inputs, output=self.model_mask.get_layer('output_rois').output)
        # self.output.predict()
        # self.conv1 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')
        # self.conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')
        # self.conv3 = Conv2D(32 * 2, (3, 3), activation='relu', padding='same')
        # self.conv4 = Conv2D(32 * 2, (3, 3), activation='relu', padding='same')

    def match_dataset(self, images, labels):
        # images_concat = [img1+img2 for img1,img2 in images]
        images_concat = []
        for img1, img2 in images:
            img = tf.concat(0, [img1, img2])
            images_concat.append(tf.convert_to_tensor(img))
        mask_images = self.output.predict(images)
        dataset = tf.data.Dataset.from_tensor_slices((mask_images, labels))
        dataset = dataset.shuffle(buffer_size=1000)
        return dataset

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
        prediction = keras.layers.Dense(1, activation=keras.activations.softmax, use_bias=True)(fc1)

        # 基于Model方法构建模型
        model = Model(inputs=dataset, outputs=prediction)
        # 编译模型
        model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        # # 训练配置，仅供参考
        model.fit(dataset, epochs=5, steps_per_epoch=4442)
        model.save('mn.h5')

    def build_match(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        # x1 = x1.view(x1.size(0), -1)
        # x2 = x2.view(x2.size(0), -1)
        x1 = Flatten()(x1)
        x1 = Dense(256, activation='relu')(x1)
        x2 = Flatten()(x2)
        x2 = Dense(256, activation='relu')(x2)

        x = x1 - x2
        x = x ** 2
        x = Flatten()(x)
        fc1 = Dense(256, activation='relu')(x)
        prediction = Dense(2, activation='relu', use_bias=True)(fc1)

        # 基于Model方法构建模型
        model = Model(inputs=x, outputs=prediction)

        # # 编译模型
        # model.compile(optimizer=tf.train.AdamOptimizer(0.01),
        #               loss=keras.losses.categorical_crossentropy,
        #               metrics=['accuracy'])
        # # # 训练配置，仅供参考
        # model.fit(dataset, epochs=5, steps_per_epoch=4442)
        # model.save('mn.h5')

    def predict(self, x1, x2, match_model_dir):
        """
        和一般的模型不同之处在于，正常是对一张图片做分类预测，这里是输入两个图片做预测
        :return:
        """
        model = load_model(match_model_dir)
        # 把图片数组联合在一起
        x = np.concatenate([x1, x2])
        load_model(match_model_dir)
        # model = ResNet50(weights='imagenet')
        y = model.predict(x)
        print('Predicted:', decode_predictions(y, top=3))
