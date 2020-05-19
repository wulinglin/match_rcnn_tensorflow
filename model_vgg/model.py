# from tensorflow.python.keras import layers, models, Sequential
from keras import layers, models, Sequential
import tensorflow as tf
import keras
"""使用keras 则会：ValueError: Could not interpret optimizer identifier: <tensorflow.python.keras.optimizers.Adam object at """


def VGG(feature, im_height=224, im_width=224, class_num=1000):
    # tensorflow中的tensor通道排序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(class_num)(x)
    output = layers.Softmax()(x)
    # output = layers.softmax()(x)
    model = models.Model(inputs=input_image, outputs=output)
    return model


def features(cfg, input_shape):
    feature_layers = []
    for idx, v in enumerate(cfg):
        if idx == 0:
            conv2d = layers.Conv2D(v, kernel_size=3, padding="SAME", activation="relu", input_shape=input_shape)
            feature_layers.append(conv2d)
        elif v == "M":
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            conv2d = layers.Conv2D(v, kernel_size=3, padding="SAME", activation="relu")
            feature_layers.append(conv2d)
    print('feature_layers', feature_layers)
    return Sequential(feature_layers, name="feature")


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", im_height=224, im_width=224, class_num=1000):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    input_shape = (im_height, im_width, 3)
    model = VGG(features(cfg, input_shape), im_height=im_height, im_width=im_width, class_num=class_num)
    return model
