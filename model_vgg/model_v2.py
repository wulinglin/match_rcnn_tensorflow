import tensorflow as tf
from tensorflow import keras
from keras import Model,Sequential
from keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, MaxPooling2D, GlobalMaxPooling2D


def VGG16(num_classes):

    image_input = Input(shape = (224,224,3))
    # 第一个卷积部分
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(image_input)
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv2')(x)
    x = MaxPooling2D((2,2), strides = (2,2), name = 'block1_pool')(x)

    # 第二个卷积部分
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')(x)

    # 第三个卷积部分
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')(x)

    # 第四个卷积部分
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')(x)

    # 第五个卷积部分
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block5_pool')(x)

    # 分类部分
    x = Conv2D(256,(7,7),activation = 'relu',padding = 'valid', name = 'block6_conv4')(x)
    x = Flatten(name = 'flatten')(x)
    x = Dense(256,activation = 'relu',name = 'fullc1')(x)
    x = Dense(256,activation = 'relu',name = 'fullc2')(x)
    x = Dense(num_classes,activation = 'softmax',name = 'fullc3')(x)
    x = Dropout(0.2)(x)
    model = Model(image_input,x,name = 'vgg16')

    return model
