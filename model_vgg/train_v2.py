# import utils
import time

import cv2
import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils

from model_vgg.model_v2 import VGG16

K.set_image_dim_ordering('tf')


def generate_arrays_from_file(lines, batch_size, num_classes):
    # 获取总长度
    n = len(lines)
    i = 0
    size = (224, 224)
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            box = eval(lines[i].split(';')[-1])
            # 从文件中读取图像
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            # x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            y1, x1, y2, x2 = box[0], box[1], box[2], box[3]  # 跟官方不一致哎
            img = img[x1:x2, y1:y2]
            # cv2.imshow('test',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            try:
                img = cv2.resize(img, size)
            except Exception:
                print('error img resize', lines[i], img)
                continue

            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])  # 加int or 不加
            # 读完一个周期后重新开始
            i = (i + 1) % n
        # 处理图像

        # X_train = utils.resize_image(X_train, (224, 224))
        X_train = np.array(X_train)
        X_train = X_train.reshape(-1, 224, 224, 3)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=num_classes)
        yield (X_train, Y_train)


def merge_file_lines(train_file_list, valid_file_list):
    """数据集分散在不同的文件里面需要合并"""

    def get_file_lines(path):
        df = pd.read_csv(path)
        lines = []
        for idx, row in df.iterrows():
            lines.append('{};{};{}\n'.format(row['path'], row['class'], row['box']))
        return lines

    train_lines, valid_lines = [], []
    for p in train_file_list:
        train_lines.extend(get_file_lines(p))

    for p in valid_file_list:
        valid_lines.extend(get_file_lines(p))
    return train_lines, valid_lines


def main(train_file_list, valid_file_list):
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    train_lines, valid_lines = merge_file_lines(train_file_list, valid_file_list)

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)

    np.random.seed(10102)
    np.random.shuffle(valid_lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = len(valid_lines)
    num_train = len(train_lines)

    num_classes = 23  # todo 23
    # 建立AlexNet模型
    model = VGG16(num_classes)

    # 注意要开启skip_mismatch和by_name todo path
    model.load_weights("./logs/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True, skip_mismatch=True)

    # 指定训练层
    for i in range(0, len(model.layers) - 5):
        model.layers[i].trainable = False

    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-2,decay=0.9),
                  # metrics=['accuracy']
                  metrics=[keras.metrics.categorical_accuracy]
                  )

    # 一次的训练集大小
    batch_size = 16

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(train_lines, batch_size, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(valid_lines, batch_size, num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=30,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])
    t=time.time()
    model.save_weights(log_dir + 'model_vgg_{}.h5'.format(int(t)))
