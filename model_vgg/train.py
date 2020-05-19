from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from constant import train_img_dir, valid_img_dir, path_head_save
from model_vgg.dataset_util import DataGeneratorVGG
from model_vgg.model import vgg
import tensorflow as tf
import json
import keras
import os


# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
# image_path = data_root + "/data_set/flower_data/"  # flower data set path
# train_dir = image_path + "train"
# validation_dir = image_path + "val"
train_dir = train_img_dir
validation_dir = valid_img_dir

# create direction for saving weights
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

im_height = 224
im_width = 224
batch_size = 32
epochs = 10

mode='train'
train_txt_path = path_head_save + 'live2vgg_{}.csv'.format(mode)
width, height = 224, 224
IMAGE_SIZE = (width, height, 3)
classes = 23
train_data_gen = DataGeneratorVGG(train_txt_path, 100, IMAGE_SIZE, classes)
# x, y = next(train_gen.get_mini_batch())

total_train = train_data_gen.num_of_examples

mode='valid'
valid_txt_path = path_head_save + 'live2vgg_{}.csv'.format(mode)
val_data_gen = DataGeneratorVGG(valid_txt_path, 100, IMAGE_SIZE, classes)
x, y = next(val_data_gen.get_mini_batch())
total_val = val_data_gen.num_of_examples


# train_image_generator = ImageDataGenerator(rescale=1. / 255,
#                                            horizontal_flip=True)
# validation_image_generator = ImageDataGenerator(rescale=1. / 255)
#
# train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
#                                                            batch_size=batch_size,
#                                                            shuffle=True,
#                                                            target_size=(im_height, im_width),
#                                                            class_mode='categorical')
# total_train = train_data_gen.n

# get class dict
# class_indices = train_data_gen.class_indices
#
# # transform value and key of dict
# inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
# json_str = json.dumps(inverse_dict, indent=4)
# with open('class_indices.json', 'w') as json_file:
#     json_file.write(json_str)
#
# val_data_gen = train_image_generator.flow_from_directory(directory=validation_dir,
#                                                          batch_size=batch_size,
#                                                          shuffle=True,
#                                                          target_size=(im_height, im_width),
#                                                          class_mode='categorical')
# total_val = val_data_gen.n

model = vgg("vgg16", 224, 224, 5)
model.summary()

# using keras high level api for training
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
#               loss=tf.keras.losses.categorical_crossentropy(from_logits=False),
#               metrics=["accuracy"])
# model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
#               loss=keras.losses.categorical_crossentropy, # from_logits=False
#               metrics=[keras.metrics.acc]
#               )

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss='sparse_categorical_crossentropy', # from_logits=False
              metrics=['accuracy']
              )
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./logs/myvgg_{epoch}.h5',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                monitor='val_loss')]
# import keras
# keras.callbacks.M

# tensorflow2.1 recommend to using fit
history = model.fit(x=x,
                    y=y,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    callbacks=callbacks)
