import keras
import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D
from keras.models import load_model, Model
from keras_applications.imagenet_utils import decode_predictions

from lib.model_new import MaskRCNN
from lib.config import Config


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
    def __init__(self, images, mode, config, model_dir=None):
        self.config = config
        ###self.config = Config()
        self.model_mask = MaskRCNN("inference", config, model_dir)
        self.model_mask.load_weights(model_dir, by_name=True)
        self.model_mask_keras = self.model_mask.keras_model
        self.output = Model(inputs=self.model_mask_keras.inputs, output= \
                        [ self.model_mask_keras.get_layer('roi_align_classifier').output, self.model_mask_keras.get_layer('mrcnn_class_logits').output] )

    def match_dataset(self, images, labels):
        print("img:", len(images) )
        print("img_label:", len(labels) )

        images_concat = []
        img_data = []
        for img1,img2 in images:
            images_concat.extend( [img1,img2] )
        #img1_data = []
        #img2_data = []
        for img1 in images_concat:
            # Mold inputs to format expected by the neural network
            ###molded_images, image_metas, windows = self.model_mask.mold_inputs([img1])
            #print( self.config.BATCH_SIZE )            
            ##print( self.config.BATCH_SIZE )
            images_tmp = [ img1 ] 
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
            output_rois = self.output.predict([molded_images, image_metas, anchors], verbose=0)

            print("len ", len(output_rois) ) 
            print("output_rois[0].shape:", output_rois[0].shape)
            print("output_rois[1].shape:", output_rois[1].shape)
            print("output_rois[1].shape:", np.max(output_rois[1], axis = 2).shape)
            output_rois_score = np.max(output_rois[1], axis = 2 )
            print( output_rois_score.shape)

            output_rois_score_maxindex = np.argmax(output_rois_score, axis = 1)
            print( output_rois_score_maxindex.shape)

            #np.armax(output_rois[1] , axis = 2)
            img_data.append( output_rois[0][0][output_rois_score_maxindex[0]] )
            ##img2_data.append( output_rois[1] )
            
            #output_rois
            #images_concat.append( output_rois )

            ##break
            # print(img1.shape)
            # images_concat.extend([img1 , img2])
        # # print(images_concat.shape)
        # # mask_images = self.output.predict(images_concat)
        # for img in images_concat:
        #
        #     mask_images = self.output.predict([img])
        #     print(mask_images)
        img1_data = []
        img2_data = []
        for i in range( int(len(img_data) / 2) ) :
            img1_data.append( img_data[2*i])
            img2_data.append( img_data[2*i+1])

        print('hhhhhhh')
        print( len(img1_data) )
        print( len(img2_data) )
        
        ##dataset = tf.data.Dataset.from_tensor_slices((images_concat, labels))
        ###dataset = dataset.shuffle(buffer_size=1000)
        return img1_data, img2_data, labels ###dataset



    def match_base(self, image_tensor):        
        conv1 = Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same')(image_tensor)
        conv2 = Conv2D(256, (2, 2), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(256, (2, 2), activation='relu', padding='same')(conv2)
        pooling1 = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv3)

        conv4 = Conv2D(512, (2, 2), activation='relu', padding='same')(pooling1)
        conv5 = Conv2D(512, (2, 2), activation='relu', padding='same')(conv4)
        conv6 = Conv2D(512, (2, 2), activation='relu', padding='same')(conv5)
        #pooling2 = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv6)

        #conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(pooling2)
        #conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
        #conv9 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv8)
        #pooling3 = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv9)

        flat = Flatten()(conv6)
        net_base = Dense(1024, activation='relu')(flat)
        return net_base
        '''
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

    def siamese( self ):
        ##dataset = self.match_dataset(images, labels)   
        input_tensor = keras.Input(shape=[7, 7, 256])
        base_model = Model(input_tensor,self.match_base(input_tensor))

        input_im1 = keras.Input(shape=[7, 7, 256])
        input_im2 = keras.Input(shape=[7, 7, 256])

        out_im1 = base_model(input_im1)
        out_im2 = base_model(input_im2)
        diff = keras.layers.Subtract()([out_im1,out_im2])

        out = Dense(1024,activation='relu')(diff)
        out = Dense(1,activation='sigmoid')(out)
        model = Model([input_im1,input_im2],out)
        return model

    def build_match_v2(self, images, labels):
        
        model = self.siamese()
        img1_rois, img2_rois, labels = self.match_dataset(images, labels)
        model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
                      loss=keras.losses.sparse_categorical_crossentropy, ##keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

        print( "************",labels )
        model.fit([img1_rois, img2_rois], labels, epochs=50,batch_size=8)#  steps_per_epoch=4442 训练配置，仅供参考
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
