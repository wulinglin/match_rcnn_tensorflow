
# -*- coding: utf-8 -*-
import codecs
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

mode_name = "matchcnn"
mode_list = ["matchcnn"]
if mode_name not in mode_list:
    print("the current model has : ", mode_list)
    exit()

MODEL_PATH = "model_all/" 
if not os.path.exists( MODEL_PATH ):
    os.makedirs(MODEL_PATH)

##NB_LABELS = 2

label_size = 2
mode = mode_name
feature1_name = ".rois_feature.npy"
feature2_name = ".fpn5_feature.npy"


def generate_batch(df, load_maxnum, batch_size, random_flag ):
        """
        Generates a batch iterator for a dataset.
        """
        input_size = df.shape[0]
        # Shuffle the data
        if random_flag == True:
            df = df.sample(frac = 1 )

        load_time = int(math.ceil(float(input_size/load_maxnum)))
        for ii in range( load_time ):
            if (ii+1)*load_maxnum > input_size:
                df_tmp = df[ii*load_maxnum : ] 
            else:
                df_tmp = df[ii*load_maxnum : (ii+1)*load_maxnum]
            frame_fea1 = []
            frame_fea2 = []
            image_fea1 = []
            image_fea2 = []
            input_y = []
            frame_fea1_path = df_tmp["frame_fea1"].tolist()
            image_fea1_path = df_tmp["image_fea1"].tolist()
            frame_fea2_path = df_tmp["frame_fea2"].tolist()
            image_fea2_path = df_tmp["image_fea2"].tolist()
            label = df_tmp["label"].tolist()
            for f1,f2,i1,i2,la in zip(frame_fea1_path,frame_fea2_path,image_fea1_path,image_fea2_path, label ):
                frame_fea1.append( np.load( f1 ) )
                frame_fea2.append( np.load( f2 ) )
                image_fea1.append( np.load( i1 ) )
                image_fea2.append( np.load( i2 ) )
                input_y.append( la )

            input_size_tmp = df_tmp.shape[0]
            total_batch = int(math.ceil(float(input_size_tmp/batch_size)))
            print("total_batch:", total_batch)
            for batch_num in range(total_batch):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, input_size)
                batch_f1 = frame_fea1[start:end]
                batch_f2 = frame_fea2[start:end]
                batch_i1 = image_fea1[start:end]
                batch_i2 = image_fea2[start:end]
                batch_y = input_y[start:end]
                yield (batch_f1, batch_f2, batch_i1, batch_i2, batch_y)


def get_feature_info( df ):
    video_feature_path = df['video_feature_path'].tolist()
    video_frame_name = df["video_frame_name"].tolist()
    img_feature_path = df["img_feature_path"].tolist()
    img_name = df["img_name"].tolist()
    train_label  = df["label"].tolist()

    frame_fea1_path = []
    image_fea1_path = []
    frame_fea2_path = []
    image_fea2_path = []
    label_input = []
    for v1,v2, i1, i2 , label_tmp in zip(video_feature_path, video_frame_name, img_feature_path, img_name, train_label):
        frame_fea1_path.append(  v1.strip() + "/" + v2 + feature1_name)
        image_fea1_path.append(  i1.strip() + "/" + i2 + feature1_name)

        frame_fea2_path.append(  v1.strip() + "/" + v2 + feature2_name)
        image_fea2_path.append(  i1.strip() + "/" + i2 + feature2_name)
        if int(label_tmp) == 1: 
            label_input.append( [0,1] )
        else:
            label_input.append( [1,0] )
    df_info = pd.DataFrame({"frame_fea1":frame_fea1_path,"image_fea1":image_fea1_path,"frame_fea2":frame_fea2_path, \
                                                                                "image_fea2":image_fea2_path,"label":label_input})
    return df_info


def get_convfea1( input_feature, block_name, reuse_flag = tf.AUTO_REUSE, is_train = False):
    with tf.variable_scope(block_name, reuse=reuse_flag) as scope:
        conv1_weights = tf.get_variable("weight1",[2,2,256,256],initializer=tf.contrib.layers.variance_scaling_initializer())
        conv1_biases = tf.get_variable("bias1", [256], initializer=tf.constant_initializer(0.1))
        conv1 = tf.nn.conv2d(input_feature, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_biases)
        conv1 = tf.layers.batch_normalization(conv1,training=is_train)
        conv1_out = tf.nn.relu(conv1)

        conv2_weights = tf.get_variable("weight2",[2,2,256,256],initializer=tf.contrib.layers.variance_scaling_initializer())
        conv2_biases = tf.get_variable("bias2", [256], initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(conv1_out, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_biases)
        conv2 = tf.layers.batch_normalization(conv2,training=is_train)
        conv2_out = tf.nn.relu(conv2)

        conv12_add = conv1_out + conv2_out
        conv3_weights = tf.get_variable("weight3",[2,2,256,256],initializer=tf.contrib.layers.variance_scaling_initializer())
        conv3_biases = tf.get_variable("bias3", [256], initializer=tf.constant_initializer(0.1))
        conv3 = tf.nn.conv2d(conv12_add, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.bias_add(conv3, conv3_biases)
        conv3 = tf.layers.batch_normalization(conv3,training=is_train)
        conv3_out = tf.nn.relu(conv3)

        pool1 = tf.nn.max_pool(conv3_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("****pool1*****", pool1.get_shape() )


        conv4_weights = tf.get_variable("weight4",[2,2,256,512],initializer=tf.contrib.layers.variance_scaling_initializer())
        conv4_biases = tf.get_variable("bias4", [512], initializer=tf.constant_initializer(0.1))
        conv4 = tf.nn.conv2d(pool1, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = tf.nn.bias_add(conv4, conv4_biases)
        conv4 = tf.layers.batch_normalization(conv4,training=is_train)
        conv4_out = tf.nn.relu(conv4)

        conv5_weights = tf.get_variable("weight5",[2,2,512,512],initializer=tf.contrib.layers.variance_scaling_initializer())
        conv5_biases = tf.get_variable("bias5", [512], initializer=tf.constant_initializer(0.1))
        conv5 = tf.nn.conv2d(conv4_out, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = tf.nn.bias_add(conv5, conv5_biases)
        conv5 = tf.layers.batch_normalization(conv5,training=is_train)
        conv5_out = tf.nn.relu(conv5)

        conv45_add = conv4_out + conv5_out
        conv6_weights = tf.get_variable("weight6",[2,2,512,512],initializer=tf.contrib.layers.variance_scaling_initializer())
        conv6_biases = tf.get_variable("bias6", [512], initializer=tf.constant_initializer(0.1))
        conv6 = tf.nn.conv2d(conv45_add, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv6 = tf.nn.bias_add(conv6, conv6_biases)
        conv6 = tf.layers.batch_normalization(conv6,training=is_train)
        conv6_out = tf.nn.relu(conv6)

        pool2 = tf.nn.max_pool(conv6_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("****pool2*****", pool2.get_shape() )

        pool_dim = pool2.get_shape().as_list()[1] * pool2.get_shape().as_list()[2] * pool2.get_shape().as_list()[3]
        print("pool size:", pool_dim)

        conv6_flatten = tf.reshape(pool2,[-1,pool_dim])
        fea1 = tf.layers.dense(conv6_flatten, int(1024), activation=tf.nn.relu, reuse = reuse_flag, name="dense_layer")
        return fea1


def train( train_file, base_path ):
    
    # get info path and label
    data_df = pd.read_csv( train_file , encoding = "utf8")
    data_df = data_df.sample(frac = 1 ,  random_state = 1126)
    train_df = data_df[0 : int(data_df.shape[0] * 0.9)]
    test_df = data_df[int(data_df.shape[0] * 0.9) : ]
    train_df.to_csv("train_data_split.csv", index = False, encoding = "utf8")
    test_df.to_csv("test_data_split.csv", index = False, encoding = "utf8" )
    train_size = train_df.shape[0]
    test_size = test_df.shape[0]
    print("Train/Test split: {:d}/{:d}".format(train_size, test_size) )

    df_train_info = get_feature_info(train_df)
    df_test_info = get_feature_info(test_df)
        
    # save training result to file 
    best_accuracy = 0
    patient = 0
    f_train = codecs.open(base_path + "/train_acc_loss.txt","w",encoding = "utf8")
    f_train.write("step \tloss \taccuracy\n")
    f_test = codecs.open(base_path +  "/test_acc_loss.txt","w",encoding = "utf8")
    f_test.write("step \tloss \taccuracy\n")

    # train model
    train_epoch = 5000
    batch_size = 8
    load_maxnum = 10000
    learn_rate = 0.0001

    for epoch in range(train_epoch):
        training_batches = generate_batch(df_train_info, load_maxnum = load_maxnum, batch_size = batch_size, random_flag= True)
        print ('epoch {}'.format(epoch + 1))
        for batch_f1, batch_f2, batch_i1, batch_i2, batch_y in training_batches:
            time_start = time.time()
            feed_dict = dict()
            feed_dict[img1_fea1_ph] = batch_f1
            feed_dict[img1_fea2_ph] = batch_f2
            feed_dict[img2_fea1_ph] = batch_i1
            feed_dict[img2_fea2_ph] = batch_i2
            feed_dict[input_y_ph] = batch_y
            feed_dict[dropout_prob] = 0.8
            feed_dict[is_train] = True
            feed_dict[learning_rate] = learn_rate
            fetches = [train_op, global_step, loss, accuracy]
            _, global_step_tmp, train_loss, train_accuracy = session.run(fetches, feed_dict)
            step_time = time.time() - time_start
            sample_psec = batch_size / step_time
            print ("Train, step {}, loss {:g}, acc {:g}, step-time {:g}, examples/sec {:g}".format(global_step_tmp, train_loss, train_accuracy, step_time, sample_psec))
            f_train.write(str(global_step_tmp) + "\t" + str(train_loss) + "\t" + str(train_accuracy) + "\n")
            f_train.flush()

        # evaluate_test_set
        print("df_test_info_data: ", df_test_info.shape)
        test_time = time.time()
        testing_batches = generate_batch(df_test_info, load_maxnum = load_maxnum, batch_size = batch_size, random_flag= False)
        test_loss_all = []
        test_prediction_all = []
        test_pred_y = []
        test_true_y = []
        for batch_f1, batch_f2, batch_i1, batch_i2, batch_y in testing_batches:
            feed_dict = dict()
            feed_dict[img1_fea1_ph] = batch_f1
            feed_dict[img1_fea2_ph] = batch_f2
            feed_dict[img2_fea1_ph] = batch_i1
            feed_dict[img2_fea2_ph] = batch_i2
            feed_dict[input_y_ph] = batch_y
            feed_dict[dropout_prob] = 1.0
            feed_dict[is_train] = False

            fetches = [loss, correct_prediction,pre_y]
            test_loss, test_prediction,pre_y_tmp = session.run(fetches, feed_dict)
            test_loss_all.append(test_loss)
            test_prediction_all.append(test_prediction)
            test_pred_y.append( pre_y_tmp )
            test_true_y.append(  np.argmax(batch_y, axis =1 ) )
        step_time = time.time() - test_time
        sample_psec = len( np.concatenate(test_true_y ) ) / step_time
        tess_loss = np.mean(test_loss_all)
        test_accuracy = np.mean(np.concatenate(test_prediction_all))
        test_pred_y_all = np.concatenate(test_pred_y )
        test_true_y_all = np.concatenate(test_true_y )
        print(classification_report(test_true_y_all, test_pred_y_all))
        print ("Test, loss {:g}, acc {:g}, step-time {:g}, examples/sec {:g}".format(test_loss, test_accuracy, step_time, sample_psec))
        f_test.write(str(global_step_tmp) + "\t" + str(test_loss) + "\t" + str(test_accuracy) + "\n")
        f_test.flush()

        if test_accuracy > best_accuracy:
            patient = 0
            best_accuracy = test_accuracy
            model_path = saver.save(session, checkpoint_prefix, global_step=global_step_tmp)
            print("Saved model checkpoint to {}\n".format(model_path))
        else:
            patient += 1
            print ("no_improve", patient,"best accuracy is ", best_accuracy, "learning_rate :", learn_rate)
            if patient > 3:
                learn_rate = learn_rate / 5.0
        if  patient > 10:
            print ("no_improve", patient,"best acc is ", best_accuracy, "learning_rate :", learn_rate)
            print("accruracy has not imporved at {} time , training model done!".format(patient) )
            break


# create model
with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    session = tf.Session(config=config)
    ##with sess.as_default():
    img1_fea1_ph = tf.placeholder(tf.float32, shape = [None, 14,14,256], name="video_fea1")
    img2_fea1_ph = tf.placeholder(tf.float32, shape = [None, 14,14,256], name="img_fea1")

    img1_fea2_ph = tf.placeholder(tf.float32, shape = [None, 32,32,256], name="video_fea2")
    img2_fea2_ph = tf.placeholder(tf.float32, shape = [None, 32,32,256], name="img_fea2")

    dropout_prob = tf.placeholder(tf.float32, name="keep_prob")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    input_y_ph = tf.placeholder(tf.int32, shape = (None, label_size), name="input_label")
    is_train = tf.placeholder(tf.bool,name='is_train')


    with tf.variable_scope("fea1_layer"):
        img1_fea1_dense1 = get_convfea1( img1_fea1_ph, block_name = "fea1_conv", reuse_flag = None, is_train = is_train) ###tf.AUTO_REUSE,
        img2_fea1_dense1 = get_convfea1( img2_fea1_ph, block_name = "fea1_conv", reuse_flag = True, is_train = is_train) 


    with tf.variable_scope("fea2_layer"):
        img1_fea2_dense2 = "" ##get_convfea1( img1_fea1_ph, block_name = "fea1_conv", reuse_flag = None, is_train = is_train) ###tf.AUTO_REUSE,
        img2_fea2_dense2 = ""##get_convfea1( img2_fea1_ph, block_name = "fea1_conv", reuse_flag = True, is_train = is_train) 


    with tf.variable_scope("inter_layer"):
        img1_fea_all = img1_fea1_dense1 #tf.concat([img1_fea1_dense1, img1_fea2_dense2], 1)
        img2_fea_all = img2_fea1_dense1 #tf.concat([img2_fea1_dense1, img2_fea2_dense2], 1)
        img1_fea_all = tf.nn.dropout(img1_fea_all, dropout_prob)
        img2_fea_all = tf.nn.dropout(img2_fea_all, dropout_prob)

        img1_fea_all_output = tf.layers.dense(img1_fea_all, 1024, activation = tf.tanh, name = "merge1")
        img2_fea_all_output = tf.layers.dense(img2_fea_all, 1024, activation = tf.tanh, reuse = True, name ="merge1")
        
        '''
        img1_fea_all_output1 = tf.nn.dropout(img1_fea_all_output, dropout_prob)
        img2_fea_all_output1 = tf.nn.dropout(img2_fea_all_output, dropout_prob)
           img1_fea_all_output = tf.layers.dense(img1_fea_all_output1, 512, activation = tf.tanh, name = "merge2")
        img2_fea_all_output = tf.layers.dense(img2_fea_all_output1, 512, activation = tf.tanh, reuse = True, name ="merge2")
        '''

    with tf.variable_scope("output_layer"):
        f_x1x2 = tf.reduce_sum(tf.multiply(img1_fea_all_output, img2_fea_all_output), 1)
        norm_fx1 = tf.sqrt(tf.reduce_sum(tf.square(img1_fea_all_output),1))
        norm_fx2 = tf.sqrt(tf.reduce_sum(tf.square(img2_fea_all_output),1))
        Ew = f_x1x2 / (norm_fx1 * norm_fx2 +  1e-10)

        cos_one = 0.5 * Ew + 0.5
        cos_zero = 1 - cos_one
        scores = tf.concat([tf.expand_dims(cos_zero,-1), tf.expand_dims(cos_one,-1)], -1)
        pre_proba = tf.nn.softmax(scores,name = "softmax_proba") 
        pre_y = tf.argmax(scores, 1, name="prediction")
        correct_prediction = tf.equal(pre_y, tf.argmax(input_y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


    with tf.name_scope("loss_layer"): 
        l_1 = 0.25 * tf.square(1 - Ew)
        l_0 = tf.square( tf.maximum(Ew + 0.1, 0) )
        label = tf.to_float( tf.argmax(input_y_ph, 1, name="label") )
        loss = 0.5 * tf.reduce_mean(label * l_1 + (1 - label) * l_0)
        ##loss = tf.add_n(tf.get_collection("losses"))
        ##loss = loss1 + loss2
    
    global_step = tf.Variable(0, name="global_step", trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9,beta2=0.999,epsilon=1e-08)
        train_vars = tf.trainable_variables()
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    #    train_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_all)


    '''''''''''''''''''''''''''''''''''''''''''''
    TRAIN
    '''''''''''''''''''''''''''''''''''''''''''''
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep= 1)

    base_path = os.path.abspath(os.path.join(MODEL_PATH, mode) )
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    checkpoint_prefix = os.path.join(base_path, "model")

    train_file = sys.argv[1]
    train( train_file, base_path)