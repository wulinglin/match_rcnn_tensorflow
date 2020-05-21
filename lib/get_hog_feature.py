'''
说明：利用python/numpy/opencv实现图像HOG特征的提取
算法思路：
算法思路:
        1)以灰度图的方式加载图片，resize到（128,64）;
        2）灰度图像gamma校正;
		3)利用一阶微分算子Sobel函数，分别计算出灰度图像X方向和Y方向上的一阶微分/梯度图像，根据得到的两幅
        梯度图像(X方向上的梯度图像和Y方向上的梯度图像)，计算出这两幅梯度图像所对应的梯度幅值图像gradient_magnitude、
        梯度方向图像gradient_angle
		4)构造(cell_x = 128/8 =16, cell_y= 64/8 =8)大小的cell图像----梯度幅值的grad_cell图像，梯度方向的ang_cell图像，
        每个cell包含有8*8 = 64个值；
		5)将每个cell根据角度值（0-180）分为9个bin，并计算每个cell中的梯度方向直方图,每个cell有9个值；
		6)每（2*2）个cell为一个block，总共15*7个block,计算每个block的梯度方向直方图，并进行归一化处理，每个block中有9*4=36个值；
		7)计算整幅图像的梯度方向直方图HOG:将计算出来的所有的Block的HOG梯度方向直方图的特征向量首尾相接组成一个维度很大的向量
        长度为：15*7*36 = 3780，
        这个特征向量就是整幅图像的梯度方向直方图特征，这个特征可用于SVM分类。
'''
import cv2
import h5py
import numpy as np


# 灰度图像gamma校正
def gamma(img):
    # 不同参数下的gamma校正
    # img1 = img.copy()
    # img2 = img.copy()
    # img1 = np.power( img1 / 255.0, 0.5 )
    # img2 = np.power( img2 / 255.0, 2.2 )
    return np.power(img / 255.0, 1)


# 获取梯度值cell图像，梯度方向cell图像
def div(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x, axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell


# 获取梯度方向直方图图像，每个像素点有9个值
def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = np.int8(grad_cell[i, j].flatten())  # 每个cell中的64个梯度值展平，并转为整数
            ang_list = ang_cell[i, j].flatten()  # 每个cell中的64个梯度方向展平)
            ang_list = np.int8(ang_list / 20.0)  # 0-9
            ang_list[ang_list >= 9] = 0
            for m in range(len(ang_list)):
                binn[ang_list[m]] += int(grad_list[m])  # 不同角度对应的梯度值相加，为直方图的幅值
            # 每个cell的梯度方向直方图可视化
            # N = 9
            # x = np.arange( N )
            # str1 = ( '0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140', '140-160', '160-180' )
            # plt.bar( x, height = binn, width = 0.8, label = 'cell histogram', tick_label = str1 )
            # for a, b in zip(x, binn):
            # plt.text( a, b+0.05, '{}'.format(b), ha = 'center', va = 'bottom', fontsize = 10 )
            # plt.show()
            bins[i][j] = binn
    return bins


# 计算图像HOG特征向量，长度为 15*7*36 = 3780
def hog(img, cell_x, cell_y, cell_w):
    height, width = img.shape
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x方向梯度
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y方向梯度
    gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
    gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)
    print(gradient_magnitude.shape, gradient_angle.shape)
    # plt.figure()
    # plt.subplot( 1, 2, 1 )
    # plt.imshow(gradient_angle)
    # 角度转换至（0-180）
    gradient_angle[gradient_angle > 0] *= 180 / 3.14
    gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + 3.14) * 180 / 3.14
    # plt.subplot( 1, 2, 2 )
    # plt.imshow( gradient_angle )
    # plt.show()

    grad_cell = div(gradient_magnitude, cell_x, cell_y, cell_w)
    ang_cell = div(gradient_angle, cell_x, cell_y, cell_w)
    bins = get_bins(grad_cell, ang_cell)
    feature = []
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = []
            tmp.append(bins[i, j])
            tmp.append(bins[i + 1, j])
            tmp.append(bins[i, j + 1])
            tmp.append(bins[i + 1, j + 1])
            tmp -= np.mean(tmp)
            feature.append(tmp.flatten())
    return np.array(feature).flatten()

def get_hog_feauture_by_path(image_name_path, box):
    size = (128, 64)
    img = cv2.imread(image_name_path, cv2.IMREAD_GRAYSCALE)
    if (img is None):
        print('Not read image.')
        return
    y1, x1, y2, x2 = box[0], box[1], box[2], box[3]  # 跟官方不一致哎
    img = img[x1:x2, y1:y2]
    try:
        resizeimg = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    except Exception:
        print('error img resize', image_name_path, img)
        return
    cell_w = 8
    cell_x = int(resizeimg.shape[0] / cell_w)  # cell行数
    cell_y = int(resizeimg.shape[1] / cell_w)  # cell列数
    gammaimg = gamma(resizeimg) * 255
    feature = hog(gammaimg, cell_x, cell_y, cell_w)
    return feature


def save_hog_feature(df_list, test_cos_image_feature_path, test_cos_frame_feature_path):
    image_h5_file = h5py.File(test_cos_image_feature_path, 'w')
    video_h5_file = h5py.File(test_cos_frame_feature_path, 'w')

    for each in df_list:
        image_name_path = each['path']
        class_dict_index = each['class']
        box = each['box']
        feature = get_hog_feauture_by_path(image_name_path, box)
        # print(feature.shape, image_name_path)

        index = image_name_path.split('/')[-2]
        img_or_video = image_name_path.split('/')[-3]
        if img_or_video == 'video_cut':
            video_h5_file.create_dataset(index, data=feature)
        else:
            image_h5_file.create_dataset(index, data=feature)
    video_h5_file.close()
    image_h5_file.close()

def save_cascade_hog_feature(df_list, test_cos_image_feature_path, test_cos_frame_feature_path):
    image_h5_file = h5py.File(test_cos_image_feature_path, 'w')
    video_h5_file = h5py.File(test_cos_frame_feature_path, 'w')

    for each in df_list:
        image_name_path = each['path']
        class_dict_index = each['class']
        box = each['box']
        feature = get_hog_feauture_by_path(image_name_path, box)
        # print(feature.shape, image_name_path)

        index = image_name_path.split('/')[-2]
        img_or_video = image_name_path.split('/')[-3]
        if img_or_video == 'video_cut':
            video_h5_file.create_dataset(index, data=feature)
        else:
            image_h5_file.create_dataset(index, data=feature)
    video_h5_file.close()
    image_h5_file.close()


def save_rcca_hog_feature(df_list, test_cos_image_feature_path, test_cos_frame_feature_path):
    image_h5_file = h5py.File(test_cos_image_feature_path, 'w')
    video_h5_file = h5py.File(test_cos_frame_feature_path, 'w')

    for each in df_list:

        for each_img in each['image']:
            image_name_path = each['path']
            class_dict_index = each['class']
            box = each['box']
            feature = get_hog_feauture_by_path(image_name_path, box)

        for each_video in each['video']:
            pass # todo

        # print(feature.shape, image_name_path)

        index = image_name_path.split('/')[-2]
        img_or_video = image_name_path.split('/')[-3]
        if img_or_video == 'video_cut':
            video_h5_file.create_dataset(index, data=feature)
        else:
            image_h5_file.create_dataset(index, data=feature)
    video_h5_file.close()
    image_h5_file.close()


# if __name__ == '__main__':
# img = cv2.imread('./data/basketball1.png', cv2.IMREAD_GRAYSCALE)
# if (img is None):
#     print('Not read image.')
# print(img.shape)
# resizeimg = cv2.resize(img, (128, 64), interpolation=cv2.INTER_CUBIC)
# cell_w = 8
# cell_x = int(resizeimg.shape[0] / cell_w)  # cell行数
# cell_y = int(resizeimg.shape[1] / cell_w)  # cell列数
# print('The size of cellmap is {}*{} '.format(cell_x, cell_y))
# gammaimg = gamma(resizeimg) * 255
# feature = hog(gammaimg, cell_x, cell_y, cell_w)
# print(feature.shape)
