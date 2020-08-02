import cv2 as cv
import numpy as np
import os
from os import path
import matplotlib.pyplot as plt


def FT(img):
    """
    # 利用的是颜色特征和亮度特征。
    # 对图像img进行高斯滤波得到gfrgb；
    # 将图像imgrgb由RGB颜色空间转换为LAB颜色空间imglab；
    # 对转换后的图像imglab 的L,A,B三个通道的图像分别取均值得到lm,am,bm；
    # 计算显著值，即对分别对三个通道的均值图像和滤波得到的图像取欧氏距离并求和；
    # 利用最大值对显著图归一化。
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.GaussianBlur(img, (5, 5), 0)
    gray_lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)

    l_mean = np.mean(gray_lab[:, :, 0])
    a_mean = np.mean(gray_lab[:, :, 1])
    b_mean = np.mean(gray_lab[:, :, 2])
    lab = np.square(gray_lab - np.array([l_mean, a_mean, b_mean]))
    lab = np.sum(lab, axis=2)
    lab = lab / np.max(lab)

    return lab


def super_pixel(img):
    # 初始化slic项，超像素平均尺寸20（默认为10），平滑因子20
    slic = cv.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler=20.0)
    slic.iterate(10)  # 迭代次数，越大效果越好
    mask_slic = slic.getLabelContourMask()  # 获取Mask，超像素边缘Mask==1
    label_slic = slic.getLabels()  # 获取超像素标签
    number_slic = slic.getNumberOfSuperpixels()  # 获取超像素数目
    mask_inv_slic = cv.bitwise_not(mask_slic)
    img_slic = cv.bitwise_and(img, img, mask=mask_inv_slic)  # 在原图上绘制超像素边界
    return img_slic


if __name__ == '__main__':
    p = 'salience_img'
    imgs = [path.join(p, x) for x in os.listdir(p)]
    img_post = ['.jpg', '.png', '.jpeg']
    for i in imgs:
        for j in img_post:
            if i.endswith(j):
                img = cv.imread(i)
                res = FT(img)
                # res = super_pixel(img)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(res, cmap='gray')
                plt.savefig(path.join(p, 'salience_' + path.basename(i)), bbox_inches='tight')

