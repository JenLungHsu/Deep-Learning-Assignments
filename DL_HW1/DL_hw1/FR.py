#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from numpy import linalg as LA
import torchvision.models as models
import torch
import torch.nn as nn
import cv2
import numpy as np

# import os
# dd=os.listdir('TIN')                                 #TIN 中的所有文件和子目錄, 即每張照片
# f1 = open('train.txt', 'w')                          #打開文件, 寫入模式
# f2 = open('test.txt', 'w')                           #打開文件, 寫入模式
# for i in range(len(dd)):                             #len(dd) 代表有幾張照片
#     d2 = os.listdir ('TIN/%s/images/'%(dd[i]))       #images 中的所有文件和子目錄, 即每張照片中的小物件
#     for j in range(len(d2)-2):                       #留下最後一張當作測試集
#         str1='TIN/%s/images/%s'%(dd[i], d2[j])       #建立每張照片中小物件的完整目錄
#         f1.write("%s %d\n" % (str1, i))              #寫入此目錄, 加以編號
#     str1='TIN/%s/images/%s'%(dd[i], d2[-1])          #建立最後一張照片中小物件的完整目錄
#     f2.write("%s %d\n" % (str1, i))                  #寫入此目錄, 加以編號

# f1.close()                                           #關閉文件
# f2.close()                                           #關閉文件


# 提取SIFT特征
def extract_sift_feature(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    if descriptors is None:
        descriptors = np.array([]).reshape(-1, 128)

    # 確保特徵數量為64
    if descriptors.shape[0] < 64:
        descriptors = np.vstack(
            [descriptors, np.zeros((64 - descriptors.shape[0], 128))])
    elif descriptors.shape[0] > 64:
        descriptors = descriptors[:64, :]

    return descriptors

# 提取ORB特征


def extract_orb_features(image):
    # 创建ORB特征提取器
    orb = cv2.ORB_create()

    # 检测关键点并计算描述符
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # 如果未检测到特征，则返回空特征矩阵
    if descriptors is None:
        descriptors = np.zeros((64, 32))  # 直接生成全零特徵矩陣
    else:
        # 確保特徵數量為64
        if descriptors.shape[0] < 64:
            descriptors = np.vstack(
                [descriptors, np.zeros((64 - descriptors.shape[0], 32))])
        elif descriptors.shape[0] > 64:
            descriptors = descriptors[:64, :]

    return descriptors


def extract_brisk_features(image):
    # 创建BRISK特征提取器
    brisk = cv2.BRISK_create()

    # 检测关键点并计算描述符
    keypoints, descriptors = brisk.detectAndCompute(image, None)

    # 如果未检测到特征，则返回空特征矩阵
    if descriptors is None:
        descriptors = np.zeros((64, 64))  # 直接生成全零特徵矩陣

    # 確保特徵數量為64
    if descriptors.shape[0] < 64:
        descriptors = np.vstack(
            [descriptors, np.zeros((64 - descriptors.shape[0], 64))])
    elif descriptors.shape[0] > 64:
        descriptors = descriptors[:64, :]

    return descriptors


def load_img_features(f):  # 輸入: 文件名

    f = open(f)  # 打開文件, 讀取模式
    lines = f.readlines()  # 讀取文件的所有行, 並儲存在列表中

    imgs, lab, sift_f, orb_f, brisk_f, resnet_f = [], [], [], [], [], []

    for i in range(len(lines)):
        fn, label = lines[i].split(' ')  # 儲存文件路徑和標註
        label = int(label)
        if label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:

            im1 = cv2.imread(fn)  # 讀取圖像文件
            im1 = cv2.resize(im1, (256, 256))  # 將圖像調整大小為 256*256
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)  # 彩色轉灰色

            im_tensor = torch.Tensor(im1).unsqueeze(
                0).unsqueeze(0)  # 添加批次和通道維度

            '''===============================
            影像處理的技巧可以放這邊，來增強影像的品質
        
            ==============================='''

            # # 對比度增強
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # im1 = clahe.apply(im1)

            # # 銳化
            # kernel = np.array(
            #     [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
            # im1 = cv2.filter2D(im1, -1, kernel)

            # # 去噪
            # im1 = cv2.GaussianBlur(im1, (5, 5), 0)

            '''===============================
            三個特徵提取方法: SIFT, ORB, BRISK

            ==============================='''

            sift_features = extract_sift_feature(im1)    # (64, 128)
            orb_features = extract_orb_features(im1)     # (64, 32)
            brisk_features = extract_brisk_features(im1)  # (64, 64)

            '''===============================
            基於學習的特徵提取方法: RESNET

            ==============================='''

            pretrained_resnet = models.resnet18(
                pretrained='imagenet')  # 加載預訓練的ResNet模型
            pretrained_resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 將預訓練模型的輸入通道數修改為1
            pretrained_resnet.fc = nn.Identity()  # 刪除最後一層全連接層
            with torch.no_grad():
                feature_map = pretrained_resnet(im_tensor)

            '''===============================
            二維轉為一維

            ==============================='''

            vec = np.reshape(im1, [-1])  # 二維的灰色圖像轉為一維
            imgs.append(vec)  # 添加到列表中
            lab.append(int(label))  # 添加到列表中

            sift_f.append(np.reshape(sift_features, [-1]))
            orb_f.append(np.reshape(orb_features, [-1]))
            brisk_f.append(np.reshape(brisk_features, [-1]))
            resnet_f.append(feature_map.squeeze().numpy())

    imgs = np.asarray(imgs, np.float32)  # 列表轉為NumPy數組
    lab = np.asarray(lab, np.int32)      # 列表轉為NumPy數組

    sift_f = np.asarray(sift_f, np.float32)
    print("Shape of SIFT features array:", sift_f.shape)
    orb_f = np.asarray(orb_f, np.float32)
    print("Shape of ORB features array:", orb_f.shape)
    brisk_f = np.asarray(brisk_f, np.float32)
    print("Shape of BRISK features array:", brisk_f.shape)
    resnet_f = np.array(resnet_f)
    print("Shape of features array:", resnet_f.shape)

    return imgs, lab, sift_f, orb_f, brisk_f, resnet_f  # 輸出: 特徵矩陣(轉為一維)＆標註向量


# x, y = load_img('train.txt')
# tx, ty = load_img('test.txt')


# ======================================
# X就是資料，Y是Label，請設計不同分類器來得到最高效能
# 必須要計算出分類的正確率
# ======================================
