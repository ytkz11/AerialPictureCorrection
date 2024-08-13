#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2024/8/12 上午9:09 
# @File : test.py 
'''

'''
from overlay_image_areas import same_area
from overlay_image_sift import ransac

import cv2
import numpy as np
import PIL.Image as Image
import warp
# 定义函数用于拼接多张图片
def stitch_images(images):
    # 创建SIFT检测器对象
    sift = cv2.SIFT_create()

    # 初始化关键点列表和描述符列表
    keypoints_list = []
    descriptors_list = []

    # 对每张图片提取特征点和描述符
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # 用于保存所有匹配点对的列表
    all_matches = []
    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
    # 使用FLANN匹配器匹配特征点
    for i in range(len(images) - 1):
        matches = flann.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)

        # 选择最佳匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        good = []  # 较好的匹配
        pts1 = []  # img1中较好的匹配的坐标
        pts2 = []

        for j, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append(m)
                pts1.append(keypoints_list[i][m.queryIdx].pt)
                pts2.append(keypoints_list[i+1][m.trainIdx].pt)

        best_f, distances = ransac(pts1, pts2)
        matchesMask = [[0, 0] for j in range(len(good))]
        ransac_good = []
        for j, k in enumerate(distances):
            if k <= 1:
                matchesMask[j] = [1, 0]

                ransac_good.append(good[j])

        # 保存匹配点对
        # all_matches.append(good_matches)
        all_matches.append(ransac_good)
        # 保存前100对
        # good_matches = ransac_good[:100]
        # for m in good_matches:
        #     pt1 = keypoints_list[i][m.queryIdx].pt
        #     pt2 = keypoints_list[i + 1][m.trainIdx].pt
        #     pt1 = tuple(map(int, pt1))
        #     pt2 = tuple(map(int, pt2))
        #     cv2.circle(images[i], pt1, 10, (0, 255, 0), -1)
        #     cv2.circle(images[i + 1], pt2, 10, (0, 255, 0), -1)
        #
        # img_matches = cv2.drawMatches(images[i], keypoints_list[i], images[i + 1], keypoints_list[i + 1], good_matches,
        #                               None, matchColor=(255, 0, 0), matchesMask=None,
        #                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # cv2.imshow(f'Matches between image {i + 1} and image {i + 2}', img_matches)
        # cv2.waitKey(0)
    # 计算各个图像之间的透视变换
    result = images[0]
    for i in range(len(images) - 1):
        points1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in all_matches[i]]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in all_matches[i]]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)


        result = cv2.warpPerspective(result, H, (result.shape[1] + images[i + 1].shape[1], result.shape[0]))
        # 点也转换

        result[0:images[i + 1].shape[0], 0:images[i + 1].shape[1]] = images[i + 1]
    return result






if __name__ == '__main__':
    # 读取图片
    image1 = cv2.imread('hill1.JPG')
    image2 = cv2.imread('hill2.jpg')
    raster1 = r'D:\无人机\test\DJI_20230410091557_0118.tif'
    raster2 = r'D:\无人机\test\DJI_20230410091600_0119.tif'
    image1, image2, img1_overlap_coor, img2_overlap_coor = same_area(raster1, raster2)

    # 拼接图片
    result = stitch_images([image1, image2])
    # result = stitch_images([image1, image2, image3])

    # 显示拼接结果
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_rgb)
    img.save('test_tow_img_merge.PNG')