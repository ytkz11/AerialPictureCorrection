#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2024/8/7 上午11:28 
# @File : overlay_image_sift.py 
'''

'''
from overlay_image_areas import same_area
import cv2
import numpy as np
import matplotlib.pyplot as plt
def match_3d(d1, d2):
  '''
  使两个矩阵尺寸相等
  :param d1:
  :param d2:
  :return:
  '''
  x1, y1, z1 = np.shape(d1)
  x2, y2, z2 = np.shape(d2)

  if x1 > x2 and y1 > y2:
    d3 = np.zeros((x1, y1, 3))
    for i in range(z1):
      print(i)
      d3[5:5 + x2, 5:5 + y2, i] = d2[:, :, i]
  else:

    if y1 < y2:
      d3 = np.zeros((x2, y2, 3))

      for i in range(z1):
        d3[x2-x1:x2-x1 + x1, y2-y1:y2-y1 + y1, i] = d1[:, :, i]

  d = []
  x3, y3, z3 = np.shape(d3)
  datagray = np.zeros(shape=(x3, y3, z3, 2))
  datagray[:, :, :, 0] = d3
  if x3 == x1:
    datagray[:, :, :, 1] = d1
  elif x3 == x2:
    datagray[:, :, :, 1] = d2

  return datagray
def sift(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子
    kp2, des2 = sift.detectAndCompute(img2, None)  # des是描述子
    # Create a BFMatcher object to match descriptors
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher_create(
        cv2.NORM_HAMMING)  # NORM_HAMMING specifies the distance as a measurement of similarity between two descriptors

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    plt.imshow(img3, ), plt.show()
    cv2.imwrite( 'overlay_image_sift_test.jpg', img3)


if __name__ == '__main__':
    # 加载两个图像
    raster1 = r'D:\无人机\test\DJI_20230410091557_0118.tif'
    raster2 = r'D:\无人机\test\DJI_20230410091600_0119.tif'
    img1,img2 = same_area(raster1, raster2)
    data2 = match_3d(img1, img2)
    cv2.imwrite('overlay_image_part_image.jpg', data2)

    sift(img1,img2)

