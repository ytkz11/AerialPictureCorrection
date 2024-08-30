# -*- coding: utf-8 -*-
#Import library
#import libraries
import sys
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
        d3[5:5 + x1, 5:5 + y1, i] = d1[:, :, i]

  d = []
  x3, y3, z3 = np.shape(d3)
  datagray = np.zeros(shape=(x3, y3, z3, 2))
  datagray[:, :, :, 0] = d3
  if x3 == x1:
    datagray[:, :, :, 1] = d1
  elif x3 == x2:
    datagray[:, :, :, 1] = d2

  return datagray
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1,
                                                                                            2)  # coordinates of a reference image
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1,
                                                                                       2)  # coordinates of second image

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)  # calculate the transformation matrix

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img
#folfer containing images from drones, sorted by name 
import glob
path = sorted(glob.glob("*.jpg"))

path = sorted(glob.glob(r"D:\无人机\test\*.png"))
img_list = []
for img in path:
    n = cv2.imread(img)
    img_list.append(n)
"""Functions for stitching"""
# Use ORB detector to extract keypoints

sift = cv2.xfeatures2d.SIFT_create()
orb = cv2.ORB_create(nfeatures=20000)
# orb = cv2.ORB_create()
while True:
    img1 = img_list.pop(0)
    img2 = img_list.pop(0)

    data2 = match_3d(img1, img2)
    img2 = np.array(data2[:, :, :, 0]).astype('uint8')
    img1 = np.array(data2[:, :, :, 1]).astype('uint8')
    # Find the key points and descriptors with ORB
    # keypoints1, descriptors1 = orb.detectAndCompute(img1,
    #                                                 None)  # descriptors are arrays of numbers that define the keypoints
    # keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子
    kp2, des2 = sift.detectAndCompute(img2, None)  # des是描述子

    # Create a BFMatcher object to match descriptors
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher_create(
        cv2.NORM_HAMMING)  # NORM_HAMMING specifies the distance as a measurement of similarity between two descriptors

    # Find matching points
    matches = bf.knnMatch(des1, des2, k=2)

    all_matches = []
    for m, n in matches:
        all_matches.append(m)
    # Finding the best matches
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:  # Threshold
            good.append([m]) # orb需要加[]，sift 不需要加[]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),

                       flags=2)
    img5 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good, None, **draw_params)
    cv2.imwrite( 'test.jpg', img5)

    # cv2.imshow("BFmatch", img5)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Set minimum match condition
    MIN_MATCH_COUNT = 5

    if len(good) > MIN_MATCH_COUNT:

        # Convert keypoints to an argument for findHomography
        # src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        src_pts = np.float32([keypoints1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Establish a homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        result = warpImages(img1, img2, M)

        img_list.insert(0, result)

        if len(img_list) == 1:
            break
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.show()