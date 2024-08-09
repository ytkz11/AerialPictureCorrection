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
import random
import math
import PIL.Image as Image
from skimage import transform
def ransac(pts1, pts2):
    best_inlinenums = 0
    best_f = np.zeros([3, 3])
    best_distance = []
    i = 0
    while i < 25:
        # 随机选择8个点
        index = set()
        while len(index) < 8:
            index.add(random.randrange(len(pts1)))
        # 根据这8个点生成矩阵
        a = np.zeros([8, 9])
        for j, item in enumerate(index):
            (x1, y1) = pts1[item]
            (x2, y2) = pts2[item]
            a[j][0] = x1 * x2
            a[j][1] = x2 * y1
            a[j][2] = x2
            a[j][3] = x1 * y2
            a[j][4] = y1 * y2
            a[j][5] = y2
            a[j][6] = x1
            a[j][7] = y1
            a[j][8] = 1
        u, d, vt = np.linalg.svd(a)
        f = vt[8]
        f = f.reshape(3, 3)
        # 根据F计算内点数，首先计算极线
        one = np.ones(len(pts1))
        pts1_new = np.insert(pts1, 2, values=one, axis=1)  # 构造齐次坐标系
        elines = [np.dot(f, pts1_new[i]) for i in range(len(pts1))]  # 极线
        # 计算pts2中每一个点到对应极线的距离
        pts2_new = np.insert(pts2, 2, values=one, axis=1)  # 构造齐次坐标系
        distances = []
        inline_num = 0
        for pt, l in zip(pts2_new, elines):
            div = abs(np.dot(np.transpose(pt), l))
            dived = math.sqrt(l[0] * l[0] + l[1] * l[1])
            d = div / dived
            if d <= 0.25:
                # if d <= 3:

                inline_num = inline_num + 1
            distances.append(d)
        if inline_num > best_inlinenums:
            best_f = f[:]
            best_inlinenums = inline_num
            best_distance = distances[:]
        i += 1
    return best_f, best_distance


def match_3d( d1, d2):
    x1, y1, z1 = np.shape(d1)
    x2, y2, z2 = np.shape(d2)

    if x1 > x2 and y1 > y2:
        d3 = np.zeros((x1, y1, 3)).astype('uint8')
        for i in range(z1):
            print(i)
            d3[1:1 + x2, 1:1 + y2, i] = d2[:, :, i]
    else:

        if y1 < y2:
            d3 = np.zeros((x2, y2, 3))

            for i in range(z1):
                d3[1:1 + x1, 1:1 + y1, i] = d1[:, :, i]

    d = []
    x3, y3, z3 = np.shape(d3)
    datagray = np.zeros(shape=(x3, y3 * 2, z3)).astype('uint8')
    datagray[:, 0:y3, :] = d3
    if x3 == x1 and y3 == y1:
        datagray[:, y3:y3 * 2, :] = d1
    elif x3 == x2 and y3 == y2:
        datagray[:, y3:y3 * 2, :] = d2

    datagray[datagray == 0] = 255

    return np.array(datagray).astype('uint8')
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

    good = []  # 较好的匹配
    pts1 = []  # img1中较好的匹配的坐标
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    best_f, distances = ransac(pts1, pts2)
    matchesMask = [[0, 0] for i in range(len(good))]
    ransac_good = []
    for i, k in enumerate(distances):
        if k <= 1:
            matchesMask[i] = [1, 0]

            ransac_good.append([good[i]])




    # ratio test as per Lowe's paper
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.6 * n.distance:
    #         matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       # singlePointColor=(255, 0, 0),
                       # matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, ransac_good, None, **draw_params)

    # plt.imshow(img3, ), plt.show()
    cv2.imwrite( 'overlay_image_sift_test2.jpg', img3)
    merge_image(img1,img2,kp1,kp2,ransac_good)
    print()


def apply_perspective_transform(points, M):
    # 将点转换为齐次坐标
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))

    # 应用变换
    transformed_points_homogeneous = np.dot(M, points_homogeneous.T).T

    # 归一化齐次坐标
    transformed_points = (transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2][:, None]).astype(
        np.int32)

    return transformed_points



def warpImages(img1, img2, M):

    h,w,z = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    # 检查变换后的坐标
    print("Transformed coordinates dst:")
    print(dst)

    # 确保变换后的坐标合理
    if np.any(dst < 0) or np.any(dst > np.array([img2.shape[1], img2.shape[0]])):
        print("Some points are outside the image boundaries.")
    else:
        print("All points are within the image boundaries.")

    # 如果有必要，扩展目标图像
    min_x = int(np.min(dst[:, :, 0]))
    max_x = int(np.max(dst[:, :, 0]))
    min_y = int(np.min(dst[:, :, 1]))
    max_y = int(np.max(dst[:, :, 1]))

    if min_x < 0 or min_y < 0 or max_x > img2.shape[1] or max_y > img2.shape[0]:
        # 计算新的宽度和高度

        # 将原图像粘贴到新图像中
        dx = abs(min_x) if min_x < 0 else 0
        dy = abs(min_y) if min_y < 0 else 0
        new_width = max(max_x, abs(min_x)) + dx+int(dx*0.5)
        new_height = max(max_y, abs(min_y)) + dy

        # 创建一个新的空白图像
        new_img = np.zeros((new_height, new_width+dx, 3), dtype=np.uint8)

        new_img[dy:dy + img2.shape[0], dx:dx + img2.shape[1]] = img2


        # 更新目标图像
        img3 = new_img


        # 更新变换后的坐标
        dst[:, :, 0] += dx+int(dx*0.5)
        dst[:, :, 1] += dy

        # 绘制多边形
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

        # 显示结果图像

        # plt.imshow(img3, ), plt.show()

    # 将 img1 按照变换矩阵 M 变换到 img2 的空间
    warped_img1 = cv2.warpPerspective(img1, M, (img3.shape[1], img3.shape[0]))

    temp_arr = warped_img1
    temp_arr[temp_arr== 0] = img3[temp_arr== 0]
    # img = Image.fromarray(temp_arr)
    # img.save('merge.PNG')
    return warped_img1, img3
    # 将 img1 和 img2 拼接在一起
    # result = cv2.addWeighted(img3, 1, warped_img1, 1, 0)
    # plt.imshow(result, ), plt.show()
    # img = Image.fromarray(result)
    # img.save('merge_result.PNG')
def merge_image(img1,img2,kp1,kp2,good):

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)



    # ransac 后的 匹配点
    ransac_good = []
    for i,j  in enumerate (_.tolist() ):
        if j[0] ==1:
            ransac_good.append(good[i])
    print(ransac_good)

    ransac_src_pts = [kp1[m[0].queryIdx].pt for m in ransac_good]
    ransac_dst_pts = [kp2[m[0].trainIdx].pt for m in ransac_good]

    # 根据匹配点判断是左图还是右图
    if ransac_src_pts[0][0] < ransac_dst_pts[0][0]:
        # 左图
        print('img2是左图')
        left_img = img2
        right_img = img1
    else:
        # 右图
        left_img = img1
        right_img = img2

        # 把 ransac_dst_pts点 显示在img2上
    hl, wl = left_img.shape[:2]

    hr, wr = right_img.shape[:2]
    stitch_img = np.zeros((max(hl, hr), wl + wr, 3),
                          dtype="int")  # create the (stitch)big image accroding the imgs height and width
    stitch_img[:hl, :wl] = left_img
    warped_right_img= cv2.warpPerspective(right_img, M, (stitch_img.shape[1], stitch_img.shape[0]))


    # for pt in ransac_dst_pts:
    #     cv2.circle(img2, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    #     cv2.imwrite('img2_ransac_good.jpg', img2)

    # 把 ransac_src_pts点 显示在img1上
    img1= np.ascontiguousarray(img1)
    for pt in ransac_src_pts:
        cv2.circle(img1, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    cv2.imwrite('img1_ransac_good.jpg', img1)


    transformed_pts1 = apply_perspective_transform(src_pts, M)
    transformed_pts2 = apply_perspective_transform(dst_pts, M)

    # 检查变换后的坐标
    print("Transformed coordinates transformed_pts1:")

    result = warpImages(img1, img2, M)
    img = Image.fromarray(result)
    img.save('merge.PNG')

    # plt.imshow(result),plt.show()
if __name__ == '__main__':
    # 加载两个图像
    raster1 = r'h:\无人机\test\DJI_20230410091557_0118.tif'
    raster2 = r'h:\无人机\test\DJI_20230410091600_0119.tif'
    img1, img2, img1_overlap_coor, img2_overlap_coor = same_area(raster1, raster2)
    sift(img1, img2)


    data2 = match_3d(img1, img2)
    data2 = np.array(data2).astype('uint8')
    img = Image.fromarray(data2)
    img.save('overlay_image_part_image.PNG')
    img = Image.fromarray(img2)
    img.save('img2.png')


