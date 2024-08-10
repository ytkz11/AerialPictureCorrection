#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: AutomaticPanoramicImageStitching.py
# time: 2024/8/10 2:27
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random, math
from overlay_image_areas import same_area
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

class Stitcher:
    def __init__(self):
        pass

    def stitch(self, imgs, blending_mode="linearBlending", ratio=0.75):
        '''
            The main method to stitch image
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        print("Left img size (", hl, "*", wl, ")")
        print("Right img size (", hr, "*", wr, ")")

        # Step1 - extract the keypoints and features by SIFT detector and descriptor
        print("Step1 - Extract the keypoints and features by SIFT detector and descriptor...")
        kps_l, features_l = self.detectAndDescribe(img_left)
        kps_r, features_r = self.detectAndDescribe(img_right)

        # Step2 - extract the match point with threshold (David Lowe’s ratio test)
        print("Step2 - Extract the match point with threshold (David Lowe’s ratio test)...")
        # matches_pos = self.matchKeyPoint(kps_l, kps_r, features_l, features_r, ratio)
        matches_pos_2,matches_pos_int_2, HomoMat = self.matchKeyPoint_cv2(kps_l, kps_r, features_l, features_r, ratio=0.7)
        matches_pos = matches_pos_int_2
        print("The number of matching points:", len(matches_pos))

        # Step2 - draw the img with matching point and their connection line
        self.drawMatches([img_left, img_right], matches_pos)

        # Step3 - fit the homography model with RANSAC algorithm
        print("Step3 - Fit the best homography model with RANSAC algorithm...")
        # HomoMat = self.fitHomoMat(matches_pos)

        # Step4 - Warp image to create panoramic image
        print("Step4 - Warp image to create panoramic image...")
        warp_img = self.warp([img_left, img_right], HomoMat, blending_mode)

        return warp_img

    def detectAndDescribe(self, img):
        '''
        The Detector and Descriptor
        '''
        # SIFT detector and descriptor
        sift = cv2.xfeatures2d.SIFT_create()
        kps, features = sift.detectAndCompute(img, None)

        return kps, features

    def matchKeyPoint(self, kps_l, kps_r, features_l, features_r, ratio):
        '''
            Match the Keypoints beteewn two image
        '''
        Match_idxAndDist = []  # min corresponding index, min distance, seccond min corresponding index, second min distance
        for i in range(len(features_l)):
            min_IdxDis = [-1, np.inf]  # record the min corresponding index, min distance
            secMin_IdxDis = [-1, np.inf]  # record the second corresponding min index, min distance
            for j in range(len(features_r)):
                dist = np.linalg.norm(features_l[i] - features_r[j])
                if (min_IdxDis[1] > dist):
                    secMin_IdxDis = np.copy(min_IdxDis)
                    min_IdxDis = [j, dist]
                elif (secMin_IdxDis[1] > dist and secMin_IdxDis[1] != min_IdxDis[1]):
                    secMin_IdxDis = [j, dist]

            Match_idxAndDist.append([min_IdxDis[0], min_IdxDis[1], secMin_IdxDis[0], secMin_IdxDis[1]])

        # ratio test as per Lowe's paper
        goodMatches = []
        for i in range(len(Match_idxAndDist)):
            if (Match_idxAndDist[i][1] <= Match_idxAndDist[i][3] * ratio):
                goodMatches.append((i, Match_idxAndDist[i][0]))

        goodMatches_pos = []
        for (idx, correspondingIdx) in goodMatches:
            psA = (int(kps_l[idx].pt[0]), int(kps_l[idx].pt[1]))
            psB = (int(kps_r[correspondingIdx].pt[0]), int(kps_r[correspondingIdx].pt[1]))
            goodMatches_pos.append([psA, psB])

        return goodMatches_pos

    def matchKeyPoint_cv2(self, kps_l, kps_r, features_l, features_r, ratio):
        '''
            Match the Keypoints beteewn two image
        '''
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(features_l, features_r, k=2)

        good = []  # 较好的匹配
        pts1 = []  # img1中较好的匹配的坐标
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < ratio * n.distance:
                good.append(m)
                pts1.append(kps_l[m.queryIdx].pt)
                pts2.append(kps_r[m.trainIdx].pt)

                # 进度条

            print("\rCompare with the second point: [{0:50s}] {1:.1f}%".format('#' * int(i / len(matches) * 50),
                                                                i / len(matches) * 100), end="",
                      flush=True)
        print("/n")
        best_f, distances = ransac(pts1, pts2)
        matchesMask = [[0, 0] for i in range(len(good))]
        ransac_good = []
        for i, k in enumerate(distances):
            if k <= 1:
                matchesMask[i] = [1, 0]

                ransac_good.append([good[i]])
            print("\rRansac filtration: [{0:50s}] {1:.1f}%".format('#' * int(i / len(distances) * 50),
                                                                i / len(distances) * 100), end="",
                      flush=True)
        print("/n")
        src_pts = np.float32([kps_l[m[0].queryIdx].pt for m in ransac_good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps_r[m[0].trainIdx].pt for m in ransac_good]).reshape(-1, 1, 2)

        # Establish a homography
        M, _ = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, 5.0)

        # ransac 后的 匹配点
        ransac_good_second = []
        temp = _.tolist()
        for i, j in enumerate(temp):
            if j[0] == 1:
                ransac_good_second.append(ransac_good[i])

                print("\rHomography filtration: [{0:50s}] {1:.1f}%".format('#' * int(i / len(temp) * 50),
                                                                i / len(temp) * 100), end="",
                      flush=True)
        print("/n")
        # print(ransac_good_second)

        # ransac_src_pts = [kps_l[m[0].queryIdx].pt for m in ransac_good_second]
        # ransac_dst_pts = [kps_r[m[0].trainIdx].pt for m in ransac_good_second]

        goodMatches_pos_by_cv2 = [[kps_l[m[0].queryIdx].pt, kps_r[m[0].trainIdx].pt] for m in ransac_good_second]

        goodMatches_pos_by_cv2_int = []
        for i in goodMatches_pos_by_cv2:
            goodMatches_pos_by_cv2_int.append([(int(i[0][0]), int(i[0][1])), (int(i[1][0]) , int(i[1][1]))])
        return goodMatches_pos_by_cv2, goodMatches_pos_by_cv2_int, M

        # Sort the matches by distance
    def drawMatches(self, imgs, matches_pos):
        '''
            Draw the match points img with keypoints and connection line
        '''

        # initialize the output visualization image
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        vis = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
        vis[0:hl, 0:wl] = img_left
        vis[0:hr, wl:] = img_right

        # Draw the match
        for (img_left_pos, img_right_pos) in matches_pos:
            pos_l = img_left_pos
            pos_r = img_right_pos[0] + wl, img_right_pos[1]
            cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)
            cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)
            cv2.line(vis, pos_l, pos_r, (255, 0, 0), 1)

        # return the visualization
        plt.figure(4)
        plt.title("img with matching points")
        plt.imshow(vis[:, :, ::-1])
        cv2.imwrite("matching.jpg", vis)

        return vis

    def fitHomoMat(self, matches_pos):
        '''
            Fit the best homography model with RANSAC algorithm - noBlending、linearBlending、linearBlendingWithConstant
        '''
        dstPoints = []  # i.e. left image(destination image)
        srcPoints = []  # i.e. right image(source image)
        for dstPoint, srcPoint in matches_pos:
            dstPoints.append(list(dstPoint))
            srcPoints.append(list(srcPoint))
        dstPoints = np.array(dstPoints)
        srcPoints = np.array(srcPoints)

        homography = Homography()

        # RANSAC algorithm, selecting the best fit homography
        NumSample = len(matches_pos)
        threshold = 5.0
        NumIter = 8000
        NumRamdomSubSample = 4
        MaxInlier = 0
        Best_H = None

        for run in range(NumIter):
            SubSampleIdx = random.sample(range(NumSample), NumRamdomSubSample)  # get the Index of ramdom sampling
            H = homography.solve_homography(srcPoints[SubSampleIdx], dstPoints[SubSampleIdx])

            # find the best Homography have the the maximum number of inlier
            NumInlier = 0
            for i in range(NumSample):
                if i not in SubSampleIdx:
                    concateCoor = np.hstack((srcPoints[i], [1]))  # add z-axis as 1
                    dstCoor = H @ concateCoor.T  # calculate the coordination after transform to destination img
                    if dstCoor[2] <= 1e-8:  # avoid divide zero number, or too small number cause overflow
                        continue
                    dstCoor = dstCoor / dstCoor[2]
                    if (np.linalg.norm(dstCoor[:2] - dstPoints[i]) < threshold):
                        NumInlier = NumInlier + 1
            if (MaxInlier < NumInlier):
                MaxInlier = NumInlier
                Best_H = H

        print("The Number of Maximum Inlier:", MaxInlier)

        return Best_H

    def warp(self, imgs, HomoMat, blending_mode):
        '''
           Warp image to create panoramic image
           There are three different blending method - noBlending、linearBlending、linearBlendingWithConstant
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        stitch_img = np.zeros((max(hl, hr), wl + wr, 3),
                              dtype="int")  # create the (stitch)big image accroding the imgs height and width

        if (blending_mode == "noBlending"):
            stitch_img[:hl, :wl] = img_left

        # Transform Right image(the coordination of right image) to destination iamge(the coordination of left image) with HomoMat
        inv_H = np.linalg.inv(HomoMat)
        for i in range(stitch_img.shape[0]):
            for j in range(stitch_img.shape[1]):
                coor = np.array([j, i, 1])
                img_right_coor = inv_H @ coor  # the coordination of right image
                img_right_coor /= img_right_coor[2]

                # you can try like nearest neighbors or interpolation
                y, x = int(round(img_right_coor[0])), int(round(img_right_coor[1]))  # y for width, x for height

                # if the computed coordination not in the (hegiht, width) of right image, it's not need to be process
                if (x < 0 or x >= hr or y < 0 or y >= wr):
                    continue
                # else we need the tranform for this pixel
                stitch_img[i, j] = img_right[x, y]

        # create the Blender object to blending the image
        blender = Blender()
        if (blending_mode == "linearBlending"):
            stitch_img = blender.linearBlending([img_left, stitch_img])
        elif (blending_mode == "linearBlendingWithConstant"):
            stitch_img = blender.linearBlendingWithConstantWidth([img_left, stitch_img])

        # remove the black border
        stitch_img = self.removeBlackBorder(stitch_img)

        return stitch_img
    def warp2(self, imgs, HomoMat, blending_mode):
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]

        # Create the stitched image according to the images' height and width
        stitch_img = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")

        if blending_mode == "noBlending":
            stitch_img[:hl, :wl] = img_left

        # Invert the homography matrix to transform coordinates from the stitched image back to the right image
        inv_H = np.linalg.inv(HomoMat)

        # Generate coordinate grid for the stitched image
        X, Y = np.meshgrid(np.arange(stitch_img.shape[1]), np.arange(stitch_img.shape[0]))
        coors = np.vstack((X.flatten(), Y.flatten(), np.ones_like(X.flatten()))).T

        # Transform coordinates
        transformed_coors = inv_H @ coors.T
        transformed_coors /= transformed_coors[2]
        transformed_coors = transformed_coors[:2].T

        # Extract valid coordinates
        valid_indices = (transformed_coors[:, 1] >= 0) & (transformed_coors[:, 1] < hr) & (
                    transformed_coors[:, 0] >= 0) & (transformed_coors[:, 0] < wr)
        valid_coors = transformed_coors[valid_indices].round().astype(int)

        # Ensure that the coordinates are within the boundaries of the image
        valid_coors[:, 1] = np.clip(valid_coors[:, 1], 0, hr - 1)
        valid_coors[:, 0] = np.clip(valid_coors[:, 0], 0, wr - 1)

        # Map valid coordinates back to original grid
        valid_indices_grid = np.unravel_index(np.where(valid_indices), (stitch_img.shape[0], stitch_img.shape[1]))

        # Map pixels from the right image to the stitched image
        stitch_img[valid_indices_grid] = img_right[valid_coors[:, 1], valid_coors[:, 0]]
        blender = Blender()
        if (blending_mode == "linearBlending"):
            stitch_img = blender.linearBlending([img_left, stitch_img])
        elif (blending_mode == "linearBlendingWithConstant"):
            stitch_img = blender.linearBlendingWithConstantWidth([img_left, stitch_img])

        # remove the black border
        stitch_img = self.removeBlackBorder(stitch_img)
        return stitch_img
    def removeBlackBorder(self, img):
        '''
        Remove img's the black border
        '''
        h, w = img.shape[:2]
        reduced_h, reduced_w = h, w
        # right to left
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if (np.count_nonzero(img[i, col]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_w = reduced_w - 1

        # bottom to top
        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(reduced_w):
                if (np.count_nonzero(img[row, i]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_h = reduced_h - 1

        return img[:reduced_h, :reduced_w]


class Blender:
    def linearBlending(self, imgs):
        '''
        linear Blending(also known as Feathering)
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")

        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1

        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1

        # Plot the overlap mask
        plt.figure(21)
        plt.title("overlap_mask")
        plt.imshow(overlap_mask.astype(int), cmap="gray")

        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr))  # alpha value depend on left image
        for i in range(hr):
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j

            if (minIdx == maxIdx):  # represent this row's pixels are all zero, or only one pixel not zero
                continue

            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))

        linearBlending_img = np.copy(img_right)
        linearBlending_img[:hl, :wl] = np.copy(img_left)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[
                        i, j]

        return linearBlending_img

    def linearBlendingWithConstantWidth(self, imgs):
        '''
        linear Blending with Constat Width, avoiding ghost region
        # you need to determine the size of constant with
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        constant_width = 3  # constant width

        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1

        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1

        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr))  # alpha value depend on left image
        for i in range(hr):
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j

            if (minIdx == maxIdx):  # represent this row's pixels are all zero, or only one pixel not zero
                continue

            decrease_step = 1 / (maxIdx - minIdx)

            # Find the middle line of overlapping regions, and only do linear blending to those regions very close to the middle line.
            middleIdx = int((maxIdx + minIdx) / 2)

            # left
            for j in range(minIdx, middleIdx + 1):
                if (j >= middleIdx - constant_width):
                    alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                else:
                    alpha_mask[i, j] = 1
            # right
            for j in range(middleIdx + 1, maxIdx + 1):
                if (j <= middleIdx + constant_width):
                    alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                else:
                    alpha_mask[i, j] = 0

        linearBlendingWithConstantWidth_img = np.copy(img_right)
        linearBlendingWithConstantWidth_img[:hl, :wl] = np.copy(img_left)
        # linear blending with constant width
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (
                                1 - alpha_mask[i, j]) * img_right[i, j]

        return linearBlendingWithConstantWidth_img


class Homography:
    def solve_homography(self, P, m):
        """
        Solve homography matrix

        Args:
            P:  Coordinates of the points in the original plane,
            m:  Coordinates of the points in the target plane


        Returns:
            H: Homography matrix
        """
        try:
            A = []
            for r in range(len(P)):
                # print(m[r, 0])
                A.append([-P[r, 0], -P[r, 1], -1, 0, 0, 0, P[r, 0] * m[r, 0], P[r, 1] * m[r, 0], m[r, 0]])
                A.append([0, 0, 0, -P[r, 0], -P[r, 1], -1, P[r, 0] * m[r, 1], P[r, 1] * m[r, 1], m[r, 1]])

            u, s, vt = np.linalg.svd(A)  # Solve s ystem of linear equations Ah = 0 using SVD
            # pick H from last line of vt
            H = np.reshape(vt[8], (3, 3))
            # normalization, let H[2,2] equals to 1
            H = (1 / H.item(8)) * H
        except:
            print("Error occur!")

        return H


if __name__ == "__main__":

        raster1 = r'h:\无人机\test\DJI_20230410091600_0119.tif'
        raster2 = r'h:\无人机\test\DJI_20230410091557_0118.tif'
        img1, img2, img1_overlap_coor, img2_overlap_coor = same_area(raster1, raster2)


        # img1 = cv2.imread(r'hill1.JPG')
        # img2 = cv2.imread(r'hill2.JPG')
        # The stitch object to stitch the image
        blending_mode = "noBlending"  # three mode - noBlending、linearBlending、linearBlendingWithConstant
        stitcher = Stitcher()
        warp_img = stitcher.stitch([img1, img2], blending_mode)

        # plot the stitched image
        plt.figure(13)
        plt.title("warp_img")
        plt.imshow(warp_img[:, :, ::-1].astype(int)),plt.show()

        # save the stitched iamge
        saveFilePath = "u4.jpg"
        cv2.imwrite(saveFilePath, warp_img)


