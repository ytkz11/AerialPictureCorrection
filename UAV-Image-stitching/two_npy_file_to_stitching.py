#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2024/8/30 下午1:22 
# @File : two_npy_file_to_stitching.py 
'''

'''
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
class two_npy_file_to_stitching:
    def __init__(self, right_img_file, left_img_file, config_file):
        self.right_img_file = right_img_file
        self.left_img_file = left_img_file
        self.config_file = config_file
        self.cover_part_img_info = self.read_config_file()

        self.read_npy()
        self.warp()
    def read_npy(self) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 读取npy
        self.right_img = np.load(self.right_img_file)
        self.left_img = np.load(self.left_img_file)


        print(self.right_img.shape)
        print(self.left_img.shape)
    def warp(self):
        width = self.right_img.shape[1] + self.left_img.shape[1]
        height = self.right_img.shape[0] + self.left_img.shape[0]

        H = np.array(self.cover_part_img_info["Homography"])
        result = cv2.warpPerspective(self.right_img, H, (width, height))
        result[:self.left_img.shape[0], :self.left_img.shape[1]] = self.left_img
        plt.imshow(result)
        plt.show()
        print()
    def read_config_file(self):
        with open(self.config_file, 'r') as f:
            fcc_data = json.load(f)
            print(fcc_data)
        return fcc_data

if __name__ == '__main__':

    right_img_file = r'D:\temp\aerial_data\右图的右边.npy'
    left_img_file = r'D:\temp\aerial_data\左图的左边.npy'
    config_file = r'D:\temp\aerial_data\DJI_20230410091600_0119_DJI_20230410091557_0118.json'
    A = two_npy_file_to_stitching(right_img_file, left_img_file, config_file)

    print()
