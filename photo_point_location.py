#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : photo_point_location.py 
'''

'''
from AerialCorrection import AerialCorrection, get_file_names
import matplotlib.pyplot as plt
import os
class PhotoPointLocation():
    def __init__(self,img_path):
        self.img_path = get_file_names(img_path, ['.JPG'])
    def get_centre_point_info(self):
        centre_point_info_list = []

        for img in self.img_path:

            AC = AerialCorrection(img)
            latitude1, longitude1 = AC.get_gps_info()
            centre_point_info_list.append([longitude1, latitude1, img])
        return centre_point_info_list

    def show_point(self):
        centre_point_info_list = self.get_centre_point_info()
        # 提取经度和纬度值
        longitudes = [point[0] for point in centre_point_info_list]
        latitudes = [point[1] for point in centre_point_info_list]


        # for i in centre_point_info_list:
        #     with open(r'temp.txt','a') as f:
        #         f.write(str(i[0])+' ')
        #         f.write(str(i[1])+' ')
        #         f.write(str(i[2])+' ')
        #         f.write('\n')

        # 创建图形
        plt.figure(figsize=(8, 6))

        # 绘制点
        plt.scatter(longitudes, latitudes, color='red', marker='o')

        # # 设置坐标轴标签
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')

        # 设置标题
        plt.title('LAT LON ')

        # # 设置坐标轴范围
        # plt.xlim(min(longitudes) - 1, max(longitudes) + 1)
        # plt.ylim(min(latitudes) - 1, max(latitudes) + 1)
        # 设置坐标轴格式为小数
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_yaxis().get_major_formatter()._useMathText = False
        # 显示网格
        # plt.grid(True)

        # 显示图形
        plt.show()


if __name__ == '__main__':
    A = PhotoPointLocation(r'D:\无人机\1')
    a = A.show_point()
    print()
