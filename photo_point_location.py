#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : photo_point_location.py 
'''

'''
from AerialCorrection import AerialCorrection, get_file_names
import matplotlib.pyplot as plt
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

        # 显示网格
        # plt.grid(True)

        # 显示图形
        plt.show()


if __name__ == '__main__':
    A = PhotoPointLocation(r'D:\无人机\1')
    a = A.show_point()
    print()
