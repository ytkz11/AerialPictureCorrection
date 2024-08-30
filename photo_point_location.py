#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : photo_point_location.py 
'''

'''
from AerialCorrection import AerialCorrection, get_file_names
import matplotlib.pyplot as plt
from create_point_gpkg import create_point_gpkg
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

        # 创建图形
        plt.figure(figsize=(8, 6))

        # 绘制点
        plt.scatter(longitudes, latitudes, color='red', marker='o')

        # 设置标题
        plt.title('LAT LON ')

        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_yaxis().get_major_formatter()._useMathText = False

        # 显示图形
        plt.show()
        self.save_img_point_to_gpkg()
    def save_img_point_to_gpkg(self):
        centre_point_info_list = self.get_centre_point_info()
        lon_lat = [(i[0], i[1]) for i in centre_point_info_list]
        create_point_gpkg(lon_lat, filename=r'd:/temp/centre_point.gpkg')

        print()
    def image_position_relationships(self):
        # save as gpkg file
        centre_point_info_list = self.get_centre_point_info()

        print()
if __name__ == '__main__':
    A = PhotoPointLocation(r'D:\无人机')
    a = A.show_point()

    print()