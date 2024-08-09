#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : overlay_image_areas.py 
"""
两景影像的重叠区域
"""
from osgeo import gdal, osr
import matplotlib.pyplot as plt
import numpy as np
def get_raster_extent(raster_path):
    """
    获取栅格的地理范围
    """
    dataset = gdal.Open(raster_path)
    if not dataset:
        raise FileNotFoundError(f"Raster file {raster_path} not found or unable to open.")

    transform = dataset.GetGeoTransform()
    minx = transform[0]
    maxy = transform[3]
    maxx = minx + transform[1] * dataset.RasterXSize
    miny = maxy + transform[5] * dataset.RasterYSize

    return (minx, maxy, maxx, miny)

def calculate_overlap(extent1, extent2):
    """
    计算两个地理范围的重叠部分
    """
    minx1, maxy1, maxx1, miny1 = extent1
    minx2, maxy2, maxx2, miny2 = extent2

    overlap_minx = max(minx1, minx2)
    overlap_maxy = min(maxy1, maxy2)
    overlap_maxx = min(maxx1, maxx2)
    overlap_miny = max(miny1, miny2)

    if overlap_maxx > overlap_minx and overlap_miny < overlap_maxy:
        return (overlap_minx, overlap_maxy, overlap_maxx, overlap_miny)
    else:
        return None  # 没有重叠

def geo_to_pixel(geo_transform, x, y):
    """
    将地理坐标转换为像素坐标
    """

    col = int((x-geo_transform[0]) / geo_transform[1])
    row = int((geo_transform[3]-y) / abs(geo_transform[5]))

    return (row, col)

def extended_range(top_left, bottom_right,ds ):
    '''
    Extended overlay range
    :return:
    '''
    x = ds.RasterXSize
    y = ds.RasterYSize

    extended_top_left = [0,0]
    extended_bottom_right = [0,0]
    extended_top_left[0]  = top_left[0] if  top_left[0] - 500< 0 else top_left[0] - 500
    extended_top_left[1] = top_left[1] if top_left[1] - 500 < 0 else top_left[1] -500

    extended_bottom_right[0] = bottom_right[0] if bottom_right[0] +500 > x else bottom_right[0] +500
    extended_bottom_right[1] = bottom_right[1] if bottom_right[1] + 500 > y else bottom_right[1] + 500

    return extended_top_left, extended_bottom_right
def same_area(raster1, raster2):
    extent1 = get_raster_extent(raster1)
    extent2 = get_raster_extent(raster2)

    overlap = calculate_overlap(extent1, extent2)

    if overlap:
        print(f"重叠的地理范围为: {overlap}")

        # 获取栅格文件的地理变换矩阵
        ds1 = gdal.Open(raster1)
        ds2 = gdal.Open(raster2)
        gt1 = ds1.GetGeoTransform()
        gt2 = ds2.GetGeoTransform()

        # 转换重叠区域的地理坐标到像素坐标
        top_left = geo_to_pixel(gt1, overlap[0], overlap[1])
        bottom_right = geo_to_pixel(gt1, overlap[2], overlap[3])
        extended_top_left, extended_bottom_right = extended_range(top_left, bottom_right, ds1)  # 扩张范围
        # extended_top_left, extended_bottom_right = top_left, bottom_right  # 不扩张范围
        print(f"在第一个图像中的重叠区域的像素坐标为: Top Left: {top_left}, Bottom Right: {bottom_right}")

        top_left2 = geo_to_pixel(gt2, overlap[0], overlap[1])
        bottom_right2 = geo_to_pixel(gt2, overlap[2], overlap[3])
        extended_top_left2, extended_bottom_right2 = extended_range(top_left2, bottom_right2, ds2)

        data1 = ds1.ReadAsArray()
        data2 =  ds2.ReadAsArray()

        data1 = np.transpose(data1, (1, 2, 0))
        data2 = np.transpose(data2, (1, 2, 0))

        data11 = data1[extended_top_left[0]:extended_bottom_right[0],extended_top_left[1]:extended_bottom_right[1],:]
        data22 = data2[extended_top_left2[0]:extended_bottom_right2[0], extended_top_left2[1]:extended_bottom_right2[1],:]

        # plt.imshow(data11), plt.show()
        # plt.imshow(data22), plt.show()
        print(f"在第二个图像中的重叠区域的像素坐标为: Top Left: {top_left2}, Bottom Right: {bottom_right2}")

        return data11, data22
    else:
        print("两景影像没有重叠。")
        return None
if __name__ == '__main__':
    # 加载两个图像
    raster1 = r'D:\无人机\test\DJI_20230410091557_0118.tif'
    raster2 = r'D:\无人机\test\DJI_20230410091600_0119.tif'
    same_area(raster1, raster2)
