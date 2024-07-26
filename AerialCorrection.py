#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: AerialCorrection.py
# time: 2024/5/12 16:49

# 航片校正

import os
import shutil

try:
    import Image
    import ImageDraw
except:
    from PIL import Image
    from PIL.ExifTags import TAGS
import math
from osgeo import  osr
import glob
import cv2
from zipfile import ZipFile
from osgeo import gdal
import codecs
import numpy as np
import PIL.Image as Image


class tif2kmz:
   def __init__(self, tif_file):
       self.tif_file = tif_file
       self.kmz_file = os.path.splitext(self.tif_file)[0] + '.kmz'
       self.kml_file = os.path.splitext(self.tif_file)[0] + '.kml'
       self.png_file = os.path.splitext(self.tif_file)[0] + '.png'

   def get_arrtibute(self):
       # 获取tif文件的属性, 用于生成kml文件
       ds = gdal.Open(self.tif_file, gdal.GA_ReadOnly)
       gt = ds.GetGeoTransform()
       self.east_longitude = gt[0]
       self.west_longitude = self.east_longitude + (ds.RasterXSize * gt[1])
       self.north_latitude = gt[3]
       self.south_latitude = self.north_latitude + (ds.RasterYSize * gt[5])

   def create_temp_png(self):
       ds = gdal.Open(self.tif_file, gdal.GA_ReadOnly)
       png_file = 'overlay.png'
       data = ds.ReadAsArray()
       z,x,y = data.shape
       temp_arr = np.zeros(shape=(x, y, 4))
       for i in range(3):
           temp_arr[:, :, i] = data[i, :, :]

       temp_arr[:, :, 3][data[0, :, :] != 0] = 255  # 透明度.
       temp_arr = temp_arr.astype(np.uint8)
       img = Image.fromarray(temp_arr)
       img.save(png_file)

       shutil.copy(png_file,self.png_file)


   def create_kml(self):
       self.get_arrtibute()
       overlay_name = "KML overlay"
       kml = (
                 '<?xml version="1.0" encoding="UTF-8"?>\n'
                 '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
                 ' <Folder>\n'
                 '   <name>Ground Overlays</name>\n'
                 '   <description>Examples of ground overlays</description>\n'
                 '   <GroundOverlay>\n'
                 '     <name>%s</name>\n'
                 '     <Icon>\n'
                 '       <href>overlay.png</href>\n'
                 '     </Icon>\n'
                 '     <LatLonBox>\n'
                 '       <north>%f</north>\n'
                 '       <south>%f</south>\n'
                 '       <east>%f</east>\n'
                 '       <west>%f</west>\n'
                 '     </LatLonBox>\n'
                 '   </GroundOverlay>\n'
                 ' </Folder>\n'
                 '</kml>\n'
            ) % (overlay_name, self.north_latitude, self.south_latitude, self.east_longitude, self.west_longitude)

       with codecs.open('overlay.kml', encoding='utf-8', mode='w+') as kmlFile:
           kmlFile.write(kml)

   def create_kmz(self):
       self.create_temp_png()
       self.create_kml()
       with ZipFile(self.kmz_file, 'w') as zipObj:
           # zipObj.writestr(kml_output_filename, kml) # Add doc.kml entry
           zipObj.write('overlay.kml')
           zipObj.write('overlay.png')


class AerialCorrection():
    def __init__(self, img, out_folder, pixel_size=4.4):
        self.img = img
        self.out_folder = out_folder
        self.temp_name = os.path.join(self.out_folder, os.path.splitext(os.path.basename(img))[0] + '_temp.tif')
        self.out_name = os.path.join(self.out_folder, os.path.splitext(os.path.basename(img))[0] + '.tif')
        self.pixel_size = pixel_size

    def rotation(self):
        latitude1, longitude1 = self.get_gps_info()

        roll_angle, yaw_angle, pitch_angle, altitude = self.get_image_info()
        yaw_angle = -1*yaw_angle
        roll_angle = -1*roll_angle
        img = cv2.imdecode(np.fromfile(self.img ,dtype=np.uint8),-1)
        # img = cv2.imread(self.img)  # 读取彩色图像(BGR)
        height, width = img.shape[:2]  # 图片的高度和宽度
        # 计算新的图像尺寸（这里实际上是原始尺寸的两倍）

        # 计算每个方向的填充量（这里是原始尺寸的一半）
        top = bottom = height // 2
        left = right = width // 2
        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        rotated_image = rotate_image(padded_img, yaw_angle)

        cropped_image = self.find_edge(rotated_image)

        new_height,new_width = cropped_image.shape[:2]

        focal_length = self.get_len_info()  # 镜头参数
        x_res_meter = float(self.pixel_size) * altitude / float(focal_length)  * 0.001  # 空间分辨率  单位是米

        x_res = x_res_meter / (2 * math.pi * 6371004) * 360   # 空间分辨率  单位是度 0.000008983
        left_coord = [longitude1 - x_res * new_width/ 2, latitude1 + x_res * new_height / 2]
        # left_coord = [longitude1, latitude1]
        out_png_geom = [left_coord[0], x_res, 0.0, left_coord[1], 0.0, -x_res]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        # export the png tif
        driver = gdal.GetDriverByName('GTiff')
        # out_tif = driver.Create(name=out_name, ysize=tar_y, xsize=tar_x, bands=tar_bandnum, eType=tar_datatype)
        out_tif = driver.Create(self.out_name, new_width, new_height, 3, eType=gdal.GDT_Byte)
        j=2
        for i in range(3):

            band = out_tif.GetRasterBand(i + 1).WriteArray(cropped_image[:,:,j-i])
            del band

        out_tif.SetProjection(srs.ExportToWkt())
        out_tif.SetGeoTransform(out_png_geom)
        out_tif.FlushCache()

    def find_edge(self,arr):
        # 删减图像中的黑框
        temp_arr = arr[:,:,0]
        # 二值化
        _, thresh = cv2.threshold(temp_arr, 1, 255, cv2.THRESH_BINARY)

        # 找到轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 如果没有检测到轮廓，则退出
        if not contours:
            print("No contours found.")
            exit()

        # 获取最大轮廓
        c = max(contours, key=cv2.contourArea)
        # 计算边界框
        x, y, w, h = cv2.boundingRect(c)
        cropped_image = arr[y:y+h, x:x+w,:]
        # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        # plt.show()
        return cropped_image

    def exif(self):
        """
        从图片中返回EXIF元数据
        """
        exif_data = {}

        try:
            i = Image.open(self.img)  # 使用PIL库打开图片
            tags = i._getexif()  # 获取图片的EXIF标签

            for tag, value in tags.items():
                decoded = TAGS.get(tag, tag)  # 尝试从预定义的TAGS字典中获取标签的中文描述，否则使用标签ID
                exif_data[decoded] = value  # 将标签及其值存储到exif_data字典中

        except Exception as e:
            print(e)  # 捕获所有异常并忽略，这通常不是一个好的做法，应该明确指定要捕获的异常

        return exif_data
    def get_image_size(self):
        i = Image.open(self.img)
        return i.width, i.height



    def get_image_info(self):
        """
        :param file_path: 输入图片路径
        :return: 图片的偏航角
        """
        # 获取图片偏航角
        b = b"\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e"
        a = b"\x3c\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20"
        img = open(self.img, 'rb')
        data = bytearray()
        dj_data_dict = {}
        flag = False
        for line in img.readlines():
            if a in line:
                flag = True
            if flag:
                data += line
            if b in line:
                break
        if len(data) > 0:
            data = str(data.decode('ascii'))
            lines = list(filter(lambda x: 'drone-dji:' in x, data.split("\n")))
            for d in lines:
                d = d.strip()[10:]
                key, value = d.split("=")
                dj_data_dict[key] = value

        return float(dj_data_dict["FlightRollDegree"][1:-1]),float(dj_data_dict["FlightYawDegree"][1:-1]),float(dj_data_dict["FlightPitchDegree"][1:-1]), float(dj_data_dict["RelativeAltitude"][1:-1])

    def get_len_info(self):
        exif = self.exif()
        if exif.get('FocalLength'):  #
            #'FocalLength'
            focal_length = exif['FocalLength']
        return focal_length

    def get_gps_info(self):
        """
        从EXIF元数据中提取GPS信息、偏航信息
        """
        exif = self.exif()
        lat = None  # 纬度
        lon = None  # 经度

        if exif.get('GPSInfo'):  # 如果EXIF中包含GPS信息
            # 纬度
            coords = exif['GPSInfo']
            i = coords[1]  # 纬度方向（N/S）
            d = coords[2][0]  # 纬度度数
            m = coords[2][1]  # 纬度分钟
            s = coords[2][2]  # 纬度秒
            lat = self.dms2dd(d, m, s, i)  # 将纬度转换为十进制度

            # 经度
            i = coords[3]  # 经度方向（E/W）
            d = coords[4][0]  # 经度度数
            m = coords[4][1]  # 经度分钟
            s = coords[4][2]  # 经度秒
            lon = self.dms2dd(d, m, s, i)  # 将经度转换为十进制度

        return lat, lon

    def dms2dd(self, d, m, s, i):
        """
        将度/分/秒转换为十进制度
        """
        sec = float((m * 60) + s)  # 将分和秒转换为秒
        dec = float(sec / 3600)  # 将秒转换为小数度
        deg = float(d + dec)  # 将度和小数度相加

        if i.upper() == 'W':  # 如果方向是西
            deg = deg * -1  # 将度数变为负数

        elif i.upper() == 'S':  # 如果方向是南
            deg = deg * -1  # 将度数变为负数

        return float(deg)
def rotate_image(image, yaw):
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算旋转中心
    center = (width // 2, height // 2)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, yaw, 1.0)

    # 应用偏航旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def get_file_names(file_dir, file_types):
    """
    搜索指定目录下具有给定后缀名的文件，不包括子目录。

    参数:
    file_dir (str): 目录路径。
    file_types (list[str] or str): 后缀名列表或单个后缀名（如 ['.txt', '.py'] 或 '.txt'）。

    返回:
    list[str]: 匹配的文件完整路径列表。
    """
    if isinstance(file_types, str):
        # 如果只传入了一个后缀名，将其转换为列表
        file_types = [file_types]

        # 使用glob模块搜索文件
    file_paths = []
    for file_type in file_types:
        # 使用glob的通配符模式搜索文件
        pattern = os.path.join(file_dir, '*' + file_type)
        file_paths.extend(glob.glob(pattern))
    filter_path = []
    for file in file_paths:
        if file not in filter_path:
            filter_path.append(file)
    return filter_path




def main(path,outpath,pixel_size=4.4):
    pixel_size = float(pixel_size)
    if os.path.exists(outpath) == False:
        print('输出路径不存在，创建输出路径')
        os.makedirs(outpath)
    else:
        print('输出路径已存在')
    print('开始进行航片粗几何校正')
    print('\n------------')
    file_types = ['.JPG', '.jpg']  #
    isfile = os.path.isfile(path)
    if isfile:
        print('你输入的是一个文件')
        file_type = os.path.splitext(path)[-1]
        if file_type == '.JPG' or file_type == '.jpg':
            A = AerialCorrection(path, outpath, pixel_size= pixel_size)
            A.rotation()
            tif2kmz(A.out_name).create_kmz()
        else:
            print('你输入的是文件类型不是JPG格式')
    else:
        print('你输入的是一个文件夹')
        print('即将进行批量处理')
        file_list1 = get_file_names(path, file_types)
        i = 0
        print(file_list1)
        for file in file_list1:
            A = AerialCorrection(file, outpath, pixel_size= pixel_size)
            A.rotation()

            tif2kmz(A.out_name).create_kmz()
            i += 1

            print("\r进行航片粗几何校正: [{0:50s}] {1:.1f}%".format('#' * int(i / (len(file_list1)) * 50),
                                                                  i / len(file_list1) * 100), end="",
                      flush=True)


if __name__ == '__main__':
    # A = AerialCorrection(r'D:\DJI_0154.JPG', r'D:\test', pixel_size= 2.41)
    # A = AerialCorrection(r'E:\000\DJI_20240608123140_0125_V.JPG', r'D:\test', pixel_size=4.4)
    # a = A.rotation()
    print('输入输出路径不包含中文')
    path = input('输入无人机照片路径：')
    outpath = input('输出路径：')
    pixel_size = input('输入像元尺寸：')

    # main(r'D:\111', r'D:\test', pixel_size= 4.4)
    # main(r'E:\000', r'D:\test', pixel_size=3.3)
    main(path, outpath, pixel_size)

    print('已完成')
    input('输入回车键退出')
