#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : create_point_gpkg.py
'''

'''
from osgeo import ogr, osr
def create_point_gpkg(longitude_latitude_list, filename, filetype='GPKG'):
    driver = ogr.GetDriverByName(filetype)
    data_source = driver.CreateDataSource(filename)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # 创建一个多边形层
    layer = data_source.CreateLayer("Polygon", srs, ogr.wkbPoint)
    field_name = ogr.FieldDefn("data", ogr.OFTString)
    field_name.SetWidth(14)
    layer.CreateField(field_name)




    for i, longitude_latitude in enumerate(longitude_latitude_list):
        point = ogr.Geometry(ogr.wkbPoint)
        # 向线性环中添加点
        point.AddPoint(float(longitude_latitude[0]), float(longitude_latitude[1]))
        # 将线性环添加到多边形中


        # 创建一个要素，并设置其几何形状和属性
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(point)
        feature.SetField("data", i+1)

        # 将要素添加到层中
        layer.CreateFeature(feature)

    # 清理资源
    feature = None
    data_source = None

    print(f"file {filename} has been created with a polygon feature.")

if __name__ == '__main__':
    points = [(1, 1), (3, 0), (3, 3), (0, 4)]
    output_file = r'd://temp/create_point_by_ogr.gpkg'
    create_point_gpkg(points, output_file)
