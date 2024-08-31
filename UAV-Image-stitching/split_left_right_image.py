#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : split_left_right_image.py 
'''

'''

import numpy as np

def define_line(matrix_size, points):
    """ Define a line through a series of points that should be within the index range of the matrix """
    # Here we return directly to the point set, with no additional processing required
    return np.array(points)

def split_matrix(matrix, line_points):
    """ The matrix is divided into left and right parts according to the given line points"""

    # Creates a Boolean array of the same size as the matrix to label the elements of the left matrix
    left_mask = np.zeros_like(matrix, dtype=bool)
    # Go through each row
    for j,_ in enumerate(line_points):
        if j > 0:
            left_mask[line_points[j-1][0]:line_points[j][0],:line_points[j][1]] = True
        else:
            left_mask[:line_points[j][0],:line_points[j][1]] = True

    left_matrix = matrix.copy()
    left_matrix[left_mask==False] = 0

    right_matrix = matrix.copy()
    right_matrix[left_mask] = 0


    return left_matrix, right_matrix

if __name__ == '__main__':

    # example
    matrix_size1, matrix_size2= 10, 11
    matrix = np.ones(shape=(10,11,3))
    # Definition line
    line_points = np.array([(1, 2), (3, 7), (4, 6), (6, 4), (8, 7), (9, 2)])

    # Split the matrix
    left_mat, right_mat = split_matrix(matrix, line_points)

    # print result
    print("Left Matrix:")
    print(left_mat)
    print("Right Matrix:")

    print(right_mat)
    print()