#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2024/8/13 下午4:57 
# @File : use_pcv_sift.py 
'''

'''

from PIL import Image
from pylab import *
import sys
from PCV.localdescriptors import sift



im1f = r'D:\temp\hill1.JPG'
im2f = r'D:\temp\hill2.JPG'
#  im1f = '../data/crans_1_small.jpg'
#  im2f = '../data/crans_2_small.jpg'
#  im1f = '../data/climbing_1_small.jpg'
#  im2f = '../data/climbing_2_small.jpg'
im1 = array(Image.open(im1f))
im2 = array(Image.open(im2f))

sift.process_image(im1f, 'out_sift_1.txt')
l1, d1 = sift.read_features_from_file('out_sift_1.txt')
figure()
gray()
subplot(121)
sift.plot_features(im1, l1, circle=False)

sift.process_image(im2f, 'out_sift_2.txt')
l2, d2 = sift.read_features_from_file('out_sift_2.txt')
subplot(122)
sift.plot_features(im2, l2, circle=False)

matches = sift.match(d1, d2)
matches = sift.match_twosided(d1, d2)
print ('{} matches'.format(len(matches.nonzero()[0])))

figure()
gray()
sift.plot_matches(im1, im2, l1, l2, matches, show_below=True)
show()