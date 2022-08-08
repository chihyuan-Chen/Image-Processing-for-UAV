# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:15:47 2021

@author: Ivan
"""
import cv2 as cv
import numpy as np
import time

start = time.time()

img = cv.imread('C:/Users/USER/Desktop/95213057_20181030_108EMAP.jpg')
img = cv.resize(img, (1024, 768))
img2 = img.copy()
template = cv.imread('C:/Users/USER/Desktop/1.jpg')
template = cv.resize(template,(25, 15))
cv.imshow("Original", template)
cv.waitKey(0)
cv.destroyAllWindows()

num = np.zeros(template.shape, template.dtype) + 50
template = cv.subtract(template, num)
cv.imshow("Subtract", template)
cv.waitKey(0)
cv.destroyAllWindows()

w = template.shape[1]
h = template.shape[0]

methods = ['cv.TM_SQDIFF_NORMED']#'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv.rectangle(img,top_left, bottom_right, 255, 1)
    cv.imshow("Result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    crop_image = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv.imwrite("Fifth.jpg", crop_image)
    cv.imshow("Cropped", crop_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    end = time.time()
    cost = end - start
    print("Cost = ", cost)
    
