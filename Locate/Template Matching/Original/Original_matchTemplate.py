# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:15:47 2021

@author: USER
"""
import cv2 as cv
import numpy as np
import time

#開始計時
start = time.time()

#讀取地圖圖像與模板圖像
img = cv.imread('C:/Users/USER/Desktop/95213057_20181030_108EMAP.jpg')
img = cv.resize(img, (1024, 768))
img2 = img.copy()
template = cv.imread('C:/Users/USER/Desktop/1.jpg')
template = cv.resize(template,(25, 15))
cv.imshow("Original", template)
cv.waitKey(0)
cv.destroyAllWindows()

#為了縮減地圖與模板圖像間的亮度對比，因此藉由降低像素來達到減少差異性
num = np.zeros(template.shape, template.dtype) + 50
template = cv.subtract(template, num)
cv.imshow("Subtract", template)
cv.waitKey(0)
cv.destroyAllWindows()

w = template.shape[1]
h = template.shape[0]

#選擇模板匹配的演算法
methods = ['cv.TM_SQDIFF_NORMED']#'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

#開始進行模板匹配
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
    
    #將匹配到的範圍標記在地圖圖像中
    cv.rectangle(img,top_left, bottom_right, 255, 1)
    cv.imshow("Result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    #顯示匹配到的區域
    crop_image = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv.imwrite("Fifth.jpg", crop_image)
    cv.imshow("Cropped", crop_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    #結算花費時間
    end = time.time()
    cost = end - start
    print("Cost = ", cost)
    