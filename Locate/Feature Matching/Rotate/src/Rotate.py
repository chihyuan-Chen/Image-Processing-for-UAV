# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:15:47 2021

@author: USER
"""
import cv2 as cv
import numpy as np
import time
import csv
import os

'''-------------Blurring----------------'''
def Blurred(image02):
    blurred = cv.blur(image02, (13, 13))#6 6
    return blurred

def Median_Filter(image02):
    blurred = cv.medianBlur(image02, 13)
    return blurred

def Gaussian_Filter(image02):
    blurred = cv.GaussianBlur(image02, (13, 13), 0)
    return blurred

def Bilateral_Filter(image02):
    blurred = cv.bilateralFilter(image02, 5, 0, 150)
    return blurred

'''--------------Feature Matching------------------'''
def Feature(image01, image02):
    start = time.time()
    gray01 = cv.cvtColor(image01, cv.COLOR_BGR2GRAY) #Let image01 convert color from RGB to gray
    gray02 = cv.cvtColor(image02, cv.COLOR_BGR2GRAY) #Let image02 convert color from RGB to gray
    
    detector = 'KAZE' #'SURF'、'KAZE'、'ORB'、'BRISK'、'AKAZE'
    BF_MATCHER = 0
    
    # Select detector and use matcher
    if detector == 'SIFT':
        feature = SIFT()
        keypoints1 = feature.detect(gray01)
        keypoints2 = feature.detect(gray02)
 
        (keypoints1, descriptors1) = feature.compute(gray01, keypoints1)
        (keypoints2, descriptors2) = feature.compute(gray02, keypoints2)
        
        BF_MATCHER  = BF(image01, image02, descriptors1, descriptors2, keypoints1, keypoints2, detector)

    elif detector == 'SURF':
        feature = SURF()
        keypoints1 = feature.detect(gray01)
        keypoints2 = feature.detect(gray02)
 
        (keypoints1, descriptors1) = feature.compute(gray01, keypoints1)
        (keypoints2, descriptors2) = feature.compute(gray02, keypoints2)
        
        BF_MATCHER  = BF(image01, image02, descriptors1, descriptors2, keypoints1, keypoints2, detector)
        
    elif detector == 'KAZE':
        feature = KAZE()
        keypoints1 = feature.detect(gray01)
        keypoints2 = feature.detect(gray02)
 
        (keypoints1, descriptors1) = feature.compute(gray01, keypoints1)
        (keypoints2, descriptors2) = feature.compute(gray02, keypoints2)

        BF_MATCHER  = BF(image01, image02, descriptors1, descriptors2, keypoints1, keypoints2, detector)
        
    elif detector == 'ORB':
        feature = ORB()
        keypoints1 = feature.detect(gray01)
        keypoints2 = feature.detect(gray02)
 
        (keypoints1, descriptors1) = feature.compute(gray01, keypoints1)
        (keypoints2, descriptors2) = feature.compute(gray02, keypoints2)

        BF_MATCHER  = BF(image01, image02, descriptors1, descriptors2, keypoints1, keypoints2, detector)
        
    elif detector == 'BRISK':
        feature = BRISK()
        keypoints1 = feature.detect(gray01)
        keypoints2 = feature.detect(gray02)
 
        (keypoints1, descriptors1) = feature.compute(gray01, keypoints1)
        (keypoints2, descriptors2) = feature.compute(gray02, keypoints2)

        BF_MATCHER  = BF(image01, image02, descriptors1, descriptors2, keypoints1, keypoints2, detector)

    elif detector == 'AKAZE':
        feature = AKAZE()
        keypoints1 = feature.detect(gray01)
        keypoints2 = feature.detect(gray02)
 
        (keypoints1, descriptors1) = feature.compute(gray01, keypoints1)
        (keypoints2, descriptors2) = feature.compute(gray02, keypoints2)

        BF_MATCHER  = BF(image01, image02, descriptors1, descriptors2, keypoints1, keypoints2, detector)
    
    #Time Calculation
    end = time.time()
    cost = end - start
    
    #Image show controll
    output1 = cv.resize(BF_MATCHER , (1440, 800))

    #Return the image and time
    return cost, output1

'''-------------- All kinds of detectors --------------------------'''
# Call function SIFT
def SIFT():
    # Initiate SIFT detector
    SIFT = cv.xfeatures2d.SIFT_create()

    return SIFT

# Call function SURF
def SURF():
    # Initiate SURF descriptor
    SURF = cv.xfeatures2d.SURF_create()

    return SURF

# Call function KAZE
def KAZE():
    # Initiate KAZE descriptor
    KAZE = cv.KAZE_create()

    return KAZE

# Call function ORB
def ORB():
    # Initiate ORB detector
    ORB = cv.ORB_create()

    return ORB

# Call function BRISK
def BRISK():
    # Initiate BRISK descriptor
    BRISK = cv.BRISK_create()

    return BRISK

# Call function AKAZE
def AKAZE():
    # Initiate AKAZE descriptor
    AKAZE = cv.AKAZE_create()

    return AKAZE

'''-------------- BF Matcher --------------------------'''
def BF(image01, image02, descriptors1, descriptors2, keypoints1, keypoints2, detector):
    # Se descritor for um Descritor de Recursos Locais utilizar NOME
    if (detector == 'SIFT') or (detector == 'SURF') or (detector == 'KAZE'):
        normType = cv.NORM_L2
    else:
        normType = cv.NORM_HAMMING

    # Create BFMatcher object
    BFMatcher = cv.BFMatcher(normType = normType, crossCheck = True)

    # Matching descriptor vectors using Brute Force Matcher
    matches = BFMatcher.match(queryDescriptors = descriptors1, trainDescriptors = descriptors2)

    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x: x.distance)

    # Draw first 30 matches
    output = cv.drawMatches(img1 = image01, keypoints1 = keypoints1, img2 = image02, keypoints2 = keypoints2, matches1to2 = matches[:20], outImg = None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return output

'''-------------- FLANN Matcher --------------------------'''
def FLANN(descriptors1, descriptors2, keypoints1, keypoints2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

    search_params = dict(checks = 50)

    # Converto to float32
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    # Create FLANN object
    FLANN = cv.FlannBasedMatcher(indexParams = index_params, searchParams = search_params)

    # Matching descriptor vectors using FLANN Matcher
    matches = FLANN.knnMatch(queryDescriptors = descriptors1, trainDescriptors = descriptors2, k = 2)

    # Lowe's ratio test
    ratio_thresh = 0.7

    # "Good" matches
    good_matches = []
    
    # Filter matches
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Draw only "good" matches
    output = cv.drawMatches(img1 = image01, keypoints1 = keypoints1, img2 = image02, keypoints2 = keypoints2, matches1to2 = good_matches, outImg = None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return output

'''-------------- Rotate ---------------'''
def rotate(image, angle, scale = 1.0):
    height, width, channel = image.shape
    cx = int(width / 2)
    cy = int(height / 2)
    center = (cx, cy)
    new_dim = (width, height)
    M = cv.getRotationMatrix2D(center, angle, scale)
    image = cv.warpAffine(image, M, new_dim) 
    return image

'''-------------- Main --------------------------'''
if __name__ == '__main__':
    image01 = cv.imread("C:/Users/USER/Desktop/map.jpg") # Loading image 'map.jpg' into image01 
    image02 = cv.imread("C:/Users/USER/Desktop/2.jpg") # Loading image '2.jpg' into image02 
    image02 = cv.resize(image02,(373,480))
    
    '''----------Pre-Processing----------------'''
    #image03 = Blurred(image02) 
    #image03 = Median_Filter(image02) 
    #image03 = Gaussian_Filter(image02) 
    image03 = Bilateral_Filter(image02)
    
    '''----------0, 45, 90, 135, 180, 225, 270, 315-----------'''
    for i in range(8):
        if i == 0:
            value, result = Feature(image01, image03)
            print(value)
            cv.imshow("BF Result", result)
            cv.waitKey(0)
            cv.destroyAllWindows()       
        elif i == 1:
            image04 = rotate(image03, 45)
            value, result = Feature(image01, image04)
            print(value)
            cv.imshow("BF Result", result)
            cv.waitKey(0)
            cv.destroyAllWindows()             
        elif i == 2:
            image04 = rotate(image03, 90)
            value, result = Feature(image01, image04)
            print(value)
            cv.imshow("BF Result", result)
            cv.waitKey(0)
            cv.destroyAllWindows()
        elif i == 3:
            image04 = rotate(image03, 135)
            value, result = Feature(image01, image04)
            print(value)
            cv.imshow("BF Result", result)
            cv.waitKey(0)
            cv.destroyAllWindows()
        elif i == 4:
            image04 = rotate(image03, 180)
            value, result = Feature(image01, image04)
            print(value)
            cv.imshow("BF Result", result)
            cv.waitKey(0)
            cv.destroyAllWindows()
        elif i == 5:
            image04 = rotate(image03, 225)
            value, result = Feature(image01, image04)
            print(value)
            cv.imshow("BF Result", result)
            cv.waitKey(0)
            cv.destroyAllWindows()
        elif i == 6:
            image04 = rotate(image03, 270)
            value, result = Feature(image01, image04)
            print(value)
            cv.imshow("BF Result", result)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            image04 = rotate(image03, 315)
            value, result = Feature(image01, image04)
            print(value)
            cv.imshow("BF Result", result)
            cv.waitKey(0)
            cv.destroyAllWindows()