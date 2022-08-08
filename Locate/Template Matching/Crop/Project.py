# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:15:47 2021

@author: Ivan
"""
import cv2 as cv
import numpy as np 
import time


class Preprocess():
    def __init__(self, img, list, coordinate):
        self.image = img
        
        self.image_1 = img.copy()
        
        self.width = img.shape[1]
        self.height = img.shape[0]
        
        self.list = list
        
        self.sub_width = (self.width//list[1])
        self.sub_height = (self.height//list[0])
        
        self.sub_range = [(coordinate[2]-coordinate[0])/list[0],(coordinate[3]-coordinate[1])/list[1]]
        
        self.init_longitude = coordinate[0]
        
        self.init_latitude = coordinate[1]
    
    def cut_image(self):
        box_list=[]
        position_list=[]
        
        for i in range(0,int(self.list[0])):
            for j in range(0,int(self.list[1])):
                box = self.image[i*self.sub_height:(i+1)*self.sub_height, j*self.sub_width:(j+1)*self.sub_width]
                position_list.append([self.init_longitude + j*self.sub_range[0], 
                                      self.init_latitude + i*self.sub_range[1], 
                                      self.init_longitude + (j+1)*self.sub_range[0], 
                                      self.init_latitude + (i+1)*self.sub_range[1]])
                box_list.append(box)
        
        return box_list, position_list
    
    def cut_result(self, image_list):
        number_list = []
        count = 0
        
        for image in image_list:
            count+=1
            number_list.append(str(count))
            
            image = cv.resize(image, (1024, 768))
            cv.imshow(number_list[count-1], image)
            cv.waitKey(0)
            cv.destroyAllWindows()      
    
    def show_region(self):
        for i in range(0,int(self.list[0])):
            for j in range(0,int(self.list[1])):
                cv.rectangle(self.image_1, (j*self.sub_width, i*self.sub_height), 
                             ((j+1)*self.sub_width, (i+1)*self.sub_height), 
                             (0,0,255), 20)
                
        image = cv.resize(self.image_1, (1024, 768))
        
        cv.imshow('Show_region', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
      
class Match_Template():
    def __init__(self, template, image_list, position):
        self.template = template
        
        self.t_width = 0
        self.t_height = 0
        
        self.image_list = image_list
        
        self.position = position
    
    def Zoom_out_Match(self):
        self.template = cv.resize(self.template,(70, 70))#70 70 #25 30 #25 35 #35 35 #改100 100 #70 70 #60 50#180 180
        
        self.template = self.Bilateral_Filter(self.template)
        
        self.template = cv.cvtColor(self.template, cv.COLOR_RGB2GRAY)
        self.template = self.Canny(self.template)
        
        self.t_width = self.template.shape[1]
        self.t_height = self.template.shape[0] 
        
        cv.imshow("Zoom_out_Template", self.template)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        #(cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED')
        methods = ['cv.TM_SQDIFF_NORMED']
        
        image = []
        result = []
        coordinate_list = []
        
        for i in range(0, len(self.image_list)):
            self.image_list[i] = cv.resize(self.image_list[i], (2048, 1536))
            
            #self.image_list[i] = self.Bilateral_Filter2(self.image_list[i])
            
            #self.image_list[i] = cv.cvtColor(self.image_list[i], cv.COLOR_RGB2GRAY)
            #self.image_list[i] = self.Canny2(self.image_list[i])
            
            image.append(self.image_list[i])
        
        new_position = [(self.position[0][2]-self.position[0][0])/2048, 
                        (self.position[0][3]-self.position[0][1])/1536]

        for meth in methods:
            j = 0
            for i in range(0, len(image)):
                method = eval(meth)
                res = cv.matchTemplate(image[i], self.template, method)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                threshold = 0.8
                if threshold < max_val or min_val < 0.2:
                    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                        top_left = min_loc
                    else:
                        top_left = max_loc
                    
                    coordinate = [self.position[i][0]+(top_left[0] + (self.t_width/2))*new_position[0], 
                                  self.position[i][1]+(top_left[1] + (self.t_height/2))*new_position[1]]
                    
                    coordinate_list.append([coordinate[0], coordinate[1]])
                    print("經度:", coordinate[0], "緯度:", coordinate[1])
                    
                    sub = image[i]
                    result.append(sub[top_left[1]:top_left[1] + self.t_height, top_left[0]:top_left[0] + self.t_width])
                    
                    cv.imshow("Matching_result", result[j])
                    j+=1
                    cv.waitKey(0)
                    cv.destroyAllWindows()
        
        self.Find_correct_Match(result, coordinate_list)
        
    def Bilateral_Filter(self, image):
        blurred = cv.bilateralFilter(image, 8, 280, 100)
        cv.imshow("Bilateral_Filter", blurred)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return blurred
    
    def Canny(self, image):
        edge = cv.Canny(image, 100, 200)
        cv.imshow("Canny", edge)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return edge
    
    def Bilateral_Filter2(self, image):
        blurred = cv.bilateralFilter(image, 5, 200, 100)
        cv.imshow("Bilateral_Filter2", blurred)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return blurred
    
    def Canny2(self, image):
        edge = cv.Canny(image, 40, 255)
        cv.imshow("Canny2", edge)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return edge
    
    def Find_correct_Match(self, result, coordinate_list):
        self.template = cv.resize(self.template[0:self.t_height-5, 0:self.t_width-5], (70, 70)
        
        cv.imshow("Tuning_Template", self.template)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        Max = 0
        
        Num = 0
        
        #(cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED')
        methods = ['cv.TM_CCOEFF_NORMED']
        
        for meth in methods:
            for i in range(0, len(result)): 
                method = eval(meth)
                res = cv.matchTemplate(result[i], self.template, method)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                if Max < max_val:
                    Max = max_val
                    Num = i
                    
        print("正確經度:", coordinate_list[Num][0], "正確緯度:", coordinate_list[Num][1])
        cv.imshow("Result", result[Num])
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        #end = time.time()
        #cost = end - start
        #print("Cost = ", cost)
  
                                  
if __name__=='__main__':
    #start = time.time()
    
    img = cv.imread('C:/Users/USER/Desktop/95213057_20181030_108EMAP.jpg')
    template = cv.imread('C:/Users/USER/Desktop/6.jpg')
    
    coordinate = [120.657740, 24.123560,   120.683311, 24.098121]
    
    list = [2, 2]
    
    preprocess = Preprocess(img, list, coordinate)
    preprocess.show_region()
    image_list, position_list = preprocess.cut_image()
    preprocess.cut_result(image_list)
    
    num = np.zeros(template.shape, template.dtype) + 20
    template = cv.subtract(template, num)
    
    match = Match_Template(template, image_list, position_list)
    match.Zoom_out_Match()


    
