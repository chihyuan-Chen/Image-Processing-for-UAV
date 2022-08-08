# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:15:47 2021

@author: USER
"""
import cv2 as cv
import numpy as np 
import time

#%%預處理
class Preprocess():
    '''------------參數初始化------------'''
    def __init__(self, img, list, coordinate):
        #定義輸入圖像
        self.image = img
        
        #image_1作為輸入圖像的暫存
        self.image_1 = img.copy()
        
        #取得輸入圖像的寬和高
        self.width = img.shape[1]
        self.height = img.shape[0]
        
        #定義切成幾等分
        self.list = list
        
        #定義圖像裁切後的寬和高
        self.sub_width = (self.width//list[1])
        self.sub_height = (self.height//list[0])
        
        #定義裁切後圖像的座標範圍
        self.sub_range = [(coordinate[2]-coordinate[0])/list[0],(coordinate[3]-coordinate[1])/list[1]]
        
        #定義初始經度
        self.init_longitude = coordinate[0]
        
        #定義初始緯度
        self.init_latitude = coordinate[1]
    
    '''-------------裁切圖像--------------'''
    def cut_image(self):
        #定義參數用於儲存裁切後的圖像及圖像個別的左上右下頂點座標
        box_list=[]
        position_list=[]
        
        #運用雙層迴圈進行裁切和儲存座標
        for i in range(0,int(self.list[0])):
            for j in range(0,int(self.list[1])):
                box = self.image[i*self.sub_height:(i+1)*self.sub_height, j*self.sub_width:(j+1)*self.sub_width]
                position_list.append([self.init_longitude + j*self.sub_range[0], 
                                      self.init_latitude + i*self.sub_range[1], 
                                      self.init_longitude + (j+1)*self.sub_range[0], 
                                      self.init_latitude + (i+1)*self.sub_range[1]])
                box_list.append(box)
        
        #回傳裁切後的所有圖像和對應的座標點
        return box_list, position_list
    
    '''-------------顯示裁切後的個別圖像----------------'''
    def cut_result(self, image_list):
        #定義參數用於儲存裁切後的順序
        number_list = []
        count = 0
        
        #利用迴圈儲存裁切後的順序(順序為由左至右，由上到下。設 1 起始)
        for image in image_list:
            count+=1
            number_list.append(str(count))
            
            #顯示依序顯示裁切後的結果
            image = cv.resize(image, (1024, 768))
            cv.imshow(number_list[count-1], image)
            cv.waitKey(0)
            cv.destroyAllWindows()      
    
    '''---------------顯示劃分區域----------------'''
    def show_region(self):
        #運用雙層迴圈劃分出切割的範圍
        for i in range(0,int(self.list[0])):
            for j in range(0,int(self.list[1])):
                cv.rectangle(self.image_1, (j*self.sub_width, i*self.sub_height), 
                             ((j+1)*self.sub_width, (i+1)*self.sub_height), 
                             (0,0,255), 20)
                
        #重新定義顯示的尺寸大小(1024*768
        image = cv.resize(self.image_1, (1024, 768))
        
        #顯示劃分的區域範圍
        cv.imshow('Show_region', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
#%%模板匹配       
class Match_Template():
    '''------------參數初始化------------'''
    def __init__(self, template, image_list, position):
        self.template = template
        
        #定義模板圖像的寬和高的參數
        self.t_width = 0
        self.t_height = 0
        
        #裁切後的圖像集
        self.image_list = image_list
        
        #裁切後圖像座標集
        self.position = position
    
    '''-----------縮小匹配範圍且儲存圖像縮小後的座標點------------'''
    def Zoom_out_Match(self):
        #模板圖像縮小至70*70，對應地圖圖像的比例
        self.template = cv.resize(self.template,(70, 70))#70 70 #25 30 #25 35 #35 35 #改100 100 #70 70 #60 50#180 180
        
        #開啟 Bilateral_Filter 雙邊濾波器
        self.template = self.Bilateral_Filter(self.template)
        
        #開啟 Canny 邊緣檢測
        self.template = cv.cvtColor(self.template, cv.COLOR_RGB2GRAY)
        self.template = self.Canny(self.template)
        
        #取得模板圖像的寬和高
        self.t_width = self.template.shape[1]
        self.t_height = self.template.shape[0] 
        
        #顯示縮小後的模板圖像
        cv.imshow("Zoom_out_Template", self.template)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        #選擇模板匹配的演算法(cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED')
        methods = ['cv.TM_SQDIFF_NORMED']
        
        image = []#切割後的圖
        result = []#存放匹配的結果圖像
        coordinate_list = []#存放匹配結果的座標
        
        #利用將裁切後的每張圖像縮放至2048*1536(為了獲得較好的匹配效果，進行縮放後能使圖像特徵更為彰顯)，並重新計算和儲存每張圖像的座標
        for i in range(0, len(self.image_list)):
            self.image_list[i] = cv.resize(self.image_list[i], (2048, 1536))
            
            #開啟 Bilateral_Filter 雙邊濾波器
            #self.image_list[i] = self.Bilateral_Filter2(self.image_list[i])
            
            #開啟 Canny 邊緣檢測
            #self.image_list[i] = cv.cvtColor(self.image_list[i], cv.COLOR_RGB2GRAY)
            #self.image_list[i] = self.Canny2(self.image_list[i])
            
            image.append(self.image_list[i])
        
        new_position = [(self.position[0][2]-self.position[0][0])/2048, 
                        (self.position[0][3]-self.position[0][1])/1536]

        #針對每張裁切圖像進行模板匹配
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
                    
                    #計算匹配結果圖像的座標點
                    coordinate = [self.position[i][0]+(top_left[0] + (self.t_width/2))*new_position[0], 
                                  self.position[i][1]+(top_left[1] + (self.t_height/2))*new_position[1]]
                    
                    #儲存並顯示匹配結果圖像的座標點
                    coordinate_list.append([coordinate[0], coordinate[1]])
                    print("經度:", coordinate[0], "緯度:", coordinate[1])
                    
                    #儲存匹配後的結果圖
                    sub = image[i]
                    result.append(sub[top_left[1]:top_left[1] + self.t_height, top_left[0]:top_left[0] + self.t_width])
                    
                    #顯示匹配結果
                    cv.imshow("Matching_result", result[j])
                    j+=1
                    cv.waitKey(0)
                    cv.destroyAllWindows()
        
        #做第二次匹配，尋找出最佳匹配結果與其座標
        self.Find_correct_Match(result, coordinate_list)
        
    #模板圖像的雙邊濾波器
    def Bilateral_Filter(self, image):
        blurred = cv.bilateralFilter(image, 8, 280, 100)#此處可進行參數調整 #**2乘2**8 280 100 #**5乘5**8 280 50
        cv.imshow("Bilateral_Filter", blurred)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return blurred
    
    #模板圖像的邊緣檢測
    def Canny(self, image):
        edge = cv.Canny(image, 100, 200)#此處可進行參數調整 #**都是使用此參數**100 200
        cv.imshow("Canny", edge)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return edge
    
    #地圖圖像的雙邊濾波器
    def Bilateral_Filter2(self, image):
        blurred = cv.bilateralFilter(image, 5, 200, 100)#此處可進行參數調整 #**2乘2**5 200 100 #**5乘5**3 50 50
        cv.imshow("Bilateral_Filter2", blurred)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return blurred
    
    #地圖圖像的邊緣檢測
    def Canny2(self, image):
        edge = cv.Canny(image, 40, 255)#此處可進行參數調整 #**2乘2***40 255 #**5乘5**0 255
        cv.imshow("Canny2", edge)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return edge
    
    #尋找無人機座標位置
    def Find_correct_Match(self, result, coordinate_list):
        #將模板圖像裁切集中於左上(去除模板圖像上的邊緣景物，使中間景物特徵能更明顯)，並將尺寸還原成70*70。
        self.template = cv.resize(self.template[0:self.t_height-5, 0:self.t_width-5], (70, 70))#############70 70#170 235
        
        #顯示調整後的模板圖像
        cv.imshow("Tuning_Template", self.template)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        #儲存最高相似度的值
        Max = 0
        
        #儲存最高相似度的圖像編號
        Num = 0
        
        #選擇模板匹配的演算法(cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED')
        methods = ['cv.TM_CCOEFF_NORMED']
        
        #進行模板匹配找出最佳匹配結果
        for meth in methods:
            for i in range(0, len(result)): 
                method = eval(meth)
                res = cv.matchTemplate(result[i], self.template, method)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                if Max < max_val:
                    Max = max_val
                    Num = i
                    
        #顯示最佳匹配結果的座標點及圖像
        print("正確經度:", coordinate_list[Num][0], "正確緯度:", coordinate_list[Num][1])
        cv.imshow("Result", result[Num])
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        #結算花費時間
        #end = time.time()
        #cost = end - start
        #print("Cost = ", cost)
  
#%%主程式
if __name__=='__main__':
    '''-----------開始計時-----------'''
    #start = time.time()
    
    '''-----------載入地圖及航拍即時影像-----------'''
    #定義地圖圖像為"img"
    img = cv.imread('C:/Users/USER/Desktop/95213057_20181030_108EMAP.jpg')
    #定義航拍即時影像為"模板圖像(template)"
    template = cv.imread('C:/Users/USER/Desktop/6.jpg')
    
    '''---------設置地圖座標點:依序為左上角座標"經度"、"緯度"，右上角座標"經度"、"緯度"---------'''
    coordinate = [120.657740, 24.123560,   120.683311, 24.098121]
    
    '''---------定義切成幾等分，下方list = [2, 2]為2*2=4等分---------'''
    list = [2, 2]#可改成[5, 5]切成25等分
    
    '''---------裁切地圖及儲存裁切後的座標點----------'''
    #初始化
    preprocess = Preprocess(img, list, coordinate)
    #顯示4等分裁切的範圍
    preprocess.show_region()
    #進行地圖圖像裁切
    image_list, position_list = preprocess.cut_image()
    #顯示裁切後的結果
    preprocess.cut_result(image_list)
    
    '''---------減少輸入圖像的像素質----------'''
    #為了縮減地圖與輸入圖像間的亮度對比，因此藉由降低像素來達到減少差異性
    num = np.zeros(template.shape, template.dtype) + 20
    template = cv.subtract(template, num)
    
    
    '''---------進行模板匹配找到與地圖圖像匹配的最佳結果並顯示無人機的座標位置----------'''
    match = Match_Template(template, image_list, position_list)
    match.Zoom_out_Match()


    