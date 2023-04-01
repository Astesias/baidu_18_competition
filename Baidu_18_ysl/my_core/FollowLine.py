# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:33:51 2022

@author: 鞠秉宸
"""

import cv2
import numpy as np
import time
import math
import cmath
weight = [6]*20+[6]*20+[6]*20+[10]*20+[10]*20+[10]*20+[10]*20+[10]*20+[5]*20+[4]*80
board=144
board0=144
last_angle=0
def linear_regression(x,y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
    A = np.mat([[N,sumx],[sumx,sumx2]])
    b = np.array([sumy,sumxy])

    return np.linalg.solve(A,b)
    
def get_argmax(ls):
    longest_value = np.argmax(ls)
    if longest_value == 0 and ls[0] == 0:
        longest_value=len(ls)-1
    return longest_value
    
class FollowLine(object):
    def __init__(self):
        self.LineValue = -1
        self.width = 100
        self.height = 100

    def GetValue(self,mask):

        flip_img = np.flip(mask.T,axis=1)
        global board
        global board0
        board=board0
        #cv2.imshow("disp",image)
        '''
        ls=[]
        mid = self.width//2-1
        row_index = self.height-1
        while(mask[row_index,mid]!=255 and row_index>=0):
            for i in range(mid,0,-1):
                left=i
                if(mask[row_index,i]==255):
                    break
            for i in range(mid+1,self.width):
                right=i
                if(mask[row_index,i]==255):
                    break    
            ls.append((left+right)//2)
            mid = (left+right)//2
            mask[row_index,(left+right)//2]=255
            row_index=row_index-1
        cv2.imshow("disp",mask)
        return sum(ls)//len(ls)-self.width//2
        '''
        
        #cv2.imshow("disp",flip_img)
        mid = self.width//2
        saobudao=0
        #saobudaoz=0
        #saobudaoy=0
        for value in range(0,50,10):
            for i in range(mid,0,-1):
                left=i
                if(mask[self.height-1-value,i]==255 and mask[self.height-3-value,i]==255):
                    break
            for i in range(mid+1,self.width):
                right=i
                if(mask[self.height-1-value,i]==255 and mask[self.height-3-value,i]==255):
                    break    
            if(right-left>30):
                break
      #  print("value="+str(value))
       # print("left,right="+str(left)+','+str(right))
        num=[]
        dis = 3
        for i in range(left,right,dis):
            num.append(get_argmax(flip_img[i,:]==255))
        max_index = np.argmax(np.array(num))
        long_value = num[max_index]
        
        #print('long_value='+str(long_value))
        #print("max_index="+str(max_index*dis+left))
        
        mask_left = np.flip(mask[:,:max_index*dis+left],axis=1)
        mask_right = mask[:,max_index*dis+left:]
        
        num=[]
        left_line=[]
        right_line=[]
        midline=[]
        numf=[]
        bb=(self.width)//2
        saobudaoz=0
        for i in range(self.height-1,self.height-1-long_value,-1):
            left_line.append(max_index*dis+1*left-get_argmax(mask_left[i,:]==255))
            right_line.append(max_index*dis+1*left+get_argmax(mask_right[i,:]==255))
            #bb=(max_index*dis*2+2*left-get_argmax(mask_left[i,:]==255)+get_argmax(mask_right[i,:]==255))//2
           #midline.append(bb)
            #mask[i,bb]=255
        #计数器
        count=0
        for i in range(0,long_value):
            temp=right_line[i]-left_line[i]
            bb=(right_line[i]+left_line[i])//2
            midline.append(bb)
           # midline.append((right_line[i]+left_line[i])//2)
            if (temp>30):
                if(left_line[i]>=5 and right_line[i]<=95):
                    #print('s',(max_index*dis+1*left+get_argmax(mask_right[i,:]==255)-max_index*dis+1*left-get_argmax(mask_left[i,:]==255)))
                     #print('board',board)
                     #print('i',i)
                     board=temp
                    # print('board',board)
                     num.append(bb)
                     numf.append(bb)
                     if(i<5):
                        board0=board
                        
                elif(left_line[i]>5 and right_line[i]>95):
                    #saobudaoy=saobudaoy+1
                    tt=(2*left_line[i]+board)//2
                    
                    num.append(tt)
                    numf.append(tt)
                elif(left_line[i]<5 and right_line[i]<95):
                    saobudaoz=saobudaoz+1
                    tt=(2*right_line[i]-board)//2
                    num.append(tt)
                    numf.append(tt)
                else:
                     saobudao=saobudao+1  
                     numf.append(50)
        '''
        for i in range(0,long_value-1):             
             if(midline[i]-midline[i+1]==1 ):
                #print('count',count)
                count=count+1
             elif(count>=5):
                break
             else:
                count=0
        '''
        #print('count',count)
        xieshizi=0
                           
        if(saobudao>20) and (count>=5)and (long_value>90):
           # print('xieshizi')
            xieshizi=1
            
            #print(midline[i]-midline[i+1])           
        #for i in range(0,long_value): 
            
        #print('1',right_line[long_value-1]-right_line[long_value-2])
        #print('2',left_line[long_value-1]-left_line[long_value-2])

            
           



        if len(num)<2:
            num.append(50)
            num.append(50)
            num.append(50)
        #print('num=',num)
       # num.reverse()
        global last_angle
        
        

        
        
        kp=60
        k=0.95#0.95 #1.5-0.02*abs(last_angle)
        v=43
        if(xieshizi==0):
            a0,a1 = linear_regression(np.array(range(len(num))),num)
            theta = math.atan(a1)
            x=a0-0.5*self.width
            my_angle = math.atan(x/(math.cos(theta)*(k*v-math.sin(theta)*x))+a1)
            #print("my_angle=",my_angle)

        else:
            a0,a1 = linear_regression(np.array(range(len(numf))),numf)
            theta = math.atan(a1)
            x=a0-0.5*self.width
            my_angle = math.atan(x/(math.cos(theta)*(k*v-math.sin(theta)*x))+a1)
            #print("my_angle=",my_angle)

  
        my_angle = 0.9*my_angle+0.1*last_angle
        last_angle = my_angle
        


        ''' 
        for i in range(self.height-1-long_value,self.height):
            b=int(a0+a1*(self.height-1-i))
            if (b>=255):
                b=254
            mask[i,b]=255
            
        '''  
    # print("a0,a1="+str(a0)+","+str(a1))
       # print(a0+a1*(self.height-1-long_value*0.5))
        
        
        #print('mean',mean_value)
        #mean_value = mean_value//sum(weight[0:len(num)])
        #print(mean_value)
       # mean_value=sum(num)//len(num)
       # print('sss')
       # print('mean',mean_value)
        #print('saobudaoz',saobudaoz)
        #print('saobudaoy',saobudaoy)
        #print('saobudao',saobudao)
        #for i in range(self.height):
        #print('num',num)
        #cv2.imshow("disp",mask)
        #cv2.imwrite("../save_img/mask.jpg",mask)
        return "[line:"+str(int(my_angle*kp))+"/]"
        
        
    
