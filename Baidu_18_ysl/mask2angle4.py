# -*- coding: utf-8 -*-

import cv2,os
import numpy as np
from pysl import drewlinecross
from numpy import pi,cos,sin

dir='../NewVideo/output'
#files=[os.path.join(dir,i) for i in os.listdir(dir) if '.jpg' in i]

#file=files[111]  
# file='../tmp5.jpg'
#file='../00006.jpg'

global window_size,previous_outputs,lower_thd,higher_thd,cut_piece,piece,height,width,display

height,width=480,640

piece=150
cut_piece=height//piece

window_size = 4
previous_outputs = [0,0,0,0]
lower_thd,higher_thd=np.array([40,50,50]),np.array([90,255,255])

display=True
display_mode=3

def printf(*args,**kws):
    if display:
        print(*args,**kws,end=' ')

def smooth_output(current_output):
    global previous_outputs
    previous_outputs.append(current_output)
    if len(previous_outputs) > window_size:
        previous_outputs = previous_outputs[1:]
    a,b,c,d=previous_outputs
    smoothed_output=0.1*a+0.1*b+0.1*c+0.7*d
    return smoothed_output

def linear_regression(x,y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
    A = np.mat([[N,sumx],[sumx,sumx2]])
    b = np.array([sumy,sumxy])

    return np.linalg.solve(A,b)

def get_contours(mask):
    mask = cv2.medianBlur(mask,5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
    s= cv2.findContours(mask, cv2.RETR_TREE|cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if os.name!='nt':
      contours=s[1]
    else:
      contours=s[0]
    return contours

def draw_rotate(img,center,rotate,color,thick=2,length=50):
    midx,midy=center
    p1=int(midx-length*sin(rotate/180*pi)),int(midy+length*cos(rotate/180*pi))
    p2=int(midx+length*sin(rotate/180*pi)),int(midy-length*cos(rotate/180*pi))
    cv2.line(img,p2,p1,color,thick)

def core(img=None,file=None,show=False,debug=False):
    assert file or isinstance(img,np.ndarray)
    if file:
        img = cv2.imread(file)
    b, g, r = cv2.split(img)
    img = cv2.merge([b,g,r])
    
    img_=img.copy()
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv[0:30]=0
    hsv[-100:]=0
    
    areas=np.array([])      # 各目标面积
    rotates=np.array([])    # 各目标旋转角度
    cxes=[]                 # 各目标中点x坐标
    all_cxes=[]             # 各目标中点x坐标 未筛选
    has_left=False          # 存在左边
    has_right=False         # 存在右边
    target_rotate=None      # 固定角度
    # last_top=last_lr=None
    
    box_num=0
    for ct in range(cut_piece):
        mask = cv2.inRange(hsv[-(ct+1)*piece-1:-(ct)*piece-1], lower_thd, higher_thd)
        contours=get_contours(mask)
        img=img_[-(ct+1)*piece-1:-(ct)*piece-1] # image refer
        
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area>300:
                
                contour=np.squeeze(contour,axis=1)
                
                rect=cv2.boundingRect(contour) 
                # rect_ = cv2.minAreaRect(contour)
                # cv2.drawContours(img_, [np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))],0,(0, 255, 0), 2)             
                
                x,y,w,h = rect
                left,left2border=x,x
                right,right2border=x+w,width-x-w
                top,top2border=y,y
                bottom,bottom2border=y+h,height-y-h
                
                
                cx,cy=x+w/2,y+h/2
                center=(cx,cy)
                density = area / (w * h)
                
                all_cxes.append(x+w/2)
    
                
                if width/2-60<x+w/2<width/2+60 and w<120:
                     cv2.drawContours(img, [contour], -1, (255, 255, 255), 2)
                     continue
                # if area>28000: 
                #     cv2.drawContours(img, [contour], -1, (255, 0, 255), 2)
                #     if x>width/2:
                #         has_left=False
                #         has_right=True
                #         target_rotate=-40
                #     continue                    
                # if not (left<5 or bottom>height-5 or right>width-5 or top<5 ) and not area>10000:
                #     cv2.drawContours(img, [contour], -1, (255, 255, 255), 2) # 不接边
                #     continue     
                # if (w>h*1.5 and not ( bottom2border<5 or left2border<5 or right2border<5 )):
                #     cv2.drawContours(img, [contour], -1, (0, 0, 0), 2)       #  w>h
                #     continue   
                # if w<30 or h<50 or density<0.1:
                #     cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)
                #     continue
                #if w>150:
                #    cv2.drawContours(img, [contour], -1, (0, 0, 0), 2)
                #    continue
                # if not cv2.isContourConvex(contour):
                #     cv2.drawContours(img, [contour], -1, (102, 204, 255), 2)
                #     continue
                # if cnt_contour>=4 and w>h:
                #     cv2.drawContours(img, [contour], -1, (102, 204, 255), 2)
                #     continue
                if  w>h:
                    cv2.drawContours(img, [contour], -1, (0, 0, 0), 2)
                    continue
                    pass

              
           
    
                rotate=linear_regression(contour[:,1],contour[:,0])[1]
                rotate=-np.arctan(rotate)*180/np.pi
                #printf(f'A {rotate:.0f}')
             
                
                lr=None
                # if w>h:
                #     if rotate<0:
                #         has_right=True
                #         lr=0
                #     else:
                #         has_left=True
                #         lr=1
                # else:
                if 1:
                    if x+w/2<width/2:
                        has_left=True
                        lr=0
                    else:
                        has_right=True
                        lr=1
                # if last_top and abs(bottom-last_top)<=10:
                #     lr=last_lr
                #     if last_lr:
                #         has_right=True
                #     else:
                #         has_left=True

                box_num+=1
                 
                cxes.append(x+w/2)
                rotates=np.append(rotates,rotate)
                areas=np.append(areas,area)
                
                # last_top=top
                # last_lr=lr
            


    cxavg=cxavg_half=0 # 平均中线 左右平均中线
    if not target_rotate:

        cxavg=sum(cxes)/len(cxes) if cxes else 0
        cxavg+=0
        
        hl=hr=0
        cl=cr=0
        for i in cxes:
            if i<width/2:
                hl+=i
                cl+=1
            else:
                hr+=i
                cr+=1
        if cl and cr:
            cxavg_half=(width-hr/cr)-hl/cl
            cxavg_half-=20                       ##^^^^^^^

     
    
        if not has_left and not has_right:

            cxavg=sum(all_cxes)/len(all_cxes) if all_cxes else 0
            if cxavg<width/2:
                has_left=True
            else:
                has_right=True
                
        k=areas/sum(areas)      
        rotate=sum(rotates*k)    

        if not has_left:
            rotate-=(width-cxavg)/320*30      /1
            mode=1
        elif not has_right:
            rotate+=(width/2-cxavg)/320*30    /1.5
            mode=2
        else:
            # printf(rotate,1)
            rotate=rotate-(cxavg_half)/10-3 #*(box_num/4)
            # printf(rotate,2) 
            mode=3
            
        
        rotate=smooth_output(rotate)
        
    else:
        rotate=target_rotate
        target_rotate=None
        mode=4

    #print('box_num',box_num)
    if debug:
      return img_
    if mode!=2:
      rotate= rotate/1.4
    else:
      rotate= rotate/1.2
    return rotate


    
if __name__ == '__main__':
    
    display=True
    display_mode=2 # 1 img 2 video 3 video auto
    
    # for file in files[:]:
    #     r=core(file=file,show=True)
    #     printf(f'R {r:.0f} '+('->' if r>0 else '<-'))
    
    # r=core(file=file,show=True)
    # printf(f'A {r:.0f} '+('->' if r>0 else '<-'))
    
    cap=cv2.VideoCapture('../NewVideo/3.avi')
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(22 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    while 1:
        _,frame=cap.read()
        r=core(img=frame,show=True)
        printf(f'A {r:.0f} '+('->' if r>0 else '<-'))
    
