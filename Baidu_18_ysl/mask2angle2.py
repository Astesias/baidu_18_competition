import cv2,os
import numpy as np

def linear_regression(x,y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
    A = np.mat([[N,sumx],[sumx,sumx2]])
    b = np.array([sumy,sumxy])

    return np.linalg.solve(A,b)

global window_size,previous_outputs
window_size = 4
previous_outputs = [0,0,0,0]

def smooth_output(current_output):
    global previous_outputs

    previous_outputs.append(current_output)
    
    if len(previous_outputs) > window_size:
        previous_outputs = previous_outputs[1:]
    
    # smoothed_output = sum(previous_outputs) / len(previous_outputs)
    a,b,c,d=previous_outputs
    smoothed_output=0.2*a+0.2*b+0.4*c+0.2*d
    
    return smoothed_output



def core(img=None,file=None,show=False):
    assert file or isinstance(img,np.ndarray)
    if file:
        img = cv2.imread(file)
    b, g, r = cv2.split(img)
    img = cv2.merge([b,g,r])
    height,width=img.shape[:2]
    
    img=img[:]
    # print(height,width) 480 640
    
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_thd,higher_thd=np.array([40,50,50]),np.array([90,255,255])
    
    mask = cv2.inRange(hsv, lower_thd, higher_thd)
    
    mask[0:200]=0
    #mask[-100:-10]=0
    
    mask = cv2.medianBlur(mask,5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
    
    s= cv2.findContours(mask, cv2.RETR_TREE|cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if os.name!='nt':
      contours=s[1]
    else:
      contours=s[0]

    areas=np.array([])
    rotates=np.array([])
    cxes=[]
    cyes=[]
    all_cxes=[]
    has_left=False
    has_right=False
    target_rotate=None
    
    # del_list=[]
    # for n,contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if area>1000:
    #         contour=np.squeeze(contour,axis=1)
            
    #         rect=cv2.boundingRect(contour)
    #         # center, size, _ = rect
    #         # box = cv2.boxPoints(rect)
    #         # box = np.intp(box)
    #         x, y, w, h = rect
    #         left=x
    #         right=x+w
    #         top=y
    #         bottom=y+h
            
    #         if not (left<5 or bottom>height-5 or right>width-5 or top<5 ):
    #             cv2.drawContours(img, [contour], -1, (255, 255, 255), 2)
    #             del_list.append(n)
    #             continue     
    #         if w>h and not ( bottom>height-5 and top<5):
    #             cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    #             del_list.append(n)
    #             continue   
    #     else:
    #         del_list.append(n)
            
    # contours=del_list_by_index(contours,del_list)
            
            
            
    cnt_contour=len(list(filter(lambda contour:( cv2.contourArea(contour)>300 and cv2.boundingRect(np.squeeze(contour,axis=1))[2]>=30 ),contours)))
    # print(cnt_contour)
    min_right=width
    max_left=0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area>300:
            
            contour=np.squeeze(contour,axis=1)
            
            rect=cv2.boundingRect(contour)
            # center, size, _ = rect
            # box = cv2.boxPoints(rect)
            # box = np.intp(box)
            x, y, w, h = rect
            
            
            left=x
            right=x+w
            top=y
            bottom=y+h
            center=(x+w/2,y+h/2)
            area = cv2.contourArea(contour)
            # density = area / (w * h)
            # print(f'p {density:.2f}',end=' ')
            
            all_cxes.append(x+w/2)
            # print(rect)
            
            # cv2.rectangle(img, (x, y), (right, bottom), (0, 255, 0), 2)
            # x,y=center
            # w,h=size
            
            # print(f'{x:.2f} {y:.2f} {w:.2f} {h:.2f}')
            # left=x-w
            # print(left)
            
            # print(f'area {area}',end=' ')
            # print(h,w)
            # print(x,y)

            
            if width/2-60<x+w/2<width/2+60:
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
            #     cv2.drawContours(img, [contour], -1, (255, 255, 255), 2) # ²»½Ó±ß
            #     continue     
            # if (w>h and not ( bottom>height-5 or top<5)) and not area>15000 :
            #     cv2.drawContours(img, [contour], -1, (0, 0, 0), 2)       #  w>h
            #     continue   
            if w<30:
                cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)
                continue
            # if not cv2.isContourConvex(contour):
            #     cv2.drawContours(img, [contour], -1, (102, 204, 255), 2)
            #     continue
            # if cnt_contour>=4 and w>h:
            #     cv2.drawContours(img, [contour], -1, (102, 204, 255), 2)
            #     continue
            
            
            cv2.drawContours(img, [contour], -1, (255, 255, 0), 2)
            
            rotate=linear_regression(contour[:,1],contour[:,0])[1]
            rotate=-np.arctan(rotate)*180/np.pi
            # print(f'A {rotate:.0f}',end=' ')
            
            # midx,midy=center
            # p1=int(midx-50*sin(rotate/180*pi)),int(midy+50*cos(rotate/180*pi))
            # p2=int(midx+50*sin(rotate/180*pi)),int(midy-50*cos(rotate/180*pi))
            
            # cv2.line(img,p2,p1,(125,25,25),4,3)
            
            
            lr=None
            if w>h:
                if rotate<0:
                    has_right=True
                    lr=0
                else:
                    has_left=True
                    lr=1
            else:
                if x+w/2<width/2:
                    has_left=True
                    lr=1
                else:
                    has_right=True
                    lr=0
            # drewlinecross(img,int(x+w/2))

            cv2.rectangle(img, (x, y), (right, bottom), (0, 255, 0) if lr else (0,255,255) , 2)
             
            
            cxes.append(x+w/2)
            cyes.append(y+h/2)

            
            rotates=np.append(rotates,rotate)
            areas=np.append(areas,area)
            
            if bottom>height/4*3:
                if lr==1:
                    if max_left<right:
                        max_left=right
                else:
                    if min_right>left:
                        min_right=left
            
    print(has_left,has_right,end=' ')  

    '''
    if max_left>width/4*1:
        target_rotate=20
    elif min_right<width/3*2:
        target_rotate=-20
    '''
        
    cxavg=cxavg_half=0
    # if not target_rotate:

    cxavg=sum(cxes)/len(cxes) if cxes else 0
    
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
        print(f'half: {cxavg_half:.0f} ',end='')
        # drewlinecross(img,int((hr/cr+hl/cl)/2),color=(200,100,50))
        
    print(f'cxavg: {cxavg:.0f} ',end='')
    # drewlinecross(img,int(cxavg),color=(100,100,100))
    
    cntl=cntr=0
    rl=rr=0
    flag=False
    for i in rotates:
        if i<0:
            rl+=i
            cntl+=1
        else:
            rr+=i
            cntr+=1
    if cntr and cntl and abs(cntr)>10 and abs(cntl)>10:
        if cntr*cntl>0:
            flag=True
        
    print(f'cxavg: {cxavg:.0f} ',end='')
    # drewlinecross(img,int(cxavg),color=(100,100,100))
        
    # else:
    #     rotate=target_rotate

    if not has_left and not has_right:
        cxavg=sum(all_cxes)/len(all_cxes) if all_cxes else 0
        if cxavg<width/2:
            has_left=True
        else:
            has_right=True
            
    k=areas/sum(areas)      
    rotate=sum(rotates*k)    
    print(rotate)
    if not has_left:
        rotate-=(width-cxavg)/320*30
        mode=1
    elif not has_right:
        rotate+=cxavg/320*30
        mode=2
    else:
        # print(rotate,1)
        rotate-=(cxavg_half)/4
        # print(rotate,2)
        mode=3

    

    # if abs(rotate)>60:
    #     rotate = 60 if rotate>0 else -60
        
    if rotate<0:
      rotate*=1.5
    else:
      rotate/=1.1
        
    if target_rotate:
        rotate=target_rotate
        target_rotate=None
        mode=4
    else:
        rotate=smooth_output(rotate)
        pass
        
    print(f'mode {mode} ',end='')
    
    
    # midx,midy=int(width/2),int(height/2)
    # p1=int(midx-50*sin(rotate/180*pi)),int(midy+50*cos(rotate/180*pi))
    # p2=int(midx+50*sin(rotate/180*pi)),int(midy-50*cos(rotate/180*pi))
    
    # cv2.line(img,p2,p1,(255,200,100),4,3)
    # cv2.putText(img,'{:^5.1f}'.format(abs(rotate)),(midx+20,midy),
    #             cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255) if rotate>0 else (255,0,255))
    # if show:
    #     cv2.imshow('1',img)
    #     cv2.waitKey(0)
    
        
    return rotate


















