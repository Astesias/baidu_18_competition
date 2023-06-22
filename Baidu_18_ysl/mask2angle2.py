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

def core(img,resize=None,mask=None,show=False):
    # img = cv2.imread(file)
    img=img[10:-10,::,:]
    # b, g, r = cv2.split(img)
    # img = cv2.merge([b,g,r])
    height,width=img.shape[:2]
    if resize:
        if not isinstance(resize,(list,tuple)):
            img=cv2.resize(img,(resize,resize))
        else:
            img=cv2.resize(img,tuple(resize))
            
    if not mask:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([60, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.medianBlur(mask, 5) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) 
        
    s= cv2.findContours(mask, cv2.RETR_TREE|cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if os.name!='nt':
      contours=s[1]
    else:
      contours=s[0]
      
    areas=np.array([])
    rotates=np.array([])
    cxes=[]
    has_left=False
    has_right=False
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area>200:
            
            contour=np.squeeze(contour,axis=1)
            
            center, size, _ = cv2.minAreaRect(contour)
            x,y=center
            w,h=size
            
            # print(h,w)
            # print(x,y)

            
            if width/2-60<x<width/2+60 and max(h,w)/2<120:
                cv2.drawContours(img, [contour], -1, (255, 255, 255), 2)
                continue
            if width/2-60<x<width/2+60 and max(h,w)/2<120:
                cv2.drawContours(img, [contour], -1, (255, 255, 255), 2)
                continue         
            
            # drewlinecross(img,int(x+min(h,w)))
            # drewlinecross(img,int(x))
            cv2.drawContours(img, [contour], -1, (255, 255, 0), 2)
            
            rotate=linear_regression(contour[:,1],contour[:,0])[1]
            rotate=-np.arctan(rotate)*180/np.pi
            # print(rotate)
            
            if w>width*5/8:
                if rotate<0:
                    has_right=True
                else:
                    has_left=True
  
            elif x<width/2:
                has_left=True
            elif x+max(h,w)/2>width*7/8:
                has_right=True

            
            
            cxes.append(x)
            rotates=np.append(rotates,rotate)
            areas=np.append(areas,area)
            
    
    if show:
        print(has_left,has_right,end=' ')  
        # easy_show_img(img)
        pass
    cxavg=sum(cxes)/len(cxes) if cxes else 0
    
    hl=hr=0
    cl=cr=0
    cxavg_half=0
    for i in cxes:
        if i<width/2:
            hl+=i
            cl+=1
        else:
            hr+=i
            cr+=1
    if cl and cr:
        cxavg_half=(width-hr/cr)-hl/cl
        if show:
            print(f'half: {cxavg_half:.0f} ',end='')
    if show: 
        print(f'cxavg: {cxavg:.0f} ',end='')
    if not has_left:
        rotate=-(width-cxavg)*90/(width/3)
    elif not has_right:
        rotate=cxavg*90/(width/3)
    else:
        k=areas/sum(areas)
        rotate=sum(rotates*k) 
        rotate-=(cxavg_half)/6
    rotate/= 3
    # if abs(rotate)>60:
    #     rotate = 60 if rotate>0 else -60
    return rotate
















