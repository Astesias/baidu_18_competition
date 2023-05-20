import cv2
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

def core(img,resize=None,mask=None):
    # img = cv2.imread(file)
    img=img[10:-10,::,:]
    # b, g, r = cv2.split(img)
    # img = cv2.merge([b,g,r])
    
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
    
    mymask=np.zeros_like(mask)
    areas=np.array([])
    lrs=np.array([])
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE|cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > 50:
            
            cnt=np.squeeze(cnt,axis=1)
            mymask[cnt[:,1],cnt[:,0]]=255
            lr=linear_regression(cnt[:,1],cnt[:,0])[1]
            lr=-np.arctan(lr)*180/np.pi
            
            lrs=np.append(lrs,lr)
            areas=np.append(areas,area)
         
        
    k=areas/sum(areas)
    return sum(lrs*k)















