import cv2,os,time
import numpy as np
from pysl import drewlinecross,flatten
from numpy import pi,cos,sin

global HEIGHT,WIDTH,DROP_TOP,DROP_BOTTOM            # 输入尺寸
global PIECE_LENGTH,PIECE_NUM                       # 切片大小、数量
global WINDOW_SIZE,WINDOW_WEIGHT,WINDOW_PREVIOUS    # 平滑窗口设置
global MODE_HSV,MODE_EXG,MODE_SEARCH_GREEN          # 颜色提取模式
global LOWER_THD,HIGHT_THD                          # HSV色域
global FPS,SKIP                                     # 视频提帧设置
global MODE_IMG,MODE_AUDIO,MODE_AUDIO_AUTO          # 视频播放设置
global LOG_ENABLE                                   # LOG输出
global FRAME_DRAW_ENABLE                            # 图像标注
global SCALE

SCALE = 2
HEIGHT,WIDTH = 480//SCALE,640//SCALE


DROP_TOP,DROP_BOTTOM = 0,20

PIECE_LENGTH = HEIGHT//SCALE
PIECE_NUM = HEIGHT//PIECE_LENGTH

WINDOW_SIZE = 5
WINDOW_WEIGHT  = [0.05,0.05,0.1,0.10,0.7]
WINDOW_PREVIOUS = [0 for _ in range(WINDOW_SIZE)]

MODE_HSV = 0
MODE_EXG = 1
MODE_SEARCH_GREEN = MODE_EXG
LOWER_THD,HIGHT_THD=np.array([40,50,50]),np.array([90,255,255])

FPS = 50
SKIP = 60
MODE_IMG  = 1
MODE_AUDIO = 2
MODE_AUDIO_AUTO = 3

LOG_ENABLE = True
FRAME_DRAW_ENABLE = True

display_mode=MODE_AUDIO_AUTO

BLACK = (0, 0, 0)        # 黑
WHITE = (255, 255, 255)  # 白
RED = (0, 0, 255)        # 红
GREEN = (0, 255, 0)      # 绿
BLUE = (255, 0, 0)       # 蓝
YELLOW = (0, 255, 255)   # 黄
CYAN = (255, 255, 0)     # 青
MAGENTA = (255, 0, 255)  # 品红
GRAY = (128, 128, 128)   # 灰
ORANGE = (0, 165, 255)   # 橙
PURPLE = (128, 0, 128)   # 紫
BROWN = (42, 42, 165)    # 棕
PINK = (203, 192, 255)   # 粉
GOLD = (0, 215, 255)     # 金
SILVER = (192, 192, 192) # 银


def log(*msgs,**kws):
    if LOG_ENABLE:
        print(*msgs,**kws)

def smooth_output(current_output):
    global WINDOW_PREVIOUS
    WINDOW_PREVIOUS.append(current_output)
    if len(WINDOW_PREVIOUS) > WINDOW_SIZE:
        WINDOW_PREVIOUS = WINDOW_PREVIOUS[1:]
    smoothed_output = 0
    for w,v in zip(WINDOW_WEIGHT,WINDOW_PREVIOUS):
        smoothed_output+=w*v
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

def remove_noise_from_mask(mask, min_size):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # 创建一个与原始掩码相同尺寸的新掩码
    filtered_mask = np.zeros_like(mask)
    # 根据连通组件的大小过滤噪点
    for label in range(1, num_labels):
        # 获取连通组件的大小
        area = stats[label, cv2.CC_STAT_AREA]
        # 如果连通组件的大小超过指定的最小尺寸，则保留该连通组件
        if area >= min_size:
            filtered_mask[labels == label] = 255
    return filtered_mask

def morphologyEx(mask,close_sz=3,open_sz=3,shape_sz=3):
    if open_sz:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,        np.ones((close_sz, close_sz))) # 去除小点
    if close_sz:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,       np.ones((open_sz, open_sz)))   # 填充图像
    if shape_sz:
        mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,    np.ones((shape_sz, shape_sz))) # 形态梯度
    return mask

def get_mask_ExG(image):
    image = image.astype(np.float32) 
    image /= 255.0
    # 提取各个通道
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]
    
    exg = 2 * green_channel - red_channel - blue_channel
    exg_min = np.min(exg)
    exg_max = np.max(exg)
    exg_normalized = (exg - exg_min) / (exg_max - exg_min)
    exg_channel = (exg_normalized * 255).astype(np.uint8)

    
    # 大津法
    _, otsu_threshold = cv2.threshold(exg_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_threshold = remove_noise_from_mask(otsu_threshold,500)
    otsu_threshold = morphologyEx(otsu_threshold,3,11,0)

    return otsu_threshold

def get_mask_HSV(image):
    global LOWER_THD,HIGHT_THD
    hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,LOWER_THD,HIGHT_THD)
    mask = cv2.medianBlur(mask,5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
    return mask
    
def find_contours(mask):
    s= cv2.findContours(mask,cv2.RETR_EXTERNAL|cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if os.name!='nt':
        contours=s[1]
    else:
        contours=s[0]
    return contours
    
def draw_rotate(img,center,rotate,color,thick=2,length=50):
    global FRAME_DRAW_ENABLE
    if FRAME_DRAW_ENABLE:
        length = length//SCALE
        midx,midy=center
        p1=int(midx-length*sin(rotate/180*pi)),int(midy+length*cos(rotate/180*pi))
        p2=int(midx+length*sin(rotate/180*pi)),int(midy-length*cos(rotate/180*pi))
        cv2.line(img,p2,p1,color,thick//SCALE)
        
def draw_text(img,text,position,color):
    if FRAME_DRAW_ENABLE:
        cv2.putText(img, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8/SCALE, color, thickness = 2//SCALE)
        
def is_center_on_contour(contour,center):
    result = cv2.pointPolygonTest(contour,center, False)
    return result>=0
        
def is_centroid_on_contour(contour,rect):
    x, y, w, h = rect
    corner1 = (x, y)
    corner2 = (x + w, y)
    corner3 = (x, y + h)
    corner4 = (x + w, y + h)
    center_x = (corner1[0] + corner2[0] + corner3[0] + corner4[0]) / 4
    center_y = (corner1[1] + corner2[1] + corner3[1] + corner4[1]) / 4
    result = cv2.pointPolygonTest(contour, (center_x, center_y), False)
    return result>=0

def count_rect_point_isin_contour(contour,rect2):
    corners = cv2.boxPoints(rect2)
    corners = np.intp(corners)
    cnt=0
    for corner in corners:
        cx,cy=corner
        corner = (cx,cy)
        print(corner)
        if is_center_on_contour(contour,corner):
            cnt+=1
    return cnt

def count_poly(contour,kep=0.03):
    epsilon = kep * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return len(approx)
        
def avg(l):
    if l:
        return sum(l)/len(l)

def core(_img=None,file=None,debug=False):
    global LOG_ENABLE,FRAME_DRAW_ENABLE
    FRAME_DRAW_ENABLE=debug
    
    assert file or isinstance(_img,np.ndarray)
    if file:
        _img = cv2.imread(file)
    b, g, r = cv2.split(_img)
    
    _img = cv2.merge([b,g,r])
    _img=_img[:-DROP_BOTTOM]
    _img=_img[DROP_TOP:]
    
    img=_img.copy()
    img_draw=_img.copy()

    
    if MODE_SEARCH_GREEN == MODE_HSV:
        mask = get_mask_HSV(img)
    else:
        mask = get_mask_ExG(img)
    contours = find_contours(mask)
    cv2.drawContours(img_draw, contours, -1, WHITE, 8//SCALE)
    
    areas=np.array([])      # 各目标面积
    rotates=np.array([])    # 各目标旋转角度
    cxes=[[],[]]            # 各目标中点x坐标
    all_cxes=[]             # 各目标中点x坐标 未筛选
    has_left=False          # 存在左边
    has_right=False         # 存在右边
    target_rotate=None      # 固定设置角度
    sum_contour=0
    
    for pn in range(PIECE_NUM):
        img_draw_piece = img_draw[-(pn+1)*PIECE_LENGTH-1:-(pn)*PIECE_LENGTH-1]
        mask_piece     = mask[-(pn+1)*PIECE_LENGTH-1:-(pn)*PIECE_LENGTH-1]
            
        contours = find_contours(mask_piece)
        filter_contours = list(filter(lambda x:cv2.contourArea(x)>300,contours))
        filter_contours = list(map(lambda x:np.squeeze(x,axis=1),filter_contours))
        filter_contours_cx = list(map(lambda x:cv2.minAreaRect(x)[0][0],filter_contours))

        for index,contour in enumerate(filter_contours):
            area = cv2.contourArea(contour)
            rect=cv2.boundingRect(contour) 
            x,y,w,h = rect
            
            
            rect2=cv2.minAreaRect(contour)#x+w/2,y+h/2
            cx,cy=rect2[0]
            __w, __h = rect2[1]
            area2 = __w*__h
            
            center=(int(cx),int(cy))
            density = area / area2
            
            if pn==PIECE_NUM-1:
                all_cxes.append(cx)
            
            if 1:
                rotate=linear_regression(contour[:,1],contour[:,0])[1]
                rotate=-np.arctan(rotate)*180/np.pi
            else:
                rotate=cv2.minAreaRect(contour)[2]
                if rotate < -45:
                    rotate += 90
            
            
            
            # filters
            if h<30:
                cv2.drawContours(img_draw_piece, [contour], -1, BLACK, thickness=cv2.FILLED) # 长度过滤
                continue
        
            if WIDTH/2-WIDTH/640*60<cx<WIDTH/2+WIDTH/640*60 and w<WIDTH/640*120: 
                 cv2.drawContours(img_draw_piece, [contour], -1, RED, thickness=cv2.FILLED) # 打靶过滤
                 continue
             
            if len(filter_contours)>2 and cx!=max(filter_contours_cx) \
                                      and cx!=min(filter_contours_cx): # 中色块过滤
                cv2.drawContours(img_draw_piece, [contour], -1, WHITE, thickness=cv2.FILLED)
                # draw_text(img_draw_piece,str(int(index)),center, BLACK)
                # target_rotate = 0
                continue
            
            lr=0
            if w<WIDTH/2:
                if cx<WIDTH/2:
                    has_left=True
                    lr=-1
                    cxes[0].append(cx)
                    cv2.drawContours(img_draw_piece, [contour], -1, BLUE, 2)
                else:
                    has_right=True
                    lr=1  
                    cxes[1].append(cx)
                    cv2.drawContours(img_draw_piece, [contour], -1, PINK, 2)
            else:
                if rotate>0:
                    lr=-1
                else:
                    lr=1
                
            if count_poly(contour,0.03)==3 and w>20: 
                cv2.drawContours(img_draw_piece, [contour], -1, PINK, thickness=cv2.FILLED) # 三角过滤
                continue                
                
            if  density<0.5 and w>30:  #  ( not is_center_on_contour(contour,center) and w>60):
                cv2.drawContours(img_draw_piece, [contour], -1, GOLD, thickness=cv2.FILLED) # 十字质心过滤
                #print(rotate)
                if lr==-1:
                    rotate=7
                else:
                    rotate=-15
            if lr==1 and rotate>0 and w<WIDTH/3 and h<HEIGHT/3:
                rotate=-rotate
            if lr==-1 and rotate<0 and w<WIDTH/3 and h<HEIGHT/3:
                rotate=1    
                       
            draw_text(img_draw_piece,str(round(density,2)),center, BLACK)
                
            sum_contour+=1
            rotates=np.append(rotates,rotate)
            draw_rotate(img_draw_piece,center,rotate,PURPLE,4//SCALE)
            areas=np.append(areas,area)

            
            
            
    
    draw_text(img_draw,'LBORDER',(50,40)      ,                 ORANGE if has_left else GRAY)
    draw_text(img_draw,'RBORDER',(WIDTH-50-120//SCALE,40),             ORANGE if has_right else GRAY)
    # draw_text(img_draw,str(sum_contour),(WIDTH//2,HEIGHT//2),   ORANGE if sum_contour<7 else GRAY)
    
    # if has_left and has_right:
        
    cx_lr_avg = cx_avg = WIDTH/2
    cx_lr_avg_err = cx_avg_err = 0 
    if (not has_left and not has_right):# or ( has_left and has_right)  :
        cx_avg=sum(all_cxes)/len(all_cxes) if all_cxes else WIDTH/2
        if cx_avg<WIDTH/2:
            has_left=True
        elif cx_avg>WIDTH/2:
            has_right=True
    else:
        
        if cxes[0] or cxes[1]:
            if cxes[0] and cxes[1]:
                cx_lr_avg=avg(cxes[0])+avg(cxes[1])/2
                cx_lr_avg_err=-(WIDTH/2-cx_lr_avg)
                drewlinecross(img_draw,int(cx_lr_avg),color=SILVER)
                draw_text(img_draw,"err "+str(int(cx_lr_avg_err)),(WIDTH//2-20,50),BLUE)
            cx_avg=avg(flatten(cxes))
            drewlinecross(img_draw,int(cx_avg),color=GOLD)
            
    cx_lr_avg_err -= 10  
    cx_avg_err = abs(WIDTH/2-(cx_avg))
    
    k=areas/sum(areas)      
    Rotate=sum(rotates*k)

    LOG_ENABLE = False 
    
    
    #print(round(Rotate,1),end=' ')
    log('cx_err',round(cx_avg_err,2),'cx_lr_err',round(cx_lr_avg_err,2),'rotate',round(Rotate,2),end=' ')
    
    cx_avg_err=0
    if not has_left:
        cx_avg_err =WIDTH- cx_avg
    
        Rotate /=1.15
        Rotate-=cx_avg_err/160*30      /0.9  /2
        Rotate-=3
        mode=1
    elif not has_right:
        cx_avg_err =WIDTH/2-( WIDTH/2-cx_avg )
    
        Rotate /=0.95
        Rotate+=cx_avg_err/160*30    /1.1  /2
        Rotate+=3
        mode=2
    else:
        Rotate=Rotate+(cx_lr_avg_err)/6
        mode=3     
    #print(round(cx_avg_err),mode)
               
    log('r2',round(Rotate,2),'mode',mode)

    #Rotate = smooth_output(Rotate)
    Rotate /= 2
        
    if target_rotate!=None:       
        Rotate = target_rotate
    

    midx,midy=WIDTH//2,HEIGHT//2
    draw_rotate(img_draw,(midx,midy),Rotate,(200,255,200),4//SCALE)
    cv2.putText(img_draw,'{:^5.1f}'.format(abs(Rotate)),(midx+20,midy),
                cv2.FONT_HERSHEY_COMPLEX,1/SCALE,(255,255,255) if Rotate>0 else (255,0,255)) # total rotate
        
        
        
    if not debug:
        return Rotate
    return img_draw
    

    
    
if __name__ == '__main__':
    
    cap=cv2.VideoCapture('../NewVideo/latest.avi')
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(SKIP * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    
    cnt = 0
    t=time.time()
    
    while 1:
        _,frame=cap.read()
        frame = cv2.resize(frame,(WIDTH,HEIGHT))
        
        frame=core(_img=frame,debug=True)
        
        
        cnt+=1
        if time.time()-t>1:
            #print('fps',cnt)
            cnt=0
            t=time.time()
        
        frame = cv2.resize(frame,(640,480))
        cv2.imshow('frame',frame)
        
        if 1:
            key = cv2.waitKey(24) & 0xFF
            if key== ord('q') or not _:
                cv2.destroyAllWindows()
                break
            elif key == ord(' '):
                while not (cv2.waitKey(FPS) & 0xFF == ord(' ')):
                    pass
        else:
            cv2.waitKey(0)
            cv2.destroyAllWindows()






