import cv2
import numpy as np
import math
import itertools
import time
import math
import matplotlib.pyplot as plt
from bluetooth import *
import sys
import os


socket= BluetoothSocket( RFCOMM )

socket.connect(("98:DA:60:01:7A:7F",1))
print('bluetooth connected!')

cap = cv2.VideoCapture('car_test_3.mp4')

#trap_bottom_width=3.4#사다리꼴 아래쪽 가장자리 너비 계산 위한 백분율
#trap_top_width=0.8 # 사다리꼴 위쪽 가장자리 너비 계산을 위한 백분율
#trap_height=0.5# 사다리꼴 높이 계산을 위한 백분율

#관심 영역 계산 시 사용
#car_test_4
trap_bottom_width=1.#사다리꼴 아래쪽 가장자리 너비 계산 위한 백분율
trap_top_width=1.7 # 사다리꼴 위쪽 가장자리 너비 계산을 위한 백분율
trap_height=0.65# 사다리꼴 높이 계산을 위한 백분율

#좌우영역 구분 시 사용
slope_thresh_high=5.0
#slope_thresh_low =0.8 # 기울기 기준
slope_thresh_low =0.2
slopes = []
selected_lines = []
right_lines = []
left_lines = []
left_detect=0
right_detect=0

left_pts=[]
right_pts=[]

right_m=2
right_b=[2,2]
left_m=1
left_b=[1,1]

right_ini_x=0
right_fni_x=0
poly_points=[]

net = cv2.dnn.readNet("yolov3-3.weights", "yolov3-3.cfg")
# YOLO NETWORK 재구성
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
flag=0
list_ball_location=[]

def matching(img_color):
    max = 0
    for i in os.listdir('./dataset/'):
        path = './dataset/' + i
        src = cv2.imread(path, cv2.IMREAD_COLOR)  # dataset
        src = cv2.resize(src, (800, 800))
        if img_color is None or src is None:
            print('Image load failed!')
            sys.exit()

        feature = cv2.ORB_create()
        #sift = cv.SIFT_create()
        # 특징점 검출 및 기술자 계산
        kp1, desc1 = feature.detectAndCompute(img_color, None)
        kp2, desc2 = feature.detectAndCompute(src, None)
        '''
        # 특징점 매칭(shift)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches1 = flann.knnMatch(desc1, desc2, k=2)
        '''
        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        matches1 = matcher.knnMatch(desc1, desc2, 2)  # knnMatch로 특징점 2개 검출
        good_matches1 = []
        # ratio=0.25
        for m in matches1:  # matches는 두개의 리스트로 구성
            if m[0].distance / m[1].distance < 0.7:  # 임계점 0.7
                good_matches1.append(m)  # 저장
                #print(good_matches1)

        # print('# of good_matches:', len(good_matches1))
        if (len(good_matches1) > max):
            image = src
            max = len(good_matches1)
            max_good_matches = good_matches1
            kp_max = kp2
            max_path = path
        if 'ternel' in max_path:
            socket.send('ternel')
        elif 'down' in max_path:
            socket.send('down')
        elif 'up' in max_path:
            socket.send('up')
        elif 'parking' in max_path:
            socket.send('parking')
        elif 'curve' in max_path:
            socket.send('curve')
        elif 'rotary' in max_path:
            socket.send('rotary')
            time.sleep(8)
    print(max_path)

def filter_colors(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white=(170,170,170)
    upper_white=(255,255,255)

    lower_yellow = (0, 60, 60)
    upper_yellow = (40, 180, 180)

    img_mask_white = cv2.inRange(img, lower_white, upper_white)
    img_mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 팽창 연산 수행 (원본 배열, 연산 방법, 구조 요소, 고정점, 반복 횟수)
    img_mask_yellow= cv2.morphologyEx(img_mask_yellow, cv2.MORPH_DILATE, kernel, iterations=3)
    img_mask_white= cv2.morphologyEx(img_mask_white, cv2.MORPH_DILATE, kernel, iterations=3)

    img_mask = cv2.addWeighted(img_mask_white, 0.0, img_mask_yellow, 1.0, 0.0)
    # 구조 요소 생성 (커널의 형태[직사각형],커널의 크기)

    #cv2.imshow('mask_yellow', img_mask_yellow)
    #cv2.imshow('mask_white', img_mask_white)
    #cv2.imshow('mask', img_mask)
    #cv2.imshow('Result', img)
    return img_mask



def limit_region(canny):#자동차의 진행 방향 바닥에 존재하는 차선만을 검출
    h,w=canny.shape
    h=int(h)
    w=int(w)
    mask=np.zeros((h,w),np.uint8)
    points=np.array([[int(w*(1-trap_bottom_width)/2),h],[int(w*(1-trap_top_width)/2),int(h-h*trap_height)],[int(w-(w*(1-trap_top_width)/2)),int(h-h*trap_height)],[int(w-(w*(1-trap_bottom_width))/2),h]])
    blue=(255,0,0)
    mask=cv2.fillConvexPoly(mask,points,blue)#mask에 영역 적기
    roi=cv2.bitwise_and(canny,mask)
    #cv2.imshow('mask',mask)
    roi_mask = cv2.addWeighted(mask, 0.5, canny, 0.5, 0.0)#관심영역
    cv2.imshow('roi_mask',roi_mask)
    return roi

def Hough(roi):#허프 변환
    lines=cv2.HoughLinesP(roi,1,np.pi/180,10,None,80,20)#직선 좌표 저장
    #최대 길이, 최소 간격
    for line in lines:
        x1,y1,x2,y2=line[0]
        cv2.line(roi,(x1,y1),(x2,y2),(120,120,120),10)
    return roi,lines
def seperate_line(hough,image,lines):#추출한 직선 성분으로 좌우 차선에 있을 가능성이 있는 직선만 뽑아서 좌우 각각 하나의 직선으로 검출
    slopes=[]
    selected_lines=[]
    right_lines=[]
    left_lines=[]
    right_detect=0
    left_detect=0
    h, w = hough.shape
    global img
    for line in lines:
        x1,y1,x2,y2=line[0]
        slope=(y2-y1)/(x2-x1)#기울기 계산
        try:
            if abs(slope) > slope_thresh_low and abs(slope) < slope_thresh_high:#기울기가 수평인 선 제외
                slopes.append(slope)
                selected_lines.append(line)
                #img = cv2.line(img, (x1,y1), (x2,y2), (0, 120, 255), 10, cv2.LINE_AA) #주황
        except ZeroDivisionError:
            print('Zero')
    #좌우선으로 분류
    img_center=w/2
    l=len(selected_lines)
    for i in range(0,l):
       # print(selected_lines[i][0])
        x3,y3,x4,y4=selected_lines[i][0]
        #image = cv2.line(image, (x3,y3), (x4,y4), (0, 0, 255), 5, cv2.LINE_AA)
        if (slopes[i]>0) and (x4>=img_center)and(x3>=img_center):
            right_lines.append(selected_lines[i][0])
            right_detect=1
            #x1, y1, x2, y2 = selected_lines[i][0]
            #img = cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 10, cv2.LINE_AA)
        if slopes[i] < 0 and x4 <= img_center and x3 <= img_center:
            left_lines.append(selected_lines[i][0])
            x1, y1, x2, y2 = selected_lines[i][0]
            #img = cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 10, cv2.LINE_AA)
            #left_detect=1
            left_detect = 1
    output=(right_lines,left_lines)
    return output,image,right_detect,left_detect


def regression(sl,image,right_detect,left_detect):#두 차선 교차하는 지점이 중심점으로부터 좌우 어느쪽에 있는지
    right_pts=[]
    left_pts=[]
    global left_m,left_b,right_m,right_b
    if right_detect == 1:
        for s in sl[0]:
            x5,y5,x6,y6=s
            right_pts.append([x5,y5])
            right_pts.append([x6,y6])
        if len(right_pts)>0:
            right_pts=np.asarray(right_pts)
            right_line = cv2.fitLine(right_pts, cv2.DIST_L2, 0, 0.01, 0.01)#근사화
            right_m=float(float(right_line[1])/float(right_line[0]))#기울기
            right_b=(right_line[2],right_line[3])#점

    if left_detect == 1:
        for s in sl[1]:
            x7,y7,x8,y8=s
            left_pts.append([x7,y7])
            left_pts.append([x8,y8])
        if len(left_pts)>0:
            left_pts=np.asarray(left_pts)
            left_line = cv2.fitLine(left_pts, cv2.DIST_L2, 0, 0.01, 0.01)#근사화
            left_m=float(float(left_line[1])/float(left_line[0]))#기울기
            left_b=(left_line[2],left_line[3])#점

    h, w ,_= image.shape
    h = int(h)
    w = int(w)

    #좌우 선 각각의 두 점 계산
    #y=m*x+b-->x=(y-b)/m

    right_ini_x=int(((h-right_b[1])/right_m)+right_b[0])
    right_fni_x = int(((400 - right_b[1]) / right_m) + right_b[0])
    left_ini_x=int(((h-left_b[1])/left_m)+left_b[0])
    left_fni_x = int(((400 - left_b[1]) / left_m) + left_b[0])

    output=[[right_ini_x,h], [right_fni_x,400],[left_ini_x,h],[left_fni_x,400]]
    #image = cv2.line(image, (right_ini_x,h), (right_fni_x,300), (0, 50, 50), 30, cv2.LINE_AA)
    #image = cv2.line(image, (left_ini_x,h),(left_fni_x,300), (0, 255, 255), 20, cv2.LINE_AA)
    return output,image
def predictDir(image):#진행방향 예측
    h, w ,_ = image.shape
    h = int(h)
    w = int(w)
    img_center=w/2
    #car_test_3
    thres_vp=180
    #thres_vp=150
    vx=float(((right_m*right_b[0])-(left_m*left_b[0])-right_b[1]+left_b[1])/(right_m-left_m))

    print(vx)
    print(img_center-thres_vp)
    print(img_center+thres_vp)
    if left_detect == 0 and right_detect == 1:
        output = 'Left Turn'
        socket.send('L')

    elif left_detect == 1 and right_detect == 0:
        output = 'Right turn'
        socket.send('R')
    else:
        if vx < (img_center - thres_vp):
            output = 'Left Turn'
            socket.send('L')
        elif vx > (img_center + thres_vp):
            output = 'Right Turn'
            socket.send('R')
        elif (vx >= (img_center - thres_vp)) & (vx <= (img_center + thres_vp)):
            output = 'Straight'
            socket.send('F')
    return output
def drawLine(img, lane, dir):
    poly_points=[]
    poly_points.append(lane[2])
    poly_points.append(lane[0])
    poly_points.append(lane[1])
    poly_points.append(lane[3])
    img_copy=img
   # img=cv2.fillConvexPoly(img, poly_points,(0,230,30),cv2.LINE_AA,0)
    #print(lane)
    #print('lane\n\n\n\n')
    p0=[int(lane[0][0]),int(lane[0][1])]
    p1 = [int(lane[1][0]), int(lane[1][1])]
    p2=[int(lane[2][0]),int(lane[2][1])]
    p3 = [int(lane[3][0]), int(lane[3][1])]

    img=cv2.addWeighted(img_copy,0.3,img,0.7,0.0)
    img=cv2.line(img, p0,p1,(0,255,255),10,cv2.LINE_AA)
    img=cv2.line(img, p2,p3,(0,255,255),10,cv2.LINE_AA)

    img=cv2.putText(img, dir, (520,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3,cv2.LINE_AA)
    return img
def yolo(img):
    h, w, c = img.shape

    # YOLO 입력
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0),
                                 True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 검출 신뢰도
            if confidence > 0.5:
                # Object detected
                # 검출기의 경계상자 좌표는 0 ~ 1로 정규화되어있으므로 다시 전처리
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]
            # 경계상자와 클래스 정보 투영
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(img, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
            if label=='ternel':
                socket.send('ternel')
                #픽셀의 밝기 변화
            if label=='up':
                socket.send('up')
            if label=='down':
                socket.send('down') #원래대로
            if label=='rotary':
                socket.send('rotary')
            if label=='parking':
                socket.send('parking')
            if label=='curve':
                socket.send('curve')
    return img
lower_red1 = np.array([179,51,51])
lower_red2 = np.array([0,51,51])
lower_red3 = np.array([169,51,51])
upper_red1 = np.array([180,255,255])
upper_red2 = np.array([9,255,255])
upper_red3 = np.array([179,255,255])

lower_blue1 = np.array([101,30,30])
lower_blue2 = np.array([91,30,30])
lower_blue3 = np.array([91,30,30])
upper_blue1 = np.array([111,255,255])
upper_blue2 = np.array([101,255,255])
upper_blue3 = np.array([101,255,255])


while True:
    ret, img = cap.read()
    #img=cv2.resize(img, dsize=(500, 500))
    height, width = img.shape[:2]
    img_color = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv2.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask4 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    img_mask5 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    img_mask6 = cv2.inRange(img_hsv, lower_red3, upper_red3)
    img_mask = img_mask1 | img_mask2 | img_mask3 | img_mask4 | img_mask5 | img_mask6

    kernel = np.ones((11, 11), np.uint8)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)

    numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(img_mask)

    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])
        #print(centerX, centerY)

        if area > 10000:
            #cv2.circle(img_color, (centerX, centerY), 10, (0, 0, 255), 10)
            #cv2.rectangle(img_color, (x, y), (x + width, y + height), (0, 0, 255))
            if x * y > 10000:
                matching(img_color)

    img_mask=filter_colors(img) #white & yellow filtering
    blur = cv2.GaussianBlur(img_mask, (3, 3), 0)#가우시안
    canny = cv2.Canny(blur, 100, 200) # canny 함수
    roi=limit_region(canny) #ROI 관심영역
    hough,lines=Hough(roi)
    #yl=yolo(img)
    if lines.size>0:
         sl,slimg,rd,ld=seperate_line(hough,img,lines)#좌우 분리
         lane,rimg=regression(sl,img,rd,ld)#진행방향 예측
         dir=predictDir(img)
         result=drawLine(img,lane, dir)
    #cv2.imshow('mask', img_mask)
    #cv2.imshow('gaussian', blur)
    #cv2.imshow('image',img)
    #cv2.imshow('canny',canny)
    key = cv2.waitKey(1)
    #cv2.imshow('line',roi)
    #cv2.imshow('hough',hough)
    #cv2.imshow('rimg',rimg)
    #cv2.imshow('result',result)
    #cv2.imshow('yolo',yl)
    if key == 27:  # esc
        break
