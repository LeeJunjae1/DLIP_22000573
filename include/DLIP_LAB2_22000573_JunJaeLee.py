import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# Load image
img = cv.imread('LV3.png')#simple 2번 문제

# Display Image
cv.namedWindow('source', cv.WINDOW_AUTOSIZE) 
cv.imshow('source',img)

#r, g, b로 각각 grayscale진행행
b,g,r=cv.split(img)
# cv.imshow("b", b)
# cv.imshow("g", g)
cv.imshow("r", r)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.namedWindow('gray', cv.WINDOW_AUTOSIZE) 
cv.imshow('gray',gray)

#노이즈 제거를 위해 필터 적용용
blur = cv.blur(r,(5,5))
blur_g = cv.blur(g,(5,5))
blur_b = cv.blur(b,(5,5))
# Display Image

cv.namedWindow('BLUR', cv.WINDOW_AUTOSIZE) 
cv.imshow('BLUR',blur)

kernel = np.ones((3,3),np.uint8)

#canny를 적용해 edge 검출출
edge = cv.Canny(blur,10,90)



edge_g = cv.Canny(blur_g,50,100)
# cv.namedWindow('Edge g', cv.WINDOW_AUTOSIZE) 
# cv.imshow('Edge g',edge_g)

edge_b = cv.Canny(blur_b,50,100)
dilation = cv.dilate(edge_b,kernel,iterations = 1)
# cv.namedWindow('Edge b', cv.WINDOW_AUTOSIZE) 
# cv.imshow('Edge b',edge_b)

edge_total=edge+edge_b+edge_g
# cv.namedWindow('Edge total', cv.WINDOW_AUTOSIZE) 
# cv.imshow('Edge total',edge_total)

#roi만 남기기기
height=len(img[:,1])#이미지 전체 높이
width=len(img[1,:])#이미지 전체 너비비
cv.rectangle(edge, (0, 0), (width, 400), (0, 0, 0), -1)#이미지 윗부분 제거거
cv.rectangle(edge, (150, 0), (500, 300), (0, 0, 0), -1)#내부 부분 제거
# cv.rectangle(edge, (900, 0), (width, height), (0, 0, 0), -1)
pts = np.array([[1000, 0], [width, 0], [width, height], [500,height], [700, height-300]], np.int32)#관심 없는 부분을 제거하기 위해해
# pts = pts.reshape((-1, 1, 2))  # 꼭 reshape 해야 함
cv.fillPoly(edge, [pts], (0,0,0))  # 전체를 흰색(255)으로 채움

dilation = cv.dilate(edge,kernel,iterations = 1)
cv.namedWindow('Edge', cv.WINDOW_AUTOSIZE) 
cv.imshow('Edge',edge)

img_contour, hieracy=cv.findContours(edge,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )#contour 찾기
src_img=img.copy()#contour한 결과 출력하기 위해 원래 이미지 복사하기
poly_line_img=img.copy()#전체 결과를 출력하기 위해 원래 이미지 복사하기
# draw_con=cv.drawContours(src_img,img_contour, -1, (0, 255, 0))
curve_points=[]#contour를 통해 얻은 좌표를 저장하기 위해해
x_max=0
for cnt in img_contour:#contour 전체 찾기
    if cv.contourArea(cnt) > 40:  # 너무 작은 contour 무시
        cv.drawContours(src_img, [cnt], -1, (0, 255, 0), 1)
        for pt in cnt:#
            x, y=pt[0]#0번째 x, y 좌표값 갖고오기기
            # if x>x_max:
            #     x_max=x
            curve_points.append([x, y])#곡선 좌표 값 저장하기기

cv.namedWindow('draw con', cv.WINDOW_AUTOSIZE) 
cv.imshow('draw con',src_img)

curve_points = np.array(curve_points)#배열로 변환하기기

# for pt in curve_points:
#     x, y = pt
#     cv.circle(poly_line_img, (int(x), int(y)), 2, (255, 0, 0), -1)  # 파란 점

# x, y 좌표 분리
x_vals = curve_points[:, 0]
y_vals = curve_points[:, 1]


fit = np.polyfit(x_vals, y_vals, 2)#y=ax^2+bx+c
a, b, c = fit#2차 근사식의 계수

#최소값의 x, y값 계산산
x_center = -b / (2 * a)
y_min = a * x_center ** 2 + b * x_center + c



# # for i in img_contour
# #     draw_con=cv.drawContours(src,img_contour(i), i)
    
# cv.imshow('draw con',draw_con)
# cv.namedWindow('Dialate', cv.WINDOW_AUTOSIZE) 
# cv.imshow('Dialate',dilation)

#점선 부분
bottom=len(img[:,1])#이미지 전체 높이
right=len(img[1,:])#이미지 전체 너비비
# cv.line(img, (0, bottom-250), (right, bottom-250), (50,255,50), 2, cv.LINE_AA)

# cv.line(img, (0, bottom-120), (right, bottom-120), (255,60,50), 2, cv.LINE_AA)

# for x in range(0, right, 60):
#     cv.line(poly_line_img, (x, bottom - 250), (x + 30, bottom - 250), (0, 255, 255), 1, cv.LINE_AA)
#     cv.line(poly_line_img, (x, bottom - 120), (x + 30, bottom - 120), (255, 255, 0), 1, cv.LINE_AA)
# cv.namedWindow('results', cv.WINDOW_AUTOSIZE) 
# cv.imshow('results',img)

# score = bottom - y_min

# if score > 250:
#     level = 1
# elif score >= 120:
#     level = 2
# else:
#     level = 3

# # 곡선 좌표 생성
# x_fit = np.linspace(0, x_max, 100)
# y_fit = a * x_fit**2 + b * x_fit + c
# curve_fit = np.array([np.column_stack((x_fit, y_fit))], dtype=np.int32)

# text = f"Score : {int(score)}\nLevel : {level}"
# y0, dy = 250, 30  # 시작 y 좌표와 줄 간격
# x0=880#x좌표표
# for i, line in enumerate(text.split('\n')):
#     y = y0 + i * dy
#     cv.putText(poly_line_img, line, (x0, y), cv.FONT_HERSHEY_SIMPLEX, 
#                0.9, (0, 255, 0), 1, cv.LINE_AA)
    
# cv.rectangle(poly_line_img, (x0 - 10, y0 - 30), (x0 + 180, y0 + 50), (0, 255, 0), 2)

# # 곡선 그리기
# cv.polylines(poly_line_img, curve_fit, False, (0, 255, 0), 2)
# cv.namedWindow('Poly', cv.WINDOW_AUTOSIZE) 
# cv.imshow('Poly',poly_line_img)

# print("score: ",int(score),"\nLevel: ", level)

#score 계산 실시
score = bottom - y_min

#score에 맞게 level 표현현
if score > 250:
    level = 1
elif score >= 120:
    level = 2
else:
    level = 3

for x in range(0, right, 60):
    x1, x2 = x, x + 30#점선의 x 좌표값 계산
    #점선의 y좌표 값 계산산
    curve_y1 = a * x1**2 + b * x1 + c
    curve_y2 = a * x2**2 + b * x2 + c

    y1 = bottom - 250  # 녹색 점선
    y2 = bottom - 120  # 주황 점선

    # 곡선 아래에 있는 점선만 그림 (x와 x+30 모두 확인)
    if level == 2:
        #겹치는 부분은 점선 표현하지 안기 위해해
        if y1 > curve_y1 and y1 > curve_y2:
            cv.line(poly_line_img, (x1, y1), (x2, y1), (0, 255, 255), 1, cv.LINE_AA)
        if y2 > curve_y1 and y2 > curve_y2:
            cv.line(poly_line_img, (x1, y2), (x2, y2), (255, 255, 0), 1, cv.LINE_AA)

    elif level == 3:
        #겹치는 부분은 점선 표현하지 않기 위해해
        if y1 > curve_y1 and y1 > curve_y2:
            cv.line(poly_line_img, (x1, y1), (x2, y1), (0, 255, 255), 1, cv.LINE_AA)
        if y2 > curve_y1 and y2 > curve_y2:
            cv.line(poly_line_img, (x1, y2), (x2, y2), (255, 255, 0), 1, cv.LINE_AA)

    else:  # level == 1
        if y1 > curve_y1 and y1 > curve_y2:
            cv.line(poly_line_img, (x1, y1), (x2, y1), (0, 255, 255), 1, cv.LINE_AA)
        if y2 > curve_y1 and y2 > curve_y2:
            cv.line(poly_line_img, (x1, y2), (x2, y2), (255, 255, 0), 1, cv.LINE_AA)


# 곡선 좌표 생성
x_fit = np.linspace(int(x_center-300), int((x_center+300)), 100)
y_fit = a * x_fit**2 + b * x_fit + c
curve_fit = np.array([np.column_stack((x_fit, y_fit))], dtype=np.int32)#curve fitting 실시시

text = f"Score : {int(score)}\nLevel : {level}"#화면에 출력할 문자자
y0, dy = 250, 30  # 출력할 문자의 시작 y 좌표와 줄 간격
x0=880#출력할 문자의 x좌표
for i, line in enumerate(text.split('\n')):
    y = y0 + i * dy
    cv.putText(poly_line_img, line, (x0, y), cv.FONT_HERSHEY_SIMPLEX, 
            0.9, (0, 255, 0), 1, cv.LINE_AA)
    
#문자를 표현할 상자를 표현하기 위해해
cv.rectangle(poly_line_img, (x0 - 10, y0 - 30), (x0 + 180, y0 + 50), (0, 255, 0), 2)

# 곡선 그리기
cv.polylines(poly_line_img, curve_fit, False, (0, 255, 0), 2)
cv.namedWindow('Poly', cv.WINDOW_AUTOSIZE) 
cv.imshow('Poly',poly_line_img)

cv.waitKey(0)