import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Open the video camera no.0
cap = cv.VideoCapture('LAB3_Video.mp4')

# If not success, exit the program
if not cap.isOpened():
    print('Cannot open video')

#cv.namedWindow('MyVideo', cv.WINDOW_AUTOSIZE)

y_min_roi=700
y_min=700#c초기값 설정
x_center_roi=450
count=0#초기 프레임만 적용하기 위해
while True:

    # Load image
    ret, img = cap.read()#simple 2번 문제

    # If not success, break loop
    if not ret:
        print('Cannot read frame')
        break

    # Display Image
    cv.namedWindow('source', cv.WINDOW_AUTOSIZE) 
    cv.imshow('source',img)

    # cv.rectangle(img, (75, 0), (500, 450), (0, 0, 255), -1)
    # pts = np.array([[75, 450], [500, 450],[200,600]], np.int32)
    # # pts = pts.reshape((-1, 1, 2))  # 꼭 reshape 해야 함
    # cv.fillPoly(img, [pts], (0,0,255))  # 전체를 흰색(255)으로 채움


    b,g,r=cv.split(img)
    # cv.imshow("b", b)
    # cv.imshow("g", g)
    cv.imshow("r", r)

    # # --- 1. 빨간색 마스크 생성 ---
    # red_mask = (r > 150) & (r > g + 50) & (r > b + 50)  # 조건은 조절 가능

    # # --- 2. 마스크 영역만 추출 ---
    # red_only = np.zeros_like(r)
    # red_only[red_mask] = r[red_mask]

    # # --- 3. 블러링 (노이즈 제거) ---
    # red_blur = cv.blur(red_only, (5, 5))

    # # --- 4. Canny 엣지 검출 ---
    # edges = cv.Canny(red_blur, 40, 80)

    # # --- 5. 결과 출력 ---
    # cv.imshow('Red Edge', edges)


    # # gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # # cv.namedWindow('gray', cv.WINDOW_AUTOSIZE) 
    # # cv.imshow('gray',gray)

    blur = cv.blur(r,(5,5))
    blur_g = cv.blur(g,(5,5))
    blur_b = cv.blur(b,(5,5))
    # Display Image

    cv.namedWindow('BLUR', cv.WINDOW_AUTOSIZE) 
    cv.imshow('BLUR',blur)

    kernel = np.ones((3,3),np.uint8)

    edge = cv.Canny(blur,55,60)

    # edge_red=edge+edges



    # #roi만 남기기기
    # height=len(img[:,1])
    # width=len(img[1,:])
    # cv.rectangle(edge, (0, 0), (width, 400), (0, 255, 255), -1)
    # cv.rectangle(edge, (150, 0), (500, 300), (255, 0, 255), -1)
    # # cv.rectangle(edge, (900, 0), (width, height), (0, 0, 0), -1)
    # pts = np.array([[1000, 0], [width, 0], [width, height], [x_center_roi,height],[x_center_roi, height-70], [x_center_roi+200, y_min-50]], np.int32)
    # # pts = pts.reshape((-1, 1, 2))  # 꼭 reshape 해야 함
    # cv.fillPoly(edge, [pts], (255,255,0))  # 전체를 흰색(255)으로 채움

    #roi만 남기기기
    height=len(img[:,1])
    width=len(img[1,:])
    cv.rectangle(edge, (0, 0), (width, 450), (0, 0, 0), -1)
    cv.rectangle(edge, (150, 450), (500, 600), (0, 0, 0), -1)
    # cv.rectangle(edge, (900, 0), (width, height), (0, 0, 0), -1)
    cv.rectangle(edge, (800, 0), (width, height), (0, 0, 0), -1)
    pts = np.array([[600, y_min-150], [width, y_min-150], [width, height], [480,height], [450, y_min+10],  [600, y_min-150]], np.int32)#원래 버젼
    
    # pts = np.array([[1000, 0], [width, 0], [width, height], [x_center_roi,height],[x_center_roi, y_min+30],  [700, y_min-50]], np.int32)
    # pts = pts.reshape((-1, 1, 2))  # 꼭 reshape 해야 함
    cv.fillPoly(edge, [pts], (0,0,0))  # 전체를 흰색(255)으로 채움
    if count !=0:
        pts=pts_total
    cv.fillPoly(edge, [pts], (0,0,0))  # 전체를 흰색(255)으로 채움
    # #roi만 남기기기ver2
    # height=len(img[:,1])
    # width=len(img[1,:])
    # cv.rectangle(edge_red, (0, 0), (width, 400), (0, 0, 0), -1)
    # cv.rectangle(edge_red, (150, 0), (500, 300), (0, 0, 0), -1)
    # # cv.rectangle(edge, (900, 0), (width, height), (0, 0, 0), -1)
    # pts = np.array([[1000, 0], [width, 0], [width, height], [480,height],  [700, height-300]], np.int32)
    # # pts = pts.reshape((-1, 1, 2))  # 꼭 reshape 해야 함
    # cv.fillPoly(edge_red, [pts], (0,0,0))  # 전체를 흰색(255)으로 채움

    dilation = cv.dilate(edge,kernel,iterations = 1)
    cv.namedWindow('Edge', cv.WINDOW_AUTOSIZE) 
    cv.imshow('Edge',edge)

    # cv.namedWindow('Edge red', cv.WINDOW_AUTOSIZE) 
    # cv.imshow('Edge red',edge_red)

    # edge_g = cv.Canny(blur_g,10,40)
    # # cv.namedWindow('Edge g', cv.WINDOW_AUTOSIZE) 
    # # cv.imshow('Edge g',edge_g)

    # edge_b = cv.Canny(blur_b,10,40)
    # dilation = cv.dilate(edge_b,kernel,iterations = 1)
    # # cv.namedWindow('Edge b', cv.WINDOW_AUTOSIZE) 
    # # cv.imshow('Edge b',edge_b)

    # edge_total=edge+edge_b+edge_g
    # # cv.namedWindow('Edge total', cv.WINDOW_AUTOSIZE) 
    # # cv.imshow('Edge total',edge_total)

    img_contour, hieracy=cv.findContours(edge,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )
    src_img=img.copy()
    poly_line_img=img.copy()
    # draw_con=cv.drawContours(src_img,img_contour, -1, (0, 255, 0))
    curve_points=[]
    x_max=0
    for cnt in img_contour:
        length = cv.arcLength(cnt, False)
        # if cv.contourArea(cnt) > 20 :  # 너무 작은 contour 무시
        if 110 < length :
            cv.drawContours(src_img, [cnt], -1, (0, 255, 0), 1)
            for pt in cnt:
                x, y=pt[0]
                if x>x_max:
                    x_max=x
                curve_points.append([x, y])

    cv.namedWindow('draw con', cv.WINDOW_AUTOSIZE) 
    cv.imshow('draw con',src_img)

    # curve_points = np.array(curve_points)

    # for pt in curve_points:
    #     x, y = pt
    #     cv.circle(poly_line_img, (int(x), int(y)), 2, (255, 0, 0), -1)  # 파란 점

    # # x, y 좌표 분리
    # x_vals = curve_points[:, 0]
    # y_vals = curve_points[:, 1]

    curve_points = np.array(curve_points)

    if len(curve_points) == 0:
        # print("곡선 포인트 없음 - 다음 프레임으로 넘어감")
        # curve_points=past_curve_points
        continue  # 다음 프레임으로 넘어감
    
    # past_curve_points=curve_points
    # for pt in curve_points:
    #     x, y = pt
    #     cv.circle(poly_line_img, (int(x), int(y)), 2, (255, 0, 0), -1)  # 파란 점

    # 이제 안전하게 인덱싱 가능
    x_vals = curve_points[:, 0]
    y_vals = curve_points[:, 1]


    fit = np.polyfit(x_vals, y_vals, 2)#y=ax^2+bx+c
    a, b, c = fit

    x_center = -b / (2 * a)
    y_min = a * x_center ** 2 + b * x_center + c



    # # for i in img_contour
    # #     draw_con=cv.drawContours(src,img_contour(i), i)
        
    # cv.imshow('draw con',draw_con)
    # cv.namedWindow('Dialate', cv.WINDOW_AUTOSIZE) 
    # cv.imshow('Dialate',dilation)

    #점선 부분분
    bottom=len(img[:,1])
    right=len(img[1,:])
    # cv.line(img, (0, bottom-250), (right, bottom-250), (50,255,50), 2, cv.LINE_AA)

    # cv.line(img, (0, bottom-120), (right, bottom-120), (255,60,50), 2, cv.LINE_AA)


    # cv.namedWindow('results', cv.WINDOW_AUTOSIZE) 
    # cv.imshow('results',img)

    score = bottom - y_min

    if score > 250:
        level = 1
    elif score >= 120:
        level = 2
    else:
        level = 3

    for x in range(0, right, 60):
        x1, x2 = x, x + 30
        curve_y1 = a * x1**2 + b * x1 + c
        curve_y2 = a * x2**2 + b * x2 + c

        y1 = bottom - 250  # 녹색 점선
        y2 = bottom - 120  # 주황 점선

        # 곡선 아래에 있는 점선만 그림 (x와 x+30 모두 확인)
        if level == 2:
            if y1 > curve_y1 and y1 > curve_y2:
                cv.line(poly_line_img, (x1, y1), (x2, y1), (0, 255, 255), 1, cv.LINE_AA)
            if y2 > curve_y1 and y2 > curve_y2:
                cv.line(poly_line_img, (x1, y2), (x2, y2), (255, 255, 0), 1, cv.LINE_AA)

        elif level == 3:
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
    curve_fit = np.array([np.column_stack((x_fit, y_fit))], dtype=np.int32)


    x_fit_offset = np.linspace(int(0), int((800)), 100)
    y_fit_offset = a * (x_center-x_fit_offset)*(x_center-x_fit_offset)+ (y_min+50)
    curve_fit_offset = np.array([np.column_stack((x_fit_offset, y_fit_offset))], dtype=np.int32)
    ppts_bottom = np.array([[x_fit_offset[-1], bottom],[x_fit_offset[0], bottom]])
    pts_top = np.column_stack((x_fit_offset, y_fit_offset))#x, y좌표 결합하기
    pts_total = np.vstack((pts_top, ppts_bottom)).astype(np.int32)

    text = f"Score : {int(score)}\nLevel : {level}"
    y0, dy = 250, 30  # 시작 y 좌표와 줄 간격
    x0=880#문자 겉 테두리 상자 x좌표
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv.putText(poly_line_img, line, (x0, y), cv.FONT_HERSHEY_SIMPLEX, 
                0.9, (0, 255, 0), 1, cv.LINE_AA)
        
    cv.rectangle(poly_line_img, (x0 - 10, y0 - 30), (x0 + 180, y0 + 50), (0, 255, 0), 2)
    
    # 곡선 그리기
    cv.polylines(poly_line_img, curve_fit, False, (0, 255, 0), 2)
    # cv.polylines(poly_line_img, curve_fit_offset, False, (255, 255, 0), 2)
    # cv.fillPoly(poly_line_img, [pts_total], (255,0,255))  # 전체를 흰색(255)으로 채움
    cv.namedWindow('Poly', cv.WINDOW_AUTOSIZE) 
    cv.imshow('Poly',poly_line_img)

    # print("score: ",int(score),"\nLevel: ", level)
    count=count+1

    #cv.waitKey(0)
    if cv.waitKey(30) & 0xFF == 27:
        print('Press ESC to stop')
        break

cv.destroyAllWindows()
cap.release()