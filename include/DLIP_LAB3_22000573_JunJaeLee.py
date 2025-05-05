# /* ------------------------------------------------------ /
# *Image Proccessing with Deep Learning
# * Tension Detection of Rolling Metal Sheet
# * Created : 2025-04-22
# * Name: Junjae Lee
# ------------------------------------------------------ */
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def process_image(image_path, idx):
    # Load image
    img = cv.imread(image_path)#이미자 가져오기

    #r, g, b로 각각 grayscale진행
    b,g,r=cv.split(img)


    #노이즈 제거를 위해 필터 적용
    blur = cv.blur(r,(5,5))

    #canny를 적용해 edge 검출
    edge = cv.Canny(blur,10,90)

    #roi만 남기기기
    height=len(img[:,1])#이미지 전체 높이
    width=len(img[1,:])#이미지 전체 너비
    cv.rectangle(edge, (0, 0), (width, 400), (0, 0, 0), -1)#이미지 윗부분 제거
    cv.rectangle(edge, (150, 0), (500, 300), (0, 0, 0), -1)#내부 부분 제거
    # cv.rectangle(edge, (900, 0), (width, height), (0, 0, 0), -1)
    pts = np.array([[1000, 0], [width, 0], [width, height], [500,height], [700, height-300]], np.int32)#관심 없는 부분을 제거하기 위해
    cv.fillPoly(edge, [pts], (0,0,0))  # 전체를 흰색(255)으로 채움

    

    img_contour, hieracy=cv.findContours(edge,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )#contour 찾기
    src_img=img.copy()#contour한 결과 출력하기 위해 원래 이미지 복사하기
    poly_line_img=img.copy()#전체 결과를 출력하기 위해 원래 이미지 복사하기
    
    curve_points=[]#contour를 통해 얻은 좌표를 저장하기 위해
    for cnt in img_contour:#contour 전체 찾기
        if cv.contourArea(cnt) > 40:  # 너무 작은 contour 무시
            cv.drawContours(src_img, [cnt], -1, (0, 255, 0), 1)# contour 결과 그리기
            for pt in cnt:#
                x, y=pt[0]#x, y 좌표값 갖고오기
                curve_points.append([x, y])#곡선 좌표 값 저장하기

    

    curve_points = np.array(curve_points)#배열로 변환하기


    # x, y 좌표 분리
    x_vals = curve_points[:, 0]
    y_vals = curve_points[:, 1]


    fit = np.polyfit(x_vals, y_vals, 2)#y=ax^2+bx+c
    a, b, c = fit#2차 근사식의 계수

    #최소값의 x, y값 계산
    x_center = -b / (2 * a)
    y_min = a * x_center ** 2 + b * x_center + c


    #점선 부분
    bottom=len(img[:,1])#이미지 전체 높이
    right=len(img[1,:])#이미지 전체 너비


    #score 계산 실시
    score = bottom - y_min

    #score에 맞게 level 표현
    if score > 250:
        level = 1
    elif score >= 120:
        level = 2
    else:
        level = 3

    for x in range(0, right, 60):
        x1, x2 = x, x + 30#점선의 x 좌표값 계산
        #점선의 y좌표 값 계산
        curve_y1 = a * x1**2 + b * x1 + c
        curve_y2 = a * x2**2 + b * x2 + c

        y1 = bottom - 250  # 녹색 점선
        y2 = bottom - 120  # 주황 점선

        # 곡선 아래에 있는 점선만 그림 (x와 x+30 모두 확인)
        if level == 2:
            #겹치는 부분은 점선 표현하지 않기 위해
            if y1 > curve_y1 and y1 > curve_y2:
                cv.line(poly_line_img, (x1, y1), (x2, y1), (0, 255, 255), 1, cv.LINE_AA)
            if y2 > curve_y1 and y2 > curve_y2:
                cv.line(poly_line_img, (x1, y2), (x2, y2), (255, 255, 0), 1, cv.LINE_AA)

        elif level == 3:
            #겹치는 부분은 점선 표현하지 않기 위해
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
    curve_fit = np.array([np.column_stack((x_fit, y_fit))], dtype=np.int32)#curve fitting 실시

    text = f"Score : {int(score)}\nLevel : {level}"#화면에 출력할 문자
    y0, dy = 250, 30  # 출력할 문자의 시작 y 좌표와 줄 간격
    x0=880#출력할 문자의 x좌표
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy#줄 간격
        cv.putText(poly_line_img, line, (x0, y), cv.FONT_HERSHEY_SIMPLEX, 
                0.9, (0, 255, 0), 1, cv.LINE_AA)#문자 출력
        
    #문자를 표현할 상자를 표현하기 위해
    cv.rectangle(poly_line_img, (x0 - 10, y0 - 30), (x0 + 180, y0 + 50), (0, 255, 0), 2)

    # 곡선 그리기
    cv.polylines(poly_line_img, curve_fit, False, (0, 255, 0), 2)

    # cv.namedWindow('source', cv.WINDOW_AUTOSIZE) 
    # cv.imshow('source',img)
    # cv.imshow("r", r)
    # cv.namedWindow(f'Resulgt[{idx}]', cv.WINDOW_AUTOSIZE) 
    # cv.imshow("g", g)
    # cv.namedWindow(f'Resulbt[{idx}]', cv.WINDOW_AUTOSIZE) 
    # cv.imshow("b", b)
    # cv.namedWindow('BLUR', cv.WINDOW_AUTOSIZE) 
    # cv.imshow('BLUR',blur)
    # cv.namedWindow('Edge', cv.WINDOW_AUTOSIZE) 
    # cv.imshow('Edge',edge)
    # cv.namedWindow('draw con', cv.WINDOW_AUTOSIZE) 
    # cv.imshow('draw con',src_img)
    cv.namedWindow(f'Result[{idx}]', cv.WINDOW_AUTOSIZE) 
    cv.imshow(f'Result[{idx}]',poly_line_img)

    
def process_video(image_path, idx):
    cap = cv.VideoCapture(image_path)#영상 불러오기

    # If not success, exit the program
    if not cap.isOpened():
        print('Cannot open video')

    #cv.namedWindow('MyVideo', cv.WINDOW_AUTOSIZE)

    y_min=700#c초기값 roi 설정
    count=0#초기 프레임만 적용하기 위해

    # --- 비디오 저장 설정 ---
    fourcc = cv.VideoWriter_fourcc(*'XVID')  # avi 파일로 저장하기 위해
    fps = 30  # 영상의 프레임 속도 (원본 영상과 동일하게)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))#프레임 전체 너비 계산
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))#프레임 전체 높이 계산

    out = cv.VideoWriter('DLIP_LAB3_22000573JunjaeLee.avi', fourcc, fps, (frame_width, frame_height))#비디오 저장


    while True:

        ret, img = cap.read()#s영상 받아들이기

        # 오류 발생 시 중단하기
        if not ret:
            print('Cannot read frame')
            break

        b,g,r=cv.split(img)#r, g, b로 나눈 후 gray scale 진행
       
        blur = cv.blur(r,(5,5))#노이즈 제거를 위해 필터 처리

        edge = cv.Canny(blur,55,60)#canny를 통한 edge 검출

        #roi만 남기기
        height=len(img[:,1])#영상 전체 높이
        width=len(img[1,:])#영상 전체 너비
        cv.rectangle(edge, (0, 0), (width, 450), (0, 0, 0), -1)#영상의 제일 관심 없는 윗부분 제거용
        cv.rectangle(edge, (150, 450), (500, 600), (0, 0, 0), -1)#금속 안쪽 제거 용
        cv.rectangle(edge, (800, 0), (width, height), (0, 0, 0), -1)# 관심없는 오른쪽 부분 제거용
        pts = np.array([[600, y_min-150], [width, y_min-150], [width, height], [480,height], [450, y_min+10],  [600, y_min-150]], np.int32)#메탈을 제와한 나머지 영역을 제거하기 위해 설정
        cv.fillPoly(edge, [pts], (0,0,0))  # 전체를 흰색(255)으로 채움

        #메탈 모양과 유사하게 만들어 좀더 확실하게 제거
        if count !=0:
            pts=pts_total#이전에 curve fitting 한 것에 오프셋을 주어 평행이동 실시. 이를 통해해 관심있는 부분을 제외한 부분 제거
        cv.fillPoly(edge, [pts], (0,0,0))  # 전체를 흰색(255)으로 채움


        img_contour, hieracy=cv.findContours(edge,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )# contour 실시
        src_img=img.copy()#contour 결과를 표현하기 위해
        poly_line_img=img.copy()#최종 결과물을 표현하기 위해
        # draw_con=cv.drawContours(src_img,img_contour, -1, (0, 255, 0))

        curve_points=[]#curve fitting할 좌표를 구하기 위해
        for cnt in img_contour:
            length = cv.arcLength(cnt, False)#contour 길이 계산
            # if cv.contourArea(cnt) > 20 :  # 너무 작은 contour 무시
            if 110 < length :#너무 길이가 짧은 것은 제거거
                cv.drawContours(src_img, [cnt], -1, (0, 255, 0), 1)#contour 결과 표현하기
                for pt in cnt:
                    x, y=pt[0]#x, y값 모두 추출
                    curve_points.append([x, y])#x,y값 저장
        

        curve_points = np.array(curve_points)#배열로 변환하기

        if len(curve_points) == 0:#contour가 없을 시 해당과정은 진행하지 않음
            continue  # 다음 프레임으로 넘어감
        


        # x, y값 추출
        x_vals = curve_points[:, 0]
        y_vals = curve_points[:, 1]


        fit = np.polyfit(x_vals, y_vals, 2)#y=ax^2+bx+c
        a, b, c = fit#curve fitting을 통해 얻은 계수

        #최소값의 x, y좌표 계산
        x_center = -b / (2 * a)
        y_min = a * x_center ** 2 + b * x_center + c


        #점선 부분
        bottom=len(img[:,1])#이미지 최대 너비
        right=len(img[1,:])#이미지 최대 높이

        #score 계산산
        score = bottom - y_min

        #score를 기반으로 level 계산산
        if score > 250:
            level = 1
        elif score >= 120:
            level = 2
        else:
            level = 3

        for x in range(0, right, 60):
            
            #점선이 없어질 부분계산
            x1, x2 = x, x + 30#없어질 x좌표

            #없어질 y좌표
            curve_y1 = a * x1**2 + b * x1 + c
            curve_y2 = a * x2**2 + b * x2 + c

            y1 = bottom - 250  # 녹색 점선
            y2 = bottom - 120  # 하늘색색 점선

            # 곡선 아래에 있는 점선만 그림 (x와 x+30 모두 확인)
            if level == 2:
                #겹치는 부분은 점선 표현하지 않기 위해
                if y1 > curve_y1 and y1 > curve_y2:
                    cv.line(poly_line_img, (x1, y1), (x2, y1), (0, 255, 255), 1, cv.LINE_AA)
                if y2 > curve_y1 and y2 > curve_y2:
                    cv.line(poly_line_img, (x1, y2), (x2, y2), (255, 255, 0), 1, cv.LINE_AA)

            elif level == 3:
                #겹치는 부분은 점선 표현하지 않기 위해
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

        #roi가 아닌 부분을 제거 하기 위한 곡선의 좌표
        x_fit_offset = np.linspace(int(0), int((800)), 100)#해당 x좌표
        y_fit_offset = a * (x_center-x_fit_offset)*(x_center-x_fit_offset)+ (y_min+50)#x좌표에 해당하는 y좌표

        curve_fit_offset = np.array([np.column_stack((x_fit_offset, y_fit_offset))], dtype=np.int32)# 배열로 변환
        ppts_bottom = np.array([[x_fit_offset[-1], bottom],[x_fit_offset[0], bottom]])#다각형을 그리기 위해선 밑부분 좌표도 필요
        pts_top = np.column_stack((x_fit_offset, y_fit_offset))#x, y좌표 결합하기
        pts_total = np.vstack((pts_top, ppts_bottom)).astype(np.int32)#바닥을 그릴 좌표와 같이 결합하기

        text = f"Score : {int(score)}\nLevel : {level}"#이미지에 표현할 문자
        y0, dy = 250, 30  # 시작 y 좌표와 줄 간격
        x0=880#문자 겉 테두리 상자 x좌표
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy#줄간격격
            cv.putText(poly_line_img, line, (x0, y), cv.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 1, cv.LINE_AA)
            
        cv.rectangle(poly_line_img, (x0 - 10, y0 - 30), (x0 + 180, y0 + 50), (0, 255, 0), 2)#문자 겉 테두리에 들어갈 상자
        
        # 곡선 그리기
        cv.polylines(poly_line_img, curve_fit, False, (0, 255, 0), 2)#초록색으로 표현
        # cv.polylines(poly_line_img, curve_fit_offset, False, (255, 255, 0), 2)
        # cv.fillPoly(poly_line_img, [pts_total], (255,0,255))  # 전체를 흰색(255)으로 채움


        # Display Image
        # cv.namedWindow('source', cv.WINDOW_AUTOSIZE) 
        # cv.imshow('source',img)
        # cv.imshow("r", r)
        # cv.namedWindow('BLUR', cv.WINDOW_AUTOSIZE) 
        # cv.imshow('BLUR',blur)
        # cv.namedWindow('Edge', cv.WINDOW_AUTOSIZE) 
        # cv.imshow('Edge',edge)
        # cv.namedWindow('draw con', cv.WINDOW_AUTOSIZE) 
        # cv.imshow('draw con',src_img)
        cv.namedWindow(f'Result[{idx}]', cv.WINDOW_AUTOSIZE) 
        cv.imshow(f'Result[{idx}]',poly_line_img)
        out.write(poly_line_img)


        # print("score: ",int(score),"\nLevel: ", level)
        count=count+1

        #cv.waitKey(0)
        if cv.waitKey(30) & 0xFF == 27:
            print('Press ESC to stop')
            break

    cv.destroyAllWindows()
    cap.release()
    out.release()

#함수 호출
process_image("LV1.png", 1)
process_image("LV2.png", 2)
process_image("LV3.png", 3)
process_image("LV1_simple.png", 4)
process_image("LV2_simple.png", 5)
process_image("LV3_simple.png", 6)
process_video('LAB3_Video.mp4',7)
cv.waitKey(0)
