/* ------------------------------------------------------ /
*Image Proccessing with Deep Learning
* LAB: Grayscale Image Segmentation -Gear
* Created : 2025-03-22
* Name: Junjae Lee
------------------------------------------------------ */

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void inverse(Mat& output) {//RGB이면서 색을 inverse 시키기
	for (int i = 0; i < output.rows; i++) {
		for (int k = 0; k < output.cols; k++) {
			// Invert each channel (Blue, Green, Red)
			output.at<Vec3b>(i, k)[0] = 255 - output.at<Vec3b>(i, k)[0]; // Blue 
			output.at<Vec3b>(i, k)[1] = 255 - output.at<Vec3b>(i, k)[1]; // Green
			output.at<Vec3b>(i, k)[2] = 255 - output.at<Vec3b>(i, k)[2]; // Red 

		}
	}
}

void RGB_Thresh(Mat& src, Mat& output, int threshold) {//RGB이면서 특정값을 기준으로 흰색과 검정으로 나누기
	//기어 색깔 rgb로 적용하기
	for (int i = 0; i < output.rows; i++) {
		for (int k = 0; k < output.cols; k++) {
			if (src.at<uchar>(i, k) > threshold) {
				output.at<Vec3b>(i, k)[0] = 255;
				output.at<Vec3b>(i, k)[1] = 255;
				output.at<Vec3b>(i, k)[2] = 255;
			}
			else {
				output.at<Vec3b>(i, k)[0] = 0;
				output.at<Vec3b>(i, k)[1] = 0;
				output.at<Vec3b>(i, k)[2] = 0;
			}

		}
	}
}

void FilterandContour(Mat& src, Point anchor, int r1, int r2, int r3, int idx) {
	Mat binary;

	threshold(src, binary, 128, 255, THRESH_BINARY);//binary 변환, 임의로 threshold 값 128설정
	imshow("binary[" + to_string(idx) + "]", binary);//기어 1 binary 결과 보기

	vector<vector<Point>> contours;//기어 이빨을 확인하기 위한 contour

	//Point anchor = Point(binary.cols / 2 - 16, binary.rows / 2 - 68);//원의 중심점

	circle(binary, anchor, r1, 0, -2, LINE_8, 0);//검은색 원 그리기
	imshow("circle_fill[" + to_string(idx) + "]", binary);//binary처리한 사진에 검은색 원을 추가해 이빨만 보이도록 한 사진

	/// Find contours
	findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	/// Draw all contours excluding holes
	Mat drawing(binary.size(), CV_8U, Scalar(255));//contour 결과를 보기 위해, 흰색 배경
	drawContours(drawing, contours, -1, Scalar(0), 2);//이빨의 테두리만 검은색으로 표현
	imshow("countour[" + to_string(idx) + "]", drawing);

	int count = 0;//이빨 개수
	int fail_count = 0;//문제가 있는 기어의 수
	float area_sum = 0;//전체 기어의 면적
	Scalar contourColor;//이빨 테두리 색
	Scalar textColor;//글씨 색

	Mat drawing_color = Mat::zeros(binary.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정
	Mat output = Mat::zeros(src.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정

	RGB_Thresh(src, output, 200);//RGB이면서 흰색과 검정색만 갖도록

	circle(output, anchor, r2, Scalar(120, 120, 120), -2, LINE_8, 0);//중간 크기 회색 원 그리기
	circle(output, anchor, r3, Scalar(0, 0, 0), -2, LINE_8, 0);//제일작은 검은색 원 그리기
	imshow("circle_fill_output[" + to_string(idx) + "]", output);//rgb로 바꾼 사진

	inverse(output);//RGB inverse시키기

	threshold(output, output, 200, 255, 2);//Threshold Truncated
	imshow("output[" + to_string(idx) + "]", output);

	Point center;
	Point text_point;
	int radius = 32; // Circle radius
	int num_points = 12;// Point 수
	float angle = 0;//원을 그릴 각도
	int x = 0;//원의 x좌표
	int y = 0;//원의 y좌표

	for (int i = 0; i < contours.size(); i++)
	{
		count++;//이빨 개수 추가
		printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//해당 기어 이빨의 면적과 길이를 출력

		//if (contourArea(contours[i]) < 1000) {//기어 면적이 1000보다 작은 경우
		//	fail_count++;//문제가 있는 이빨 개수 추가
		//	contourColor = Scalar(0, 0, 255); // 빨간색 (BGR 순서)

		//	//점선 원의 중심 좌표 구하기
		//	if (contourArea(contours[i]) < 400) {
		//		center = (contours[i][contours[i].size() - 1] + contours[i][0]) / 2;
		//	}
		//	else {
		//		center = (contours[i][contours[i].size() - 11]);
		//	}
		//	for (int num = 0; num < num_points; num++) {
		//		//원 테두리 점 계산
		//		angle = 2 * CV_PI * num / num_points;
		//		x = center.x + radius * cos(angle);
		//		y = center.y + radius * sin(angle);

		//		//노란색 원 그리기
		//		circle(output, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
		//	}

		//}
		if (contourArea(contours[i]) < 1000 || contourArea(contours[i]) > 1500) {
			fail_count++;//문제가 있는 이빨 개수 추가
			contourColor = Scalar(0, 0, 255); // 빨간색 (BGR 순서)
			switch (idx) {
			case 1:
			case 2:
				//점선 원의 중심 좌표 구하기
				if (contourArea(contours[i]) < 400) {
					center = (contours[i][contours[i].size() - 1] + contours[i][0]) / 2;
				}
				else {
					center = (contours[i][contours[i].size() - 11]);
				}
				for (int num = 0; num < num_points; num++) {
					//원 테두리 점 계산
					angle = 2 * CV_PI * num / num_points;
					x = center.x + radius * cos(angle);
					y = center.y + radius * sin(angle);

					//노란색 원 그리기
					circle(output, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
				}
				break;

			case 3: //점선 원의 중심 좌표 구하기
				if (contourArea(contours[i]) < 1000) {
					center = (contours[i][contours[i].size() - 11]);
				}
				else if (contourArea(contours[i]) < 1840) {
					center = (contours[i][30] + contours[i][0]) / 2;
				}
				else if (contourArea(contours[i]) < 1860) {
					center = (contours[i][30] + contours[i][0]) / 2;
				}
				else {
					center = (contours[i][30] + contours[i][0]) / 2;
				}
				for (int num = 0; num < num_points; num++) {
					//원 테두리 점 계산
					angle = 2 * CV_PI * num / num_points;
					x = center.x + radius * cos(angle);
					y = center.y + radius * sin(angle);

					//노란색 원 그리기
					circle(output, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
				}
				break;
			case 4:
				if (contourArea(contours[i]) < 1000 || contourArea(contours[i]) > 1500) {//면적이 1000이하거나 1500 이상일 때
					fail_count++;//문제가 있는 이빨 개수 추가
					contourColor = Scalar(0, 0, 255); // 빨간색 (BGR 순서)

					//점선 테두리를 가진 원의 중심점
					if (contourArea(contours[i]) <1800) {
						center = (contours[i][30] + contours[i][0]) / 2;
					}
					else {
						center= (contours[i][35] + contours[i][0]) / 2;
					}
					//점선 원 그리기
					for (int num = 0; num < num_points; num++) {
						//원 테두리 점 계산
						angle = 2 * CV_PI * num / num_points;
						x = center.x + radius * cos(angle);//점의 x좌표
						y = center.y + radius * sin(angle);//점의 y좌표

						//노란색 원 그리기
						circle(output, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
					}
				}
				break;
			}
		}
		else {
			contourColor = Scalar(0, 255, 0); // 초록색 (BGR 순서)
		}

		area_sum += contourArea(contours[i]); // 전체 면적 구하기

		// 윤곽선 그리기
		drawContours(drawing_color, contours, i, contourColor, 2); // i 번째 윤곽선에 대해 색을 지정하여 그리기

	}

	Mat add_text = drawing_color.clone();

	for (int i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) < 1000 || contourArea(contours[i]) > 1500) {//기어 면적이 1000보다 작거나 1500보다 큰 경우
			textColor = Scalar(0, 0, 255);

		}
		else {
			textColor = Scalar(255, 255, 255);
		}

		text_point = contours[i][0];
		std::string area_text = std::to_string((int)contourArea(contours[i]));
		putText(add_text, area_text, text_point, FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1, LINE_8); // 면적을 이미지에 표시

	}

	imshow("add_text[" + to_string(idx) + "]", add_text);
	// 이빨의 전체 면적 숫자를 drawing 이미지에 추가

	imshow("add_color circle[" + to_string(idx) + "]", output);//문제가 있는 부분 점선 원으로 표현한 사진
	area_sum = area_sum / count;//기어의 평균 면적 구하기
	imshow("color contour[" + to_string(idx) + "]", drawing_color);//문제가 없는 경우 초록색, 문제가 있는 경우 빨간색 처리한 사진

	printf("Teeth numbers: %d\n", count);//이빨 개수 출력
	printf("Avg. Teeth Area: %.2f\n", area_sum);//이빨의 전체 면적 출력
	printf("Defective Teeth: %d\n", fail_count);//문제가 있는 이빨 개수 출력
	if (fail_count > 0) {
		printf("Quality: FAIL\n\n");//문제가 있는 경우
	}
	else if (fail_count == 0) {
		printf("Quality: PASS\n\n");//문제가 없는 경우
	}
}




void main()
{
	//기어 1
	cv::Mat src, binary;//행렬 생성
	src = cv::imread("Gear1.jpg", 0);
	Point anchor = Point(src.cols / 2 - 16, src.rows / 2 - 68);//원의 중심점
	int r1 = 169;
	int r2 = 140;
	int r3 = 70;
	
	FilterandContour(src, anchor, r1, r2, r3, 1);

	//threshold(src, binary, 128, 255, THRESH_BINARY);//binary 변환, 임의로 threshold 값 128설정
	//imshow("binary", binary);//기어 1 binary 결과 보기
	//
	//vector<vector<Point>> contours;//기어 이빨을 확인하기 위한 contour
	//
	//Point anchor = Point(binary.cols / 2 - 16, binary.rows / 2 - 68);//원의 중심점

	//circle(binary, anchor, 169, 0, -2, LINE_8, 0);//검은색 원 그리기
	//imshow("circle_fill", binary);//binary처리한 사진에 검은색 원을 추가해 이빨만 보이도록 한 사진

	///// Find contours
	//findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	///// Draw all contours excluding holes
	//Mat drawing(binary.size(), CV_8U, Scalar(255));//contour 결과를 보기 위해, 흰색 배경
	//drawContours(drawing, contours, -1, Scalar(0), 2);//이빨의 테두리만 검은색으로 표현
	//imshow("countour", drawing);


	//int count = 0;//이빨 개수
	//int fail_count = 0;//문제가 있는 기어의 수
	//float area_sum = 0;//전체 기어의 면적
	//Scalar contourColor;//이빨 테두리 색
	//Scalar textColor;//글씨 색

	//Mat drawing_color = Mat::zeros(binary.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정

	//Mat output = Mat::zeros(src.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정

	//RGG_Thresh(src,output,200);
	//////기어 색깔 rgb로 적용하기
	////for (int i = 0; i < output.rows; i++) {
	////	for (int k = 0; k < output.cols; k++) {
	////		if (src.at<uchar>(i,k)>200) {
	////			output.at<Vec3b>(i, k)[0] = 255;
	////			output.at<Vec3b>(i, k)[1] = 255;
	////			output.at<Vec3b>(i, k)[2] = 255;
	////		}
	////		else {
	////			output.at<Vec3b>(i, k)[0] = 0;
	////			output.at<Vec3b>(i, k)[1] = 0;
	////			output.at<Vec3b>(i, k)[2] = 0;
	////		}

	////	}
	////}

	//circle(output, anchor, 140, Scalar(120,120,120), -2, LINE_8, 0);//중간 크기 회색 원 그리기
	//circle(output, anchor, 70, Scalar(0,0,0), -2, LINE_8, 0);//제일작은 검은색 원 그리기
	//imshow("circle_fill_output", output);//rgb로 바꾼 사진

	//inverse(output);
	//배경을 흰색으로 변경하기
	//for (int i = 0; i < output.rows; i++) {
	//	for (int k = 0; k < output.cols; k++) {
	//		// Invert each channel (Blue, Green, Red)
	//		output.at<Vec3b>(i, k)[0] = 255 - output.at<Vec3b>(i, k)[0]; // Blue 
	//		output.at<Vec3b>(i, k)[1] = 255 - output.at<Vec3b>(i, k)[1]; // Green
	//		output.at<Vec3b>(i, k)[2] = 255 - output.at<Vec3b>(i, k)[2]; // Red 

	//	}
	//}

	//threshold(output, output, 200, 255, 2);//Threshold Truncated
	//imshow("output", output);

	//Point center;
	//Point text_point;
	//int radius = 32; // Circle radius
	//int num_points = 12;// Point 수
	//float angle = 0;//원을 그릴 각도
	//int x = 0;//원의 x좌표
	//int y = 0;//원의 y좌표

	//for (int i = 0; i < contours.size(); i++)
	//{
	//	count++;//이빨 개수 추가
	//	printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//해당 기어 이빨의 면적과 길이를 출력
	//	
	//	if (contourArea(contours[i]) < 1000) {//기어 면적이 1000보다 작은 경우
	//		fail_count++;//문제가 있는 이빨 개수 추가
	//		contourColor = Scalar(0, 0, 255); // 빨간색 (BGR 순서)

	//		//점선 원의 중심 좌표 구하기
	//		if (contourArea(contours[i]) < 400) {
	//			center = (contours[i][contours[i].size() - 1]+ contours[i][0])/2;
	//		}
	//		else {
	//			center = (contours[i][contours[i].size() - 11]);
	//		}
	//		for (int num = 0; num < num_points; num++) {
	//			//원 테두리 점 계산
	//			angle = 2 * CV_PI * num / num_points;
	//			x = center.x + radius * cos(angle);
	//			y = center.y + radius * sin(angle);

	//			//노란색 원 그리기
	//			circle(output, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
	//		}
	//		
	//	}
	//	else {
	//		contourColor = Scalar(0, 255, 0); // 초록색 (BGR 순서)
	//	}
	//	
	//	area_sum+= contourArea(contours[i]); // 전체 면적 구하기

	//	// 윤곽선 그리기
	//	drawContours(drawing_color, contours, i, contourColor, 2); // i 번째 윤곽선에 대해 색을 지정하여 그리기

	//}

	//Mat add_text = drawing_color.clone();

	//for (int i = 0; i < contours.size(); i++)
	//{

	//	if (contourArea(contours[i]) < 1000) {//기어 면적이 1000보다 작은 경우
	//		textColor = Scalar(0, 0, 255);

	//	}
	//	else {
	//		textColor = Scalar(255, 255, 255);
	//	}

	//	text_point = contours[i][0];
	//	std::string area_text = std::to_string((int)contourArea(contours[i]));
	//	putText(add_text, area_text, text_point, FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1, LINE_8); // 면적을 이미지에 표시

	//}

	//imshow("add_text",add_text);
	//// 이빨의 전체 면적 숫자를 drawing 이미지에 추가
	//
	//imshow("add_color circle",output);//문제가 있는 부분 점선 원으로 표현한 사진
	//area_sum = area_sum / count;//기어의 평균 면적 구하기
	//imshow("color contour", drawing_color);//문제가 없는 경우 초록색, 문제가 있는 경우 빨간색 처리한 사진
	//
	//printf("Teeth numbers: %d\n", count);//이빨 개수 출력
	//printf("Avg. Teeth Area: %.2f\n", area_sum);//이빨의 전체 면적 출력
	//printf("Defective Teeth: %d\n",fail_count);//문제가 있는 이빨 개수 출력
	//if (fail_count>0) {
	//	printf("Quality: FAIL\n\n");//문제가 있는 경우
	//}
	//else if (fail_count == 0) {
	//	printf("Quality: PASS\n\n");//문제가 없는 경우
	//}
	

	//기어 2
	src = cv::imread("Gear2.jpg", 0);
	anchor = Point(src.cols / 2 - 16, src.rows / 2 - 5);//원의 중심점
	r1 = 169;
	r2 = 140;
	r3 = 70;
	FilterandContour(src, anchor, r1, r2, r3, 2);
	//// 기어 2
	//src = cv::imread("Gear2.jpg", 0);

	//threshold(src, binary, 128, 255, THRESH_BINARY);//binary 변환, 임의로 threshold 값 128설정
	//imshow("binary_2", binary);//기어 2 binary 결과 보기

	//anchor = Point(binary.cols / 2 - 16, binary.rows / 2 -5);//원의 중심점

	//circle(binary, anchor, 169, 0, -2, LINE_8, 0);//검은색 원 그리기
	//imshow("circle_fill_2", binary);//binary처리한 사진에 검은색 원을 추가해 이빨만 보이도록 한 사진

	///// Find contours
	//findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	///// Draw all contours excluding holes
	//Mat drawing_1(binary.size(), CV_8U, Scalar(255));//contour 결과를 보기 위해, 흰색 배경
	//drawContours(drawing_1, contours, -1, Scalar(0), 2);//이빨의 테두리만 검은색으로 표현
	//imshow("countour_2", drawing_1);

	//count = 0;//이빨 개수
	//fail_count = 0;//문제가 있는 기어의 수
	//area_sum = 0;//전체 기어의 면적
	//
	//Mat drawing_color_1 = Mat::zeros(binary.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정

	//Mat output_1 = Mat::zeros(src.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정

	////기어 색깔 rgb로 적용하기
	//for (int i = 0; i < output_1.rows; i++) {
	//	for (int k = 0; k < output_1.cols; k++) {
	//		if (src.at<uchar>(i, k) > 200) {
	//			output_1.at<Vec3b>(i, k)[0] = 255;
	//			output_1.at<Vec3b>(i, k)[1] = 255;
	//			output_1.at<Vec3b>(i, k)[2] = 255;
	//		}
	//		else {
	//			output_1.at<Vec3b>(i, k)[0] = 0;
	//			output_1.at<Vec3b>(i, k)[1] = 0;
	//			output_1.at<Vec3b>(i, k)[2] = 0;
	//		}

	//	}
	//}

	//circle(output_1, anchor, 140, Scalar(120, 120, 120), -2, LINE_8, 0);//중간 크기 회색 원 그리기
	//circle(output_1, anchor, 70, Scalar(0, 0, 0), -2, LINE_8, 0);//제일작은 검은색 원 그리기
	//imshow("circle_fill_output", output_1);//rgb로 바꾼 사진

	////배경을 흰색으로 변경하기
	//for (int i = 0; i < output_1.rows; i++) {
	//	for (int k = 0; k < output_1.cols; k++) {
	//		// Invert each channel (Blue, Green, Red)
	//		output_1.at<Vec3b>(i, k)[0] = 255 - output_1.at<Vec3b>(i, k)[0]; // Blue 
	//		output_1.at<Vec3b>(i, k)[1] = 255 - output_1.at<Vec3b>(i, k)[1]; // Green 
	//		output_1.at<Vec3b>(i, k)[2] = 255 - output_1.at<Vec3b>(i, k)[2]; // Red 

	//	}
	//}

	//threshold(output_1, output_1, 200, 255, 2);//Threshold Truncated
	//imshow("output", output_1);

	//for (int i = 0; i < contours.size(); i++)
	//{
	//	count++;//이빨 개수 추가
	//	printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//해당 기어 이빨의 면적과 길이를 출력

	//	if (contourArea(contours[i]) < 1000) {//기어 면적이 1000보다 작은 경우
	//		fail_count++;//문제가 있는 이빨 개수 추가
	//		contourColor = Scalar(0, 0, 255); // 빨간색 (BGR 순서)
	//		//점선 원의 중심 좌표 구하기
	//		if (contourArea(contours[i]) < 400) {
	//			center = (contours[i][contours[i].size() - 1] + contours[i][0]) / 2;
	//		}
	//		else {
	//			center = (contours[i][contours[i].size() - 11]);
	//		}
	//		for (int num = 0; num < num_points; num++) {
	//			//원 테두리 점 계산
	//			angle = 2 * CV_PI * num / num_points;
	//			x = center.x + radius * cos(angle);
	//			y = center.y + radius * sin(angle);

	//			//노란색 원 그리기
	//			circle(output_1, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
	//		}
	//	}
	//	else {
	//		contourColor = Scalar(0, 255, 0); // 초록색 (BGR 순서)
	//	}

	//	area_sum += contourArea(contours[i]); // 전체 면적 구하기

	//	// 윤곽선 그리기
	//	drawContours(drawing_color_1, contours, i, contourColor, 2); // i 번째 윤곽선에 대해 색을 지정하여 그리기

	//}
	//Mat add_text_1 = drawing_color_1.clone();

	//for (int i = 0; i < contours.size(); i++)
	//{

	//	if (contourArea(contours[i]) < 1000) {//기어 면적이 1000보다 작은 경우
	//		textColor = Scalar(0, 0, 255);

	//	}
	//	else {
	//		textColor = Scalar(255, 255, 255);
	//	}

	//	text_point = contours[i][0];
	//	std::string area_text = std::to_string((int)contourArea(contours[i]));
	//	putText(add_text_1, area_text, text_point, FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1, LINE_8); // 면적을 이미지에 표시

	//}

	//imshow("add_text_2", add_text_1);

	//imshow("add_color circle_2", output_1);//문제가 있는 부분 점선 원으로 표현한 사진
	//area_sum = area_sum / count;//기어의 평균 면적 구하기
	//imshow("color contour_2", drawing_color_1);//문제가 없는 경우 초록색, 문제가 있는 경우 빨간색 처리한 사진

	//printf("Teeth numbers: %d\n", count);//이빨 개수 출력
	//printf("Avg. Teeth Area: %.2f\n", area_sum);//이빨의 전체 면적 출력
	//printf("Defective Teeth: %d\n", fail_count);//문제가 있는 이빨 개수 출력
	//if (fail_count > 0) {
	//	printf("Quality: FAIL\n\n");//문제가 있는 경우
	//}
	//else if (fail_count == 0) {
	//	printf("Quality: PASS\n\n");//문제가 없는 경우
	//}

	//기어 3
	src = cv::imread("Gear3.jpg", 0);
	anchor = Point(src.cols / 2 + 22, src.rows / 2 - 15);//원의 중심점
	r1 = 187;
	r2 = 140;
	r3 = 70;
	FilterandContour(src, anchor, r1, r2, r3, 3);
	//// 기어 3
	//src = cv::imread("Gear3.jpg", 0);

	//threshold(src, binary, 128, 255, THRESH_BINARY);//binary 변환, 임의로 threshold 값 128설정
	//imshow("binary_3", binary);//기어 3 binary 결과 보기

	//anchor = Point(binary.cols / 2 +22, binary.rows / 2 - 15);//원의 중심점

	//circle(binary, anchor, 187, 0, -2, LINE_8, 0);//검은색 원 그리기
	//imshow("circle_fill_3", binary);//binary처리한 사진에 검은색 원을 추가해 이빨만 보이도록 한 사진

	///// Find contours
	//findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	///// Draw all contours excluding holes
	//Mat drawing_2(binary.size(), CV_8U, Scalar(255));//contour 결과를 보기 위해, 흰색 배경
	//drawContours(drawing_2, contours, -1, Scalar(0), 2);//이빨의 테두리만 검은색으로 표현
	//imshow("countour_3", drawing_2);

	//count = 0;//이빨 개수
	//fail_count = 0;//문제가 있는 기어의 수
	//area_sum = 0;//전체 기어의 면적

	//Mat drawing_color_2 = Mat::zeros(binary.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정

	//Mat output_2 = Mat::zeros(src.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정

	////기어 색깔 rgb로 적용하기
	//for (int i = 0; i < output_2.rows; i++) {
	//	for (int k = 0; k < output_2.cols; k++) {
	//		if (src.at<uchar>(i, k) > 200) {
	//			output_2.at<Vec3b>(i, k)[0] = 255;
	//			output_2.at<Vec3b>(i, k)[1] = 255;
	//			output_2.at<Vec3b>(i, k)[2] = 255;
	//		}
	//		else {
	//			output_2.at<Vec3b>(i, k)[0] = 0;
	//			output_2.at<Vec3b>(i, k)[1] = 0;
	//			output_2.at<Vec3b>(i, k)[2] = 0;
	//		}

	//	}
	//}

	//circle(output_2, anchor, 140, Scalar(120, 120, 120), -2, LINE_8, 0);//중간 크기 회색 원 그리기
	//circle(output_2, anchor, 70, Scalar(0, 0, 0), -2, LINE_8, 0);//제일작은 검은색 원 그리기
	//imshow("circle_fill_output_3", output_2);//rgb로 바꾼 사진

	////배경을 흰색으로 변경하기
	//for (int i = 0; i < output_2.rows; i++) {
	//	for (int k = 0; k < output_2.cols; k++) {
	//		// Invert each channel (Blue, Green, Red)
	//		output_2.at<Vec3b>(i, k)[0] = 255 - output_2.at<Vec3b>(i, k)[0]; // Blue 
	//		output_2.at<Vec3b>(i, k)[1] = 255 - output_2.at<Vec3b>(i, k)[1]; // Green 
	//		output_2.at<Vec3b>(i, k)[2] = 255 - output_2.at<Vec3b>(i, k)[2]; // Red 

	//	}
	//}

	//threshold(output_2, output_2, 200, 255, 2);//Threshold Truncated
	//imshow("output_3", output_2);

	//for (int i = 0; i < contours.size(); i++)
	//{
	//	count++;//이빨 개수 추가
	//	printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//해당 기어 이빨의 면적과 길이를 출력


	//	if (contourArea(contours[i]) < 1000 || contourArea(contours[i]) > 1500) {
	//		fail_count++;//문제가 있는 이빨 개수 추가
	//		contourColor = Scalar(0, 0, 255); // 빨간색 (BGR 순서)

	//		//점선 원의 중심 좌표 구하기
	//		if (contourArea(contours[i]) < 1000) {
	//			center = (contours[i][contours[i].size() - 11]);
	//		}
	//		else if (contourArea(contours[i]) < 1840) {
	//			center = (contours[i][30] + contours[i][0]) / 2;
	//		}
	//		else if (contourArea(contours[i]) < 1860) {
	//			center = (contours[i][30] + contours[i][0]) / 2;
	//		}
	//		else {
	//			center = (contours[i][30] + contours[i][0]) / 2;
	//		}
	//		for (int num = 0; num < num_points; num++) {
	//			//원 테두리 점 계산
	//			angle = 2 * CV_PI * num / num_points;
	//			x = center.x + radius * cos(angle);
	//			y = center.y + radius * sin(angle);

	//			//노란색 원 그리기
	//			circle(output_2, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
	//		}
	//	}
	//	else {
	//		contourColor = Scalar(0, 255, 0); // 초록색 (BGR 순서)
	//	}

	//	area_sum += contourArea(contours[i]); // 전체 면적 구하기

	//	// 윤곽선 그리기
	//	drawContours(drawing_color_2, contours, i, contourColor, 2); // i 번째 윤곽선에 대해 색을 지정하여 그리기

	//}
	//Mat add_text_2 = drawing_color_2.clone();

	//for (int i = 0; i < contours.size(); i++)
	//{

	//	if (contourArea(contours[i]) < 1000) {//기어 면적이 1000보다 작은 경우
	//		textColor = Scalar(0, 0, 255);

	//	}
	//	else {
	//		textColor = Scalar(255, 255, 255);
	//	}

	//	text_point = contours[i][0];
	//	std::string area_text = std::to_string((int)contourArea(contours[i]));
	//	putText(add_text_2, area_text, text_point, FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1, LINE_8); // 면적을 이미지에 표시

	//}

	//imshow("add_text_3", add_text_2);

	//imshow("add_color circle_3", output_2);//문제가 있는 부분 점선 원으로 표현한 사진
	//area_sum = area_sum / count;//기어의 평균 면적 구하기
	//imshow("color contour_3", drawing_color_2);//문제가 없는 경우 초록색, 문제가 있는 경우 빨간색 처리한 사진

	//printf("Teeth numbers: %d\n", count);//이빨 개수 출력
	//printf("Avg. Teeth Area: %.2f\n", area_sum);//이빨의 전체 면적 출력
	//printf("Defective Teeth: %d\n", fail_count);//문제가 있는 이빨 개수 출력
	//if (fail_count > 0) {
	//	printf("Quality: FAIL\n\n");//문제가 있는 경우
	//}
	//else if (fail_count == 0) {
	//	printf("Quality: PASS\n\n");//문제가 없는 경우
	//}
	//기어 3
	src = cv::imread("Gear4.jpg", 0);
	anchor = Point(src.cols / 2 - 74, src.rows / 2 - 32);//원의 중심점
	r1 = 188;
	r2 = 140;
	r3 = 70;
	FilterandContour(src, anchor, r1, r2, r3, 4);
	//// 기어 4
	//src = cv::imread("Gear4.jpg", 0);

	//threshold(src, binary, 128, 255, THRESH_BINARY);//binary 변환, 임의로 threshold 값 128설정
	//imshow("binary_4", binary);//기어 4 binary 결과 보기

	//anchor = Point(binary.cols / 2 -74, binary.rows / 2 - 32);//원의 중심점

	//circle(binary, anchor, 188, 0, -2, LINE_8, 0);//검은색 원 그리기
	//imshow("circle_fill_4", binary);//binary처리한 사진에 검은색 원을 추가해 이빨만 보이도록 한 사진

	///// Find contours
	//findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	///// Draw all contours excluding holes
	//Mat drawing_3(binary.size(), CV_8U, Scalar(255));//contour 결과를 보기 위해, 흰색 배경
	//drawContours(drawing_3, contours, -1, Scalar(0), 2);//이빨의 테두리만 검은색으로 표현
	//imshow("countour_4", drawing_3);

	//count = 0;//이빨 개수
	//fail_count = 0;//문제가 있는 기어의 수
	//area_sum = 0;//전체 기어의 면적

	//Mat drawing_color_3 = Mat::zeros(binary.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정

	//Mat output_3 = Mat::zeros(src.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정

	////기어 색깔 rgb로 적용하기
	//for (int i = 0; i < output_3.rows; i++) {
	//	for (int k = 0; k < output_3.cols; k++) {
	//		if (src.at<uchar>(i, k) > 200) {
	//			output_3.at<Vec3b>(i, k)[0] = 255;
	//			output_3.at<Vec3b>(i, k)[1] = 255;
	//			output_3.at<Vec3b>(i, k)[2] = 255;
	//		}
	//		else {
	//			output_3.at<Vec3b>(i, k)[0] = 0;
	//			output_3.at<Vec3b>(i, k)[1] = 0;
	//			output_3.at<Vec3b>(i, k)[2] = 0;
	//		}

	//	}
	//}

	//circle(output_3, anchor, 140, Scalar(120, 120, 120), -2, LINE_8, 0);//중간 크기 회색 원 그리기
	//circle(output_3, anchor, 70, Scalar(0, 0, 0), -2, LINE_8, 0);//제일작은 검은색 원 그리기
	//imshow("circle_fill_output", output_3);//rgb로 바꾼 사진

	////배경을 흰색으로 변경하기
	//for (int i = 0; i < output_3.rows; i++) {
	//	for (int k = 0; k < output_3.cols; k++) {
	//		// Invert each channel (Blue, Green, Red)
	//		output_3.at<Vec3b>(i, k)[0] = 255 - output_3.at<Vec3b>(i, k)[0]; // Blue 
	//		output_3.at<Vec3b>(i, k)[1] = 255 - output_3.at<Vec3b>(i, k)[1]; // Green 
	//		output_3.at<Vec3b>(i, k)[2] = 255 - output_3.at<Vec3b>(i, k)[2]; // Red 

	//	}
	//}

	//threshold(output_3, output_3, 200, 255, 2);//Threshold Truncated
	//imshow("output", output_3);


	//for (int i = 0; i < contours.size(); i++)
	//{
	//	count++;//이빨 개수 추가
	//	printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//해당 기어 이빨의 면적과 길이를 출력


	//	if (contourArea(contours[i]) < 1000 || contourArea(contours[i]) > 1500) {//면적이 1000이하거나 1500 이상일 때
	//		fail_count++;//문제가 있는 이빨 개수 추가
	//		contourColor = Scalar(0, 0, 255); // 빨간색 (BGR 순서)

	//		//점선 테두리를 가진 원의 중심점
	//		if (contourArea(contours[i]) <1800) {
	//			center = (contours[i][30] + contours[i][0]) / 2;
	//		}
	//		else {
	//			center= (contours[i][35] + contours[i][0]) / 2;
	//		}
	//		//점선 원 그리기
	//		for (int num = 0; num < num_points; num++) {
	//			//원 테두리 점 계산
	//			angle = 2 * CV_PI * num / num_points;
	//			x = center.x + radius * cos(angle);//점의 x좌표
	//			y = center.y + radius * sin(angle);//점의 y좌표

	//			//노란색 원 그리기
	//			circle(output_3, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
	//		}
	//	}
	//	else {
	//		contourColor = Scalar(0, 255, 0); // 초록색 (BGR 순서)
	//	}

	//	area_sum += contourArea(contours[i]); // 전체 면적 구하기

	//	// 윤곽선 그리기
	//	drawContours(drawing_color_3, contours, i, contourColor, 2); // i 번째 윤곽선에 대해 색을 지정하여 그리기

	//}


	//Mat add_text_3 = drawing_color_3.clone();

	//for (int i = 0; i < contours.size(); i++)
	//{

	//	if (contourArea(contours[i]) < 1000) {//기어 면적이 1000보다 작은 경우
	//		textColor = Scalar(0, 0, 255);

	//	}
	//	else {
	//		textColor = Scalar(255, 255, 255);
	//	}

	//	text_point = contours[i][0];
	//	std::string area_text = std::to_string((int)contourArea(contours[i]));
	//	putText(add_text_3, area_text, text_point, FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1, LINE_8); // 면적을 이미지에 표시

	//}

	//imshow("add_text_4", add_text_3);

	//imshow("add_color circle_4", output_3);//문제가 있는 부분 점선 원으로 표현한 사진
	//area_sum = area_sum / count;//기어의 평균 면적 구하기
	//imshow("color contour_4", drawing_color_3);//문제가 없는 경우 초록색, 문제가 있는 경우 빨간색 처리한 사진

	//printf("Teeth numbers: %d\n", count);//이빨 개수 출력
	//printf("Avg. Teeth Area: %.2f\n", area_sum);//이빨의 전체 면적 출력
	//printf("Defective Teeth: %d\n", fail_count);//문제가 있는 이빨 개수 출력
	//if (fail_count > 0) {
	//	printf("Quality: FAIL\n\n");//문제가 있는 경우
	//}
	//else if (fail_count == 0) {
	//	printf("Quality: PASS\n\n");//문제가 없는 경우
	//}


	cv::waitKey(0);
}
