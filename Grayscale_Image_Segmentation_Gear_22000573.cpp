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
				//흰색으로 칠하기
				output.at<Vec3b>(i, k)[0] = 255;
				output.at<Vec3b>(i, k)[1] = 255;
				output.at<Vec3b>(i, k)[2] = 255;
			}
			else {
				//검은색으로 칠하기
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
	cv::imshow("binary[" + to_string(idx) + "]", binary);//기어 1 binary 결과 보기

	vector<vector<Point>> contours;//기어 이빨을 확인하기 위한 contour

	circle(binary, anchor, r1, 0, -2, LINE_8, 0);//검은색 원 그리기
	cv::imshow("circle_fill[" + to_string(idx) + "]", binary);//binary처리한 사진에 검은색 원을 추가해 이빨만 보이도록 한 사진

	/// Find contours
	findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	/// Draw all contours excluding holes
	Mat drawing(binary.size(), CV_8U, Scalar(255));//contour 결과를 보기 위해, 흰색 배경
	cv::drawContours(drawing, contours, -1, Scalar(0), 2);//이빨의 테두리만 검은색으로 표현
	cv::imshow("countour[" + to_string(idx) + "]", drawing);

	int count = 0;//이빨 개수
	int fail_count = 0;//문제가 있는 기어의 수
	float area_sum = 0;//전체 기어의 면적
	Scalar contourColor;//이빨 테두리 색
	Scalar textColor;//글씨 색

	Mat drawing_color = Mat::zeros(src.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정
	Mat output = Mat::zeros(src.size(), CV_8UC3); // 색상을 적용하기 위해 CV_8UC3 설정

	RGB_Thresh(src, output, 200);//RGB이면서 흰색과 검정색만 갖도록

	circle(output, anchor, r2, Scalar(120, 120, 120), -2, LINE_8, 0);//중간 크기 회색 원 그리기
	circle(output, anchor, r3, Scalar(0, 0, 0), -2, LINE_8, 0);//제일작은 검은색 원 그리기
	cv::imshow("circle_fill_output[" + to_string(idx) + "]", output);//rgb로 바꾼 사진

	inverse(output);//RGB inverse시키기

	threshold(output, output, 200, 255, 2);//Threshold Truncated
	cv::imshow("output[" + to_string(idx) + "]", output);

	int radius = 32; // Circle radius
	int num_points = 12;// Point 수
	float angle = 0;//원을 그릴 각도
	int x = 0;//원의 x좌표
	int y = 0;//원의 y좌표

	double cx, cy;//기어의 중심점
	Moments mmt;//기어의 중심점을 찾기 위해

	for (int i = 0; i < contours.size(); i++)
	{
		//contour의 중심점 구하기
		mmt=moments(contours[i]);
		cx = mmt.m10 / mmt.m00, cy = mmt.m01 / mmt.m00;


		count++;//이빨 개수 추가
		printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//해당 기어 이빨의 면적과 길이를 출력

		if (contourArea(contours[i]) < 1000 || contourArea(contours[i]) > 1500) {
			fail_count++;//문제가 있는 이빨 개수 추가
			contourColor = Scalar(0, 0, 255); // 빨간색 (BGR 순서)
			for (int num = 0; num < num_points; num++) {
				//원 테두리 점 계산
				angle = 2 * CV_PI * num / num_points;
				x = cx + radius * cos(angle);
				y = cy + radius * sin(angle);

				//노란색 원 그리기
				circle(output, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
			}
		}
		else {
			contourColor = Scalar(0, 255, 0); // 초록색 (BGR 순서)
		}

		area_sum += contourArea(contours[i]); // 전체 면적 구하기

		// 윤곽선 그리기
		cv::drawContours(drawing_color, contours, i, contourColor, 2); // i 번째 윤곽선에 대해 색을 지정하여 그리기

	}

	Mat add_text = drawing_color.clone();//색상 표현한 것 복제하기

	for (int i = 0; i < contours.size(); i++)
	{
		mmt = moments(contours[i]);
		cx = mmt.m10 / mmt.m00, cy = mmt.m01 / mmt.m00;//기어 이빨의 중심점

		if (contourArea(contours[i]) < 1000 || contourArea(contours[i]) > 1500) {//기어 면적이 1000보다 작거나 1500보다 큰 경우
			textColor = Scalar(0, 0, 255); //빨간색 글자
		}
		else {
			textColor = Scalar(255, 255, 255);//흰색 글자
		}

		x = cx+(anchor.x-cx)*0.28;//글자를 적을 위치
		y = cy+(anchor.y-cy)*0.28;//글자를 적을 위치

		std::string area_text = std::to_string((int)contourArea(contours[i]));
		cv::putText(add_text, area_text, Point(x-15,y), FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, LINE_8); // 면적을 이미지에 표시

	}

	cv::imshow("add_text[" + to_string(idx) + "]", add_text);
	// 이빨의 전체 면적 숫자를 drawing 이미지에 추가

	cv::imshow("add_color circle[" + to_string(idx) + "]", output);//문제가 있는 부분 점선 원으로 표현한 사진
	area_sum = area_sum / count;//기어의 평균 면적 구하기
	cv::imshow("color contour[" + to_string(idx) + "]", drawing_color);//문제가 없는 경우 초록색, 문제가 있는 경우 빨간색 처리한 사진

	printf("Teeth numbers: %d\n", count);//이빨 개수 출력
	printf("Avg. Teeth Area: %.2f\n", area_sum);//이빨의 전체 면적 출력
	printf("Defective Teeth: %d\n", fail_count);//문제가 있는 이빨 개수 출력
	printf("Diameter of the gear: %d\n", r1);//이뿌리원 지름 출력
	if (fail_count > 0) {
		printf("Quality: FAIL\n\n");//문제가 있는 경우
	}
	else if (fail_count == 0) {
		printf("Quality: PASS\n\n");//문제가 없는 경우
	}
	
}

// Function to calculate and display the histogram
void calc_and_display_histogram(const Mat& src, int idx) {
	// Establish the number of bins
	int histSize = 256;

	// Set the ranges (0 to 256)
	float range[] = { 0, 256 }; // the upper boundary is exclusive
	const float* histRange = { range };

	// Set histogram parameters
	bool uniform = true, accumulate = false;

	// Compute the histogram
	Mat hist1D;
	calcHist(&src, 1, 0, Mat(), hist1D, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histogram
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage1D(hist_h, hist_w, CV_8UC1, Scalar(0));

	normalize(hist1D, hist1D, 0, histImage1D.rows, NORM_MINMAX, -1, Mat());

	// Draw for each bin
	for (int i = 1; i < histSize; i++) {
		line(histImage1D,
			Point(bin_w * (i - 1), hist_h - cvRound(hist1D.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(hist1D.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	// Display the histogram
	cv::imshow("calcHist Demo[" + to_string(idx) + "]", histImage1D);
}

// Function to load the image and call the histogram display function
void process_image_and_display_histogram(const Mat& src, int idx) {
	if (src.empty()) {
		cerr << "Error loading image!" << endl;
		return;
	}

	// Display source image
	cv::imshow("Source image[" + to_string(idx) + "]", src);

	// Call function to calculate and display histogram
	calc_and_display_histogram(src, idx);
}


void main()
{
	//기어 1
	cv::Mat src;//행렬 생성
	src = cv::imread("Gear1.jpg", 0);
	process_image_and_display_histogram(src, 1);//히스토그램 보기
	Point anchor = Point(src.cols / 2 - 16, src.rows / 2 - 68);//원의 중심점
	int r1 = 169;//기어 이빨만 남길 원 반지름
	int r2 = 140;//중간 크기의 원 반지름
	int r3 = 70;//제일 작은 검은색 원 반지름
	
	FilterandContour(src, anchor, r1, r2, r3, 1);//contour 찾고 색 표현

	//기어 2
	src = cv::imread("Gear2.jpg", 0);
	process_image_and_display_histogram(src, 2);//히스토그램 보기
	anchor = Point(src.cols / 2 - 16, src.rows / 2 - 5);//원의 중심점
	r1 = 169;//기어 이빨만 남길 원 반지름
	r2 = 140;//중간 크기의 원 반지름
	r3 = 70;//제일 작은 검은색 원 반지름
	FilterandContour(src, anchor, r1, r2, r3, 2);

	//기어 3
	src = cv::imread("Gear3.jpg", 0);
	process_image_and_display_histogram(src, 3);//히스토그램 보기
	anchor = Point(src.cols / 2 + 22, src.rows / 2 - 15);//원의 중심점
	r1 = 187;//기어 이빨만 남길 원 반지름
	r2 = 140;//중간 크기의 원 반지름
	r3 = 70;//제일 작은 검은색 원 반지름
	FilterandContour(src, anchor, r1, r2, r3, 3);//contour 찾고 색 표현

	//기어 4
	src = cv::imread("Gear4.jpg", 0);
	process_image_and_display_histogram(src, 4);//히스토그램 보기
	anchor = Point(src.cols / 2 - 74, src.rows / 2 - 32);//원의 중심점
	r1 = 188;//기어 이빨만 남길 원
	r2 = 140;//중간 크기의 원 반지름
	r3 = 70;//제일 작은 검은색 원 반지름
	FilterandContour(src, anchor, r1, r2, r3, 4);//contour 찾고 색 표현

	cv::waitKey(0);
}
