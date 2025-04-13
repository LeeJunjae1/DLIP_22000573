/* ------------------------------------------------------ /
*Image Proccessing with Deep Learning
* LAB: Color Image Segmentation - Magic Cloak
* Created : 2025-04-11
* Name: Junjae Lee
------------------------------------------------------ */

//#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <ctype.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat image;//선택할 이미지
Point origin;//마우스 좌표
Rect selection;//마우스 선택 영역
bool selectObject = false;
bool trackObject = false;
//int hmin = 13, hmax = 60, smin = 70, smax = 200, vmin = 90, vmax = 200;//SAMPLE 1값
//int hmin = 24, hmax = 101, smin = 40, smax = 255, vmin = 55, vmax = 255;
int hmin = 110, hmax = 176, smin = 87, smax = 255, vmin =70, vmax = 255;//LAB2 분홍색 물체 값
VideoWriter save_video;

/// On mouse event 
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)  // for any mouse motion
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = abs(x - origin.x) + 1;
		selection.height = abs(y - origin.y) + 1;
		selection &= Rect(0, 0, image.cols, image.rows);  /// Bitwise AND  check selectin is within the image coordinate
	}

	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		selectObject = true;
		origin = Point(x, y);
		break;
	case EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.area())
			trackObject = true;
		break;
	}
}
void processVideo(const string& videoPath)
{
	Mat image_disp, hsv, hue, mask, dst, src, background, mask_src, mask_src2;
	vector<vector<Point>> contours;//contour할 영역
	Mat dst_track, dst_out, final_out;

	VideoCapture cap(videoPath);
	if (!cap.isOpened()) {
		cout << "비디오를 열 수 없습니다: " << videoPath << endl;
		return;
	}

	//저장할 이미지 프레임과 너비
	int frame_width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
	double fps = cap.get(CAP_PROP_FPS);//해당 프레임

	int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');//avi 형식으로 저장
	string filename = "DLIP_LAB2_22000573_JunJaeLee_output.avi";  // 저장파일 이름

	save_video.open(filename, codec, fps, Size(frame_width, frame_height), true);
	if (!save_video.isOpened()) {
		cout << "!!! VideoWriter 초기화 실패: 파일을 열 수 없습니다." << endl;
		return;
	}

	// 트랙바와 마우스 콜백 설정
	namedWindow("Source", 0);
	setMouseCallback("Source", onMouse, 0);
	createTrackbar("Hmin", "Source", &hmin, 179);
	createTrackbar("Hmax", "Source", &hmax, 179);
	createTrackbar("Smin", "Source", &smin, 255);
	createTrackbar("Smax", "Source", &smax, 255);
	createTrackbar("Vmin", "Source", &vmin, 255);
	createTrackbar("Vmax", "Source", &vmax, 255);

	int element_shape = MORPH_RECT;
	int n = 7;//필터 사이즈 및 morphology 사이즈
	Mat element = getStructuringElement(element_shape, Size(n, n));

	int cap_image = 0;//처음 이미지를 저장하기 위한 변수

	while (true)
	{
		if (!cap.read(src)) {
			cout << "프레임을 가져올 수 없습니다.\n";
			break;
		}

		//imshow("source video", src);

		if (cap_image == 0) {//최초 프레임일때
			src.copyTo(background);//최초 배경이미지 저장
			cap_image++;//count 증가 시키기
		}
		//imshow("background", background);
		src.copyTo(image);//영상 복사하기
		cvtColor(src, hsv, COLOR_BGR2HSV);//HSV로 변환하기
		blur(hsv, hsv, Size(5, 5));//필터 적용

		inRange(hsv,
			Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
			Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)),
			dst);//해당 범위 있는 색상 찾기

		//imshow("InRange1", dst);
		morphologyEx(dst, dst, MORPH_OPEN, element); //opening 실시
		//dilate(dst, dst, element);//DIALATE를 통해 중간에 빈 부분 채우기
		dst_out = Mat::zeros(dst.size(), CV_8UC3);//MASK를 할 부분
		final_out = Mat::zeros(dst.size(), CV_8UC3);//최종 결과물

		imshow("InRange", dst);

		//트랙바로 inrange 값 조정하기
		if (trackObject) {
			trackObject = false;
			Mat roi_HSV(hsv, selection);
			Scalar means, stddev;
			meanStdDev(roi_HSV, means, stddev);

			hmin = MAX((means[0] - stddev[0]), 0);
			hmax = MIN((means[0] + stddev[0]), 179);
			setTrackbarPos("Hmin", "Source", hmin);
			setTrackbarPos("Hmax", "Source", hmax);

			smin = MAX((means[1] - stddev[1]), 0);
			smax = MIN((means[1] + stddev[1]), 255);
			setTrackbarPos("Smin", "Source", smin);
			setTrackbarPos("Smax", "Source", smax);

			vmin = MAX((means[2] - stddev[2]), 0);
			vmax = MIN((means[2] + stddev[2]), 255);
			setTrackbarPos("Vmin", "Source", vmin);
			setTrackbarPos("Vmax", "Source", vmax);
		}

		//마우스로 영역 선택한 영역 표현하기
		if (selectObject && selection.area() > 0) {
			Mat roi_RGB(image, selection);
			bitwise_not(roi_RGB, roi_RGB);
			namedWindow("Source", 0);
			imshow("Source", image);
		}


		//contour 적용하기
		findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		if (!contours.empty()) {
			double maxArea = 0;
			int maxArea_idx = 0;
			for (int i = 0; i < contours.size(); i++) {
				double area = contourArea(contours[i]);//최대인 영역찾기
				if (area > maxArea) {//기존거보다 영역이 크면 해당 contour로 바꾸기
					maxArea = area;
					maxArea_idx = i;
				}
			}
			drawContours(dst_out, contours, maxArea_idx, Scalar(255, 255, 255), -1);// 내부까지 표현하기
			//imshow("Contour", dst_out);
		}

		mask_src = background & dst_out;//해당 물체와 겹친 부분
		//imshow("mask 1", mask_src);

		//mask한 부분 inverse취하기->반대 부분을 mask하기 위해
		for (int i = 0; i < dst_out.rows; i++) {
			for (int k = 0; k < dst_out.cols; k++) {
				dst_out.at<Vec3b>(i, k)[0] = 255 - dst_out.at<Vec3b>(i, k)[0];
				dst_out.at<Vec3b>(i, k)[1] = 255 - dst_out.at<Vec3b>(i, k)[1];
				dst_out.at<Vec3b>(i, k)[2] = 255 - dst_out.at<Vec3b>(i, k)[2];
			}
		}

		mask_src2 = src & dst_out;//해당 물체를 제와한 부분
		//imshow("mask 2", mask_src2);

		final_out = mask_src + mask_src2;//마스크를 적용한 두 부분을 or를 이용해서 겹치기
		namedWindow("final output", 0);
		imshow("final output", final_out);//최종 이미지
		save_video.write(final_out);//이미지 저장하기

		char c = (char)waitKey(10);
		if (c == 27) break;
	}

	//이미지 저장 종료하기
	save_video.release();
	cap.release();
	destroyAllWindows();
}


int main()
{
	//processVideo("LAB_MagicCloak_Sample1.mp4");
	processVideo("DLIP_LAB2_22000573_JunJaeLee_1.mp4");
	return 0;
}



