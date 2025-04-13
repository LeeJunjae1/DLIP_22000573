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

Mat image;//������ �̹���
Point origin;//���콺 ��ǥ
Rect selection;//���콺 ���� ����
bool selectObject = false;
bool trackObject = false;
//int hmin = 13, hmax = 60, smin = 70, smax = 200, vmin = 90, vmax = 200;//SAMPLE 1��
//int hmin = 24, hmax = 101, smin = 40, smax = 255, vmin = 55, vmax = 255;
int hmin = 110, hmax = 176, smin = 87, smax = 255, vmin =70, vmax = 255;//LAB2 ��ȫ�� ��ü ��
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
	vector<vector<Point>> contours;//contour�� ����
	Mat dst_track, dst_out, final_out;

	VideoCapture cap(videoPath);
	if (!cap.isOpened()) {
		cout << "������ �� �� �����ϴ�: " << videoPath << endl;
		return;
	}

	//������ �̹��� �����Ӱ� �ʺ�
	int frame_width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
	double fps = cap.get(CAP_PROP_FPS);//�ش� ������

	int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');//avi �������� ����
	string filename = "DLIP_LAB2_22000573_JunJaeLee_output.avi";  // �������� �̸�

	save_video.open(filename, codec, fps, Size(frame_width, frame_height), true);
	if (!save_video.isOpened()) {
		cout << "!!! VideoWriter �ʱ�ȭ ����: ������ �� �� �����ϴ�." << endl;
		return;
	}

	// Ʈ���ٿ� ���콺 �ݹ� ����
	namedWindow("Source", 0);
	setMouseCallback("Source", onMouse, 0);
	createTrackbar("Hmin", "Source", &hmin, 179);
	createTrackbar("Hmax", "Source", &hmax, 179);
	createTrackbar("Smin", "Source", &smin, 255);
	createTrackbar("Smax", "Source", &smax, 255);
	createTrackbar("Vmin", "Source", &vmin, 255);
	createTrackbar("Vmax", "Source", &vmax, 255);

	int element_shape = MORPH_RECT;
	int n = 7;//���� ������ �� morphology ������
	Mat element = getStructuringElement(element_shape, Size(n, n));

	int cap_image = 0;//ó�� �̹����� �����ϱ� ���� ����

	while (true)
	{
		if (!cap.read(src)) {
			cout << "�������� ������ �� �����ϴ�.\n";
			break;
		}

		//imshow("source video", src);

		if (cap_image == 0) {//���� �������϶�
			src.copyTo(background);//���� ����̹��� ����
			cap_image++;//count ���� ��Ű��
		}
		//imshow("background", background);
		src.copyTo(image);//���� �����ϱ�
		cvtColor(src, hsv, COLOR_BGR2HSV);//HSV�� ��ȯ�ϱ�
		blur(hsv, hsv, Size(5, 5));//���� ����

		inRange(hsv,
			Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
			Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)),
			dst);//�ش� ���� �ִ� ���� ã��

		//imshow("InRange1", dst);
		morphologyEx(dst, dst, MORPH_OPEN, element); //opening �ǽ�
		//dilate(dst, dst, element);//DIALATE�� ���� �߰��� �� �κ� ä���
		dst_out = Mat::zeros(dst.size(), CV_8UC3);//MASK�� �� �κ�
		final_out = Mat::zeros(dst.size(), CV_8UC3);//���� �����

		imshow("InRange", dst);

		//Ʈ���ٷ� inrange �� �����ϱ�
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

		//���콺�� ���� ������ ���� ǥ���ϱ�
		if (selectObject && selection.area() > 0) {
			Mat roi_RGB(image, selection);
			bitwise_not(roi_RGB, roi_RGB);
			namedWindow("Source", 0);
			imshow("Source", image);
		}


		//contour �����ϱ�
		findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		if (!contours.empty()) {
			double maxArea = 0;
			int maxArea_idx = 0;
			for (int i = 0; i < contours.size(); i++) {
				double area = contourArea(contours[i]);//�ִ��� ����ã��
				if (area > maxArea) {//�����ź��� ������ ũ�� �ش� contour�� �ٲٱ�
					maxArea = area;
					maxArea_idx = i;
				}
			}
			drawContours(dst_out, contours, maxArea_idx, Scalar(255, 255, 255), -1);// ���α��� ǥ���ϱ�
			//imshow("Contour", dst_out);
		}

		mask_src = background & dst_out;//�ش� ��ü�� ��ģ �κ�
		//imshow("mask 1", mask_src);

		//mask�� �κ� inverse���ϱ�->�ݴ� �κ��� mask�ϱ� ����
		for (int i = 0; i < dst_out.rows; i++) {
			for (int k = 0; k < dst_out.cols; k++) {
				dst_out.at<Vec3b>(i, k)[0] = 255 - dst_out.at<Vec3b>(i, k)[0];
				dst_out.at<Vec3b>(i, k)[1] = 255 - dst_out.at<Vec3b>(i, k)[1];
				dst_out.at<Vec3b>(i, k)[2] = 255 - dst_out.at<Vec3b>(i, k)[2];
			}
		}

		mask_src2 = src & dst_out;//�ش� ��ü�� ������ �κ�
		//imshow("mask 2", mask_src2);

		final_out = mask_src + mask_src2;//����ũ�� ������ �� �κ��� or�� �̿��ؼ� ��ġ��
		namedWindow("final output", 0);
		imshow("final output", final_out);//���� �̹���
		save_video.write(final_out);//�̹��� �����ϱ�

		char c = (char)waitKey(10);
		if (c == 27) break;
	}

	//�̹��� ���� �����ϱ�
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



