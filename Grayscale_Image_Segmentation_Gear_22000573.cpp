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

void inverse(Mat& output) {//RGB�̸鼭 ���� inverse ��Ű��
	for (int i = 0; i < output.rows; i++) {
		for (int k = 0; k < output.cols; k++) {
			// Invert each channel (Blue, Green, Red)
			output.at<Vec3b>(i, k)[0] = 255 - output.at<Vec3b>(i, k)[0]; // Blue 
			output.at<Vec3b>(i, k)[1] = 255 - output.at<Vec3b>(i, k)[1]; // Green
			output.at<Vec3b>(i, k)[2] = 255 - output.at<Vec3b>(i, k)[2]; // Red 

		}
	}
}

void RGB_Thresh(Mat& src, Mat& output, int threshold) {//RGB�̸鼭 Ư������ �������� ����� �������� ������
	//��� ���� rgb�� �����ϱ�
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

	threshold(src, binary, 128, 255, THRESH_BINARY);//binary ��ȯ, ���Ƿ� threshold �� 128����
	imshow("binary[" + to_string(idx) + "]", binary);//��� 1 binary ��� ����

	vector<vector<Point>> contours;//��� �̻��� Ȯ���ϱ� ���� contour

	//Point anchor = Point(binary.cols / 2 - 16, binary.rows / 2 - 68);//���� �߽���

	circle(binary, anchor, r1, 0, -2, LINE_8, 0);//������ �� �׸���
	imshow("circle_fill[" + to_string(idx) + "]", binary);//binaryó���� ������ ������ ���� �߰��� �̻��� ���̵��� �� ����

	/// Find contours
	findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	/// Draw all contours excluding holes
	Mat drawing(binary.size(), CV_8U, Scalar(255));//contour ����� ���� ����, ��� ���
	drawContours(drawing, contours, -1, Scalar(0), 2);//�̻��� �׵θ��� ���������� ǥ��
	imshow("countour[" + to_string(idx) + "]", drawing);

	int count = 0;//�̻� ����
	int fail_count = 0;//������ �ִ� ����� ��
	float area_sum = 0;//��ü ����� ����
	Scalar contourColor;//�̻� �׵θ� ��
	Scalar textColor;//�۾� ��

	Mat drawing_color = Mat::zeros(binary.size(), CV_8UC3); // ������ �����ϱ� ���� CV_8UC3 ����
	Mat output = Mat::zeros(src.size(), CV_8UC3); // ������ �����ϱ� ���� CV_8UC3 ����

	RGB_Thresh(src, output, 200);//RGB�̸鼭 ����� �������� ������

	circle(output, anchor, r2, Scalar(120, 120, 120), -2, LINE_8, 0);//�߰� ũ�� ȸ�� �� �׸���
	circle(output, anchor, r3, Scalar(0, 0, 0), -2, LINE_8, 0);//�������� ������ �� �׸���
	imshow("circle_fill_output[" + to_string(idx) + "]", output);//rgb�� �ٲ� ����

	inverse(output);//RGB inverse��Ű��

	threshold(output, output, 200, 255, 2);//Threshold Truncated
	imshow("output[" + to_string(idx) + "]", output);

	Point center;
	Point text_point;
	int radius = 32; // Circle radius
	int num_points = 12;// Point ��
	float angle = 0;//���� �׸� ����
	int x = 0;//���� x��ǥ
	int y = 0;//���� y��ǥ

	for (int i = 0; i < contours.size(); i++)
	{
		count++;//�̻� ���� �߰�
		printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//�ش� ��� �̻��� ������ ���̸� ���

		if (contourArea(contours[i]) < 1000) {//��� ������ 1000���� ���� ���
			fail_count++;//������ �ִ� �̻� ���� �߰�
			contourColor = Scalar(0, 0, 255); // ������ (BGR ����)

			//���� ���� �߽� ��ǥ ���ϱ�
			if (contourArea(contours[i]) < 400) {
				center = (contours[i][contours[i].size() - 1] + contours[i][0]) / 2;
			}
			else {
				center = (contours[i][contours[i].size() - 11]);
			}
			for (int num = 0; num < num_points; num++) {
				//�� �׵θ� �� ���
				angle = 2 * CV_PI * num / num_points;
				x = center.x + radius * cos(angle);
				y = center.y + radius * sin(angle);

				//����� �� �׸���
				circle(output, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
			}

		}
		else {
			contourColor = Scalar(0, 255, 0); // �ʷϻ� (BGR ����)
		}

		area_sum += contourArea(contours[i]); // ��ü ���� ���ϱ�

		// ������ �׸���
		drawContours(drawing_color, contours, i, contourColor, 2); // i ��° �������� ���� ���� �����Ͽ� �׸���

	}

	Mat add_text = drawing_color.clone();

	for (int i = 0; i < contours.size(); i++)
	{

		if (contourArea(contours[i]) < 1000) {//��� ������ 1000���� ���� ���
			textColor = Scalar(0, 0, 255);

		}
		else {
			textColor = Scalar(255, 255, 255);
		}

		text_point = contours[i][0];
		std::string area_text = std::to_string((int)contourArea(contours[i]));
		putText(add_text, area_text, text_point, FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1, LINE_8); // ������ �̹����� ǥ��

	}

	imshow("add_text[" + to_string(idx) + "]", add_text);
	// �̻��� ��ü ���� ���ڸ� drawing �̹����� �߰�

	imshow("add_color circle[" + to_string(idx) + "]", output);//������ �ִ� �κ� ���� ������ ǥ���� ����
	area_sum = area_sum / count;//����� ��� ���� ���ϱ�
	imshow("color contour[" + to_string(idx) + "]", drawing_color);//������ ���� ��� �ʷϻ�, ������ �ִ� ��� ������ ó���� ����

	printf("Teeth numbers: %d\n", count);//�̻� ���� ���
	printf("Avg. Teeth Area: %.2f\n", area_sum);//�̻��� ��ü ���� ���
	printf("Defective Teeth: %d\n", fail_count);//������ �ִ� �̻� ���� ���
	if (fail_count > 0) {
		printf("Quality: FAIL\n\n");//������ �ִ� ���
	}
	else if (fail_count == 0) {
		printf("Quality: PASS\n\n");//������ ���� ���
	}
}




void main()
{
	//��� 1
	cv::Mat src, binary;//��� ����
	src = cv::imread("Gear1.jpg", 0);
	Point anchor = Point(src.cols / 2 - 16, src.rows / 2 - 68);//���� �߽���
	int r1 = 169;
	int r2 = 140;
	int r3 = 70;
	
	FilterandContour(src, anchor, r1, r2, r3, 1);

	//threshold(src, binary, 128, 255, THRESH_BINARY);//binary ��ȯ, ���Ƿ� threshold �� 128����
	//imshow("binary", binary);//��� 1 binary ��� ����
	//
	//vector<vector<Point>> contours;//��� �̻��� Ȯ���ϱ� ���� contour
	//
	//Point anchor = Point(binary.cols / 2 - 16, binary.rows / 2 - 68);//���� �߽���

	//circle(binary, anchor, 169, 0, -2, LINE_8, 0);//������ �� �׸���
	//imshow("circle_fill", binary);//binaryó���� ������ ������ ���� �߰��� �̻��� ���̵��� �� ����

	///// Find contours
	//findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	///// Draw all contours excluding holes
	//Mat drawing(binary.size(), CV_8U, Scalar(255));//contour ����� ���� ����, ��� ���
	//drawContours(drawing, contours, -1, Scalar(0), 2);//�̻��� �׵θ��� ���������� ǥ��
	//imshow("countour", drawing);


	//int count = 0;//�̻� ����
	//int fail_count = 0;//������ �ִ� ����� ��
	//float area_sum = 0;//��ü ����� ����
	//Scalar contourColor;//�̻� �׵θ� ��
	//Scalar textColor;//�۾� ��

	//Mat drawing_color = Mat::zeros(binary.size(), CV_8UC3); // ������ �����ϱ� ���� CV_8UC3 ����

	//Mat output = Mat::zeros(src.size(), CV_8UC3); // ������ �����ϱ� ���� CV_8UC3 ����

	//RGG_Thresh(src,output,200);
	//////��� ���� rgb�� �����ϱ�
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

	//circle(output, anchor, 140, Scalar(120,120,120), -2, LINE_8, 0);//�߰� ũ�� ȸ�� �� �׸���
	//circle(output, anchor, 70, Scalar(0,0,0), -2, LINE_8, 0);//�������� ������ �� �׸���
	//imshow("circle_fill_output", output);//rgb�� �ٲ� ����

	//inverse(output);
	//����� ������� �����ϱ�
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
	//int num_points = 12;// Point ��
	//float angle = 0;//���� �׸� ����
	//int x = 0;//���� x��ǥ
	//int y = 0;//���� y��ǥ

	//for (int i = 0; i < contours.size(); i++)
	//{
	//	count++;//�̻� ���� �߰�
	//	printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//�ش� ��� �̻��� ������ ���̸� ���
	//	
	//	if (contourArea(contours[i]) < 1000) {//��� ������ 1000���� ���� ���
	//		fail_count++;//������ �ִ� �̻� ���� �߰�
	//		contourColor = Scalar(0, 0, 255); // ������ (BGR ����)

	//		//���� ���� �߽� ��ǥ ���ϱ�
	//		if (contourArea(contours[i]) < 400) {
	//			center = (contours[i][contours[i].size() - 1]+ contours[i][0])/2;
	//		}
	//		else {
	//			center = (contours[i][contours[i].size() - 11]);
	//		}
	//		for (int num = 0; num < num_points; num++) {
	//			//�� �׵θ� �� ���
	//			angle = 2 * CV_PI * num / num_points;
	//			x = center.x + radius * cos(angle);
	//			y = center.y + radius * sin(angle);

	//			//����� �� �׸���
	//			circle(output, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
	//		}
	//		
	//	}
	//	else {
	//		contourColor = Scalar(0, 255, 0); // �ʷϻ� (BGR ����)
	//	}
	//	
	//	area_sum+= contourArea(contours[i]); // ��ü ���� ���ϱ�

	//	// ������ �׸���
	//	drawContours(drawing_color, contours, i, contourColor, 2); // i ��° �������� ���� ���� �����Ͽ� �׸���

	//}

	//Mat add_text = drawing_color.clone();

	//for (int i = 0; i < contours.size(); i++)
	//{

	//	if (contourArea(contours[i]) < 1000) {//��� ������ 1000���� ���� ���
	//		textColor = Scalar(0, 0, 255);

	//	}
	//	else {
	//		textColor = Scalar(255, 255, 255);
	//	}

	//	text_point = contours[i][0];
	//	std::string area_text = std::to_string((int)contourArea(contours[i]));
	//	putText(add_text, area_text, text_point, FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1, LINE_8); // ������ �̹����� ǥ��

	//}

	//imshow("add_text",add_text);
	//// �̻��� ��ü ���� ���ڸ� drawing �̹����� �߰�
	//
	//imshow("add_color circle",output);//������ �ִ� �κ� ���� ������ ǥ���� ����
	//area_sum = area_sum / count;//����� ��� ���� ���ϱ�
	//imshow("color contour", drawing_color);//������ ���� ��� �ʷϻ�, ������ �ִ� ��� ������ ó���� ����
	//
	//printf("Teeth numbers: %d\n", count);//�̻� ���� ���
	//printf("Avg. Teeth Area: %.2f\n", area_sum);//�̻��� ��ü ���� ���
	//printf("Defective Teeth: %d\n",fail_count);//������ �ִ� �̻� ���� ���
	//if (fail_count>0) {
	//	printf("Quality: FAIL\n\n");//������ �ִ� ���
	//}
	//else if (fail_count == 0) {
	//	printf("Quality: PASS\n\n");//������ ���� ���
	//}
	

	//��� 2
	src = cv::imread("Gear2.jpg", 0);
	anchor = Point(src.cols / 2 - 16, src.rows / 2 - 5);//���� �߽���
	r1 = 169;
	r2 = 140;
	r3 = 70;
	FilterandContour(src, anchor, r1, r2, r3, 2);
	//// ��� 2
	//src = cv::imread("Gear2.jpg", 0);

	//threshold(src, binary, 128, 255, THRESH_BINARY);//binary ��ȯ, ���Ƿ� threshold �� 128����
	//imshow("binary_2", binary);//��� 2 binary ��� ����

	//anchor = Point(binary.cols / 2 - 16, binary.rows / 2 -5);//���� �߽���

	//circle(binary, anchor, 169, 0, -2, LINE_8, 0);//������ �� �׸���
	//imshow("circle_fill_2", binary);//binaryó���� ������ ������ ���� �߰��� �̻��� ���̵��� �� ����

	///// Find contours
	//findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	///// Draw all contours excluding holes
	//Mat drawing_1(binary.size(), CV_8U, Scalar(255));//contour ����� ���� ����, ��� ���
	//drawContours(drawing_1, contours, -1, Scalar(0), 2);//�̻��� �׵θ��� ���������� ǥ��
	//imshow("countour_2", drawing_1);

	//count = 0;//�̻� ����
	//fail_count = 0;//������ �ִ� ����� ��
	//area_sum = 0;//��ü ����� ����
	//
	//Mat drawing_color_1 = Mat::zeros(binary.size(), CV_8UC3); // ������ �����ϱ� ���� CV_8UC3 ����

	//Mat output_1 = Mat::zeros(src.size(), CV_8UC3); // ������ �����ϱ� ���� CV_8UC3 ����

	////��� ���� rgb�� �����ϱ�
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

	//circle(output_1, anchor, 140, Scalar(120, 120, 120), -2, LINE_8, 0);//�߰� ũ�� ȸ�� �� �׸���
	//circle(output_1, anchor, 70, Scalar(0, 0, 0), -2, LINE_8, 0);//�������� ������ �� �׸���
	//imshow("circle_fill_output", output_1);//rgb�� �ٲ� ����

	////����� ������� �����ϱ�
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
	//	count++;//�̻� ���� �߰�
	//	printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//�ش� ��� �̻��� ������ ���̸� ���

	//	if (contourArea(contours[i]) < 1000) {//��� ������ 1000���� ���� ���
	//		fail_count++;//������ �ִ� �̻� ���� �߰�
	//		contourColor = Scalar(0, 0, 255); // ������ (BGR ����)
	//		//���� ���� �߽� ��ǥ ���ϱ�
	//		if (contourArea(contours[i]) < 400) {
	//			center = (contours[i][contours[i].size() - 1] + contours[i][0]) / 2;
	//		}
	//		else {
	//			center = (contours[i][contours[i].size() - 11]);
	//		}
	//		for (int num = 0; num < num_points; num++) {
	//			//�� �׵θ� �� ���
	//			angle = 2 * CV_PI * num / num_points;
	//			x = center.x + radius * cos(angle);
	//			y = center.y + radius * sin(angle);

	//			//����� �� �׸���
	//			circle(output_1, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
	//		}
	//	}
	//	else {
	//		contourColor = Scalar(0, 255, 0); // �ʷϻ� (BGR ����)
	//	}

	//	area_sum += contourArea(contours[i]); // ��ü ���� ���ϱ�

	//	// ������ �׸���
	//	drawContours(drawing_color_1, contours, i, contourColor, 2); // i ��° �������� ���� ���� �����Ͽ� �׸���

	//}
	//Mat add_text_1 = drawing_color_1.clone();

	//for (int i = 0; i < contours.size(); i++)
	//{

	//	if (contourArea(contours[i]) < 1000) {//��� ������ 1000���� ���� ���
	//		textColor = Scalar(0, 0, 255);

	//	}
	//	else {
	//		textColor = Scalar(255, 255, 255);
	//	}

	//	text_point = contours[i][0];
	//	std::string area_text = std::to_string((int)contourArea(contours[i]));
	//	putText(add_text_1, area_text, text_point, FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1, LINE_8); // ������ �̹����� ǥ��

	//}

	//imshow("add_text_2", add_text_1);

	//imshow("add_color circle_2", output_1);//������ �ִ� �κ� ���� ������ ǥ���� ����
	//area_sum = area_sum / count;//����� ��� ���� ���ϱ�
	//imshow("color contour_2", drawing_color_1);//������ ���� ��� �ʷϻ�, ������ �ִ� ��� ������ ó���� ����

	//printf("Teeth numbers: %d\n", count);//�̻� ���� ���
	//printf("Avg. Teeth Area: %.2f\n", area_sum);//�̻��� ��ü ���� ���
	//printf("Defective Teeth: %d\n", fail_count);//������ �ִ� �̻� ���� ���
	//if (fail_count > 0) {
	//	printf("Quality: FAIL\n\n");//������ �ִ� ���
	//}
	//else if (fail_count == 0) {
	//	printf("Quality: PASS\n\n");//������ ���� ���
	//}

	//// ��� 3
	//src = cv::imread("Gear3.jpg", 0);

	//threshold(src, binary, 128, 255, THRESH_BINARY);//binary ��ȯ, ���Ƿ� threshold �� 128����
	//imshow("binary_3", binary);//��� 3 binary ��� ����

	//anchor = Point(binary.cols / 2 +22, binary.rows / 2 - 15);//���� �߽���

	//circle(binary, anchor, 187, 0, -2, LINE_8, 0);//������ �� �׸���
	//imshow("circle_fill_3", binary);//binaryó���� ������ ������ ���� �߰��� �̻��� ���̵��� �� ����

	///// Find contours
	//findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	///// Draw all contours excluding holes
	//Mat drawing_2(binary.size(), CV_8U, Scalar(255));//contour ����� ���� ����, ��� ���
	//drawContours(drawing_2, contours, -1, Scalar(0), 2);//�̻��� �׵θ��� ���������� ǥ��
	//imshow("countour_3", drawing_2);

	//count = 0;//�̻� ����
	//fail_count = 0;//������ �ִ� ����� ��
	//area_sum = 0;//��ü ����� ����

	//Mat drawing_color_2 = Mat::zeros(binary.size(), CV_8UC3); // ������ �����ϱ� ���� CV_8UC3 ����

	//Mat output_2 = Mat::zeros(src.size(), CV_8UC3); // ������ �����ϱ� ���� CV_8UC3 ����

	////��� ���� rgb�� �����ϱ�
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

	//circle(output_2, anchor, 140, Scalar(120, 120, 120), -2, LINE_8, 0);//�߰� ũ�� ȸ�� �� �׸���
	//circle(output_2, anchor, 70, Scalar(0, 0, 0), -2, LINE_8, 0);//�������� ������ �� �׸���
	//imshow("circle_fill_output_3", output_2);//rgb�� �ٲ� ����

	////����� ������� �����ϱ�
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
	//	count++;//�̻� ���� �߰�
	//	printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//�ش� ��� �̻��� ������ ���̸� ���


	//	if (contourArea(contours[i]) < 1000 || contourArea(contours[i]) > 1500) {
	//		fail_count++;//������ �ִ� �̻� ���� �߰�
	//		contourColor = Scalar(0, 0, 255); // ������ (BGR ����)

	//		//���� ���� �߽� ��ǥ ���ϱ�
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
	//			//�� �׵θ� �� ���
	//			angle = 2 * CV_PI * num / num_points;
	//			x = center.x + radius * cos(angle);
	//			y = center.y + radius * sin(angle);

	//			//����� �� �׸���
	//			circle(output_2, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
	//		}
	//	}
	//	else {
	//		contourColor = Scalar(0, 255, 0); // �ʷϻ� (BGR ����)
	//	}

	//	area_sum += contourArea(contours[i]); // ��ü ���� ���ϱ�

	//	// ������ �׸���
	//	drawContours(drawing_color_2, contours, i, contourColor, 2); // i ��° �������� ���� ���� �����Ͽ� �׸���

	//}
	//Mat add_text_2 = drawing_color_2.clone();

	//for (int i = 0; i < contours.size(); i++)
	//{

	//	if (contourArea(contours[i]) < 1000) {//��� ������ 1000���� ���� ���
	//		textColor = Scalar(0, 0, 255);

	//	}
	//	else {
	//		textColor = Scalar(255, 255, 255);
	//	}

	//	text_point = contours[i][0];
	//	std::string area_text = std::to_string((int)contourArea(contours[i]));
	//	putText(add_text_2, area_text, text_point, FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1, LINE_8); // ������ �̹����� ǥ��

	//}

	//imshow("add_text_3", add_text_2);

	//imshow("add_color circle_3", output_2);//������ �ִ� �κ� ���� ������ ǥ���� ����
	//area_sum = area_sum / count;//����� ��� ���� ���ϱ�
	//imshow("color contour_3", drawing_color_2);//������ ���� ��� �ʷϻ�, ������ �ִ� ��� ������ ó���� ����

	//printf("Teeth numbers: %d\n", count);//�̻� ���� ���
	//printf("Avg. Teeth Area: %.2f\n", area_sum);//�̻��� ��ü ���� ���
	//printf("Defective Teeth: %d\n", fail_count);//������ �ִ� �̻� ���� ���
	//if (fail_count > 0) {
	//	printf("Quality: FAIL\n\n");//������ �ִ� ���
	//}
	//else if (fail_count == 0) {
	//	printf("Quality: PASS\n\n");//������ ���� ���
	//}

	//// ��� 4
	//src = cv::imread("Gear4.jpg", 0);

	//threshold(src, binary, 128, 255, THRESH_BINARY);//binary ��ȯ, ���Ƿ� threshold �� 128����
	//imshow("binary_4", binary);//��� 4 binary ��� ����

	//anchor = Point(binary.cols / 2 -74, binary.rows / 2 - 32);//���� �߽���

	//circle(binary, anchor, 188, 0, -2, LINE_8, 0);//������ �� �׸���
	//imshow("circle_fill_4", binary);//binaryó���� ������ ������ ���� �߰��� �̻��� ���̵��� �� ����

	///// Find contours
	//findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	///// Draw all contours excluding holes
	//Mat drawing_3(binary.size(), CV_8U, Scalar(255));//contour ����� ���� ����, ��� ���
	//drawContours(drawing_3, contours, -1, Scalar(0), 2);//�̻��� �׵θ��� ���������� ǥ��
	//imshow("countour_4", drawing_3);

	//count = 0;//�̻� ����
	//fail_count = 0;//������ �ִ� ����� ��
	//area_sum = 0;//��ü ����� ����

	//Mat drawing_color_3 = Mat::zeros(binary.size(), CV_8UC3); // ������ �����ϱ� ���� CV_8UC3 ����

	//Mat output_3 = Mat::zeros(src.size(), CV_8UC3); // ������ �����ϱ� ���� CV_8UC3 ����

	////��� ���� rgb�� �����ϱ�
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

	//circle(output_3, anchor, 140, Scalar(120, 120, 120), -2, LINE_8, 0);//�߰� ũ�� ȸ�� �� �׸���
	//circle(output_3, anchor, 70, Scalar(0, 0, 0), -2, LINE_8, 0);//�������� ������ �� �׸���
	//imshow("circle_fill_output", output_3);//rgb�� �ٲ� ����

	////����� ������� �����ϱ�
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
	//	count++;//�̻� ���� �߰�
	//	printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));//�ش� ��� �̻��� ������ ���̸� ���


	//	if (contourArea(contours[i]) < 1000 || contourArea(contours[i]) > 1500) {//������ 1000���ϰų� 1500 �̻��� ��
	//		fail_count++;//������ �ִ� �̻� ���� �߰�
	//		contourColor = Scalar(0, 0, 255); // ������ (BGR ����)

	//		//���� �׵θ��� ���� ���� �߽���
	//		if (contourArea(contours[i]) <1800) {
	//			center = (contours[i][30] + contours[i][0]) / 2;
	//		}
	//		else {
	//			center= (contours[i][35] + contours[i][0]) / 2;
	//		}
	//		//���� �� �׸���
	//		for (int num = 0; num < num_points; num++) {
	//			//�� �׵θ� �� ���
	//			angle = 2 * CV_PI * num / num_points;
	//			x = center.x + radius * cos(angle);//���� x��ǥ
	//			y = center.y + radius * sin(angle);//���� y��ǥ

	//			//����� �� �׸���
	//			circle(output_3, Point(x, y), 3, Scalar(0, 255, 255), -1, LINE_8, 0);
	//		}
	//	}
	//	else {
	//		contourColor = Scalar(0, 255, 0); // �ʷϻ� (BGR ����)
	//	}

	//	area_sum += contourArea(contours[i]); // ��ü ���� ���ϱ�

	//	// ������ �׸���
	//	drawContours(drawing_color_3, contours, i, contourColor, 2); // i ��° �������� ���� ���� �����Ͽ� �׸���

	//}


	//Mat add_text_3 = drawing_color_3.clone();

	//for (int i = 0; i < contours.size(); i++)
	//{

	//	if (contourArea(contours[i]) < 1000) {//��� ������ 1000���� ���� ���
	//		textColor = Scalar(0, 0, 255);

	//	}
	//	else {
	//		textColor = Scalar(255, 255, 255);
	//	}

	//	text_point = contours[i][0];
	//	std::string area_text = std::to_string((int)contourArea(contours[i]));
	//	putText(add_text_3, area_text, text_point, FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1, LINE_8); // ������ �̹����� ǥ��

	//}

	//imshow("add_text_4", add_text_3);

	//imshow("add_color circle_4", output_3);//������ �ִ� �κ� ���� ������ ǥ���� ����
	//area_sum = area_sum / count;//����� ��� ���� ���ϱ�
	//imshow("color contour_4", drawing_color_3);//������ ���� ��� �ʷϻ�, ������ �ִ� ��� ������ ó���� ����

	//printf("Teeth numbers: %d\n", count);//�̻� ���� ���
	//printf("Avg. Teeth Area: %.2f\n", area_sum);//�̻��� ��ü ���� ���
	//printf("Defective Teeth: %d\n", fail_count);//������ �ִ� �̻� ���� ���
	//if (fail_count > 0) {
	//	printf("Quality: FAIL\n\n");//������ �ִ� ���
	//}
	//else if (fail_count == 0) {
	//	printf("Quality: PASS\n\n");//������ ���� ���
	//}


	cv::waitKey(0);
}