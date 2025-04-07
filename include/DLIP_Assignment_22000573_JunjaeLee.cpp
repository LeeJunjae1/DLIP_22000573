/* ------------------------------------------------------ /
*Image Proccessing with Deep Learning
* Line Dectecting
* Created : 2025-04-01
* Name: Junjae Lee
------------------------------------------------------ */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

void LineDecting(const string & src_img, int idx) {
	Mat dst, cdst, cdstP, canny, src;

	// Loads an image
	Mat SRC = imread(src_img);//�̹��� �ޱ�
	src = SRC.clone();//���� �̹��� �����ϱ�
	
	cvtColor(src, src, COLOR_BGR2GRAY);//grayscale�� �����ϱ�
	imshow("source[" + to_string(idx) + "]", SRC);//���� �̹��� ����ϱ�
	imshow("gray scale[" + to_string(idx) + "]", src);//gray scale ��� ����ϱ�
	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		return ;
	}

	// Edge detection
	Canny(src, dst, 200, 600, 3);

	imshow("canny[" + to_string(idx) + "]", dst);//canny ��� ���

	//���ɿ����� ����� ���� ���ɾ��� �κ��� ������ �簢������ ���
	rectangle(dst, Point(0, 0), Point(dst.cols, 440), Scalar(0), FILLED);//������ ���� �κ�
	rectangle(dst, Point(dst.cols / 2 + 320, 0), Point(dst.cols, dst.rows), Scalar(0), FILLED);//������ ������ �κ�
	imshow("delete area[" + to_string(idx) + "]", dst);//�����ִ� �κи� ���� ��� ���

	// Copy edge results to the images that will display the results in BGR
	cvtColor(dst, cdstP, COLOR_GRAY2BGR);//BRG�� ��ȯ�ϱ�, ���� ǥ���ϱ� ����
	//cdstP = cdst.clone();

	vector<Vec4i> linesP;//line ������ ���� ����
	HoughLinesP(dst, linesP, 1, CV_PI / 180, 10, 5, 10);//line ���� �ǽ�, �̶� line�� ������ �˾Ƴ�

	double theta = 0.0;//������ ������ ���ϱ� ����
	double dx = 0.0;//x���� ���̸� ���� ����
	double dy = 0.0;//y���� ������ ���� ����
//	double len = 0.0;

	double point_x_min = 0.0;//������ ������ �κ� x ��ǥ �����ϱ� ���� ����
	double point_x_min1 = 0.0;//���� ���ʺκ��� x ��ǥ�� �����ϱ� ���� ����
	double point_y_min = 0.0;//���� ������ �κ� y��ǥ�� �����ϱ� ���� ����
	double point_y_min1 = 0.0;//���� ���� �κ��� y��ǥ�� �����ϱ� ���� ����
	int max_idx = 0;//���� ���� ���� �� ���� ���� line index
	int min_idx = 0;//���� ���� ���� �� ������ ���� line index

	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		dx = l[2] - l[0], dy = l[3] - l[1];//dx, dy���
		//len = sqrt(dx * dx + dy * dy);

		if (dx == 0) {
			theta = CV_PI / 2;  // �������� ��� 90�� (��/2 ����)
		}
		else {
			theta = atan2(dx, dy) * 180 / CV_PI;  // atan2�� ����ϸ� dx=0�� �����ϰ� ó����
		}

		if (theta > 90) {
			//���� ���� ����, �̶� line�� ������ ���� ����� �κ� �� ���� line�� �����ϱ� ���� ����
			if (i == 0) {
				//������ �� ����
				point_x_min1 = l[0];
				point_y_min1 = l[1];
				max_idx = i;
			}
			else {
				//x��ǥ�� ū ��� ���ʶ����� ���ɼ��� ���� �̶� x��ǥ�� ū ���ÿ� y��ǥ�� ũ�� ���� ������
				if (l[0] > point_x_min1 && l[1] > point_y_min1) {
					point_x_min1 = l[0];//���� x��ǥ�� ����
					point_y_min1 = l[1];//���� y��ǥ�� ����
					max_idx = i;//���� �ε��� ����
				}
			}
		}
		if (theta < 90) {
			//������ ���� ����, �̶� line�� ������ ���� ����� �κ� �� ���� line�� �����ϱ� ���� ����
			if (i == 0) {
				//������ �� ����
				point_x_min = l[0];
				point_y_min = l[1];
				min_idx = i;

			}
			else {
				//x��ǥ�� ���� ��� ���ʶ����� ���ɼ��� ���� �̶� x��ǥ�� ���� ���ÿ� y��ǥ�� ũ�� ���� ������
				if (l[0] < point_x_min && l[1]>point_y_min) {
					point_x_min = l[0];//���� x��ǥ�� ����
					point_y_min = l[1];//���� y��ǥ�� ����
					min_idx = i;//���� �ε��� ����
				}
			}
		}
	}

	double extend_left_x1, extend_left_x2 = 0.0;//���� ���弱 x��ǥ
	double extend_right_x1, extend_right_x2 = 0.0;//������ ���弱 x��ǥ
	double inter_x, inter_y = 0.0;//���� x, y��ǥ
	double m1, m2, b1, b2 = 0;//m1, m2: ����, b1, b2: y����

	//���� ���� �κ�
	dx = linesP[max_idx][2] - linesP[max_idx][0];//dx���
	dy = linesP[max_idx][3] - linesP[max_idx][1];//dy���
	//extend_left_x1 = (0 - linesP[max_idx][1]) / (dy / dx) + linesP[max_idx][0];//y���� 
	m1 = dy / dx;//���� ���
	b1 = linesP[max_idx][1] - m1 * linesP[max_idx][0];//y���� ���

	extend_left_x2 = (dst.rows - linesP[max_idx][1]) / (dy / dx) + linesP[max_idx][0];//y��ǥ�� �ִ밪�� �� x�� ���ϱ�
	Point left_2 = Point(extend_left_x2, dst.rows);//���� ������ ���� �� �κ��� ��ǥ


	//������ ���� �κ�
	dx = linesP[min_idx][2] - linesP[min_idx][0];//dx ���
	dy = linesP[min_idx][3] - linesP[min_idx][1];//dy ���
	//extend_right_x1 = (0 - linesP[min_idx][1]) / (dy / dx) + linesP[min_idx][0];
	m2 = dy / dx;//���� ���
	b2 = linesP[min_idx][1] - m2 * linesP[min_idx][0];//y���� ���

	extend_right_x2 = (dst.rows - linesP[min_idx][1]) / (dy / dx) + linesP[min_idx][0];//y��ǥ�� �ִ밪�� �� x�� ���ϱ�
	Point right_2 = Point(extend_right_x2, dst.rows);//������ ������ ���� �� �κ��� ��ǥ

	inter_x = (b2 - b1) / (m1 - m2);//������ x ��ǥ
	inter_y = m1 * inter_x + b1;//������ y ��ǥ

	line(SRC, Point(inter_x, inter_y), left_2, Scalar(0, 0, 255), 3, LINE_AA);//���� ����, ���������� ǥ��
	line(SRC, Point(inter_x, inter_y), right_2, Scalar(0, 255, 0), 3, LINE_AA);//������ ����, �ʷϻ����� ǥ��

	circle(SRC, Point(inter_x, inter_y), 10, Scalar(0, 200, 255),2);//���� �κ�, ������ �߽ɺκ�
	line(SRC, Point(inter_x, inter_y), Point(inter_x, dst.rows), Scalar(255, 0, 0), 1, LINE_AA);//������ ������ ������ ���� �̰��� �������� ������ �ȹٷ� ���� �ִ��� �Ǵ� ������
	
	//�ﰢ�� �׸���
	vector<vector<Point>> fillVec = { {Point(inter_x, inter_y),right_2,left_2 } };
	double alpha = 0.8;
	
	// ���� �̹����� ���� ũ���� mask �̹��� ����� (���� ���, 3ä��)
	Mat mask = Mat::zeros(SRC.size(), SRC.type());
	fillPoly(mask, fillVec, Scalar(100, 100, 255));

	//����ġ�� �� �ﰢ���� �����ϰ� �����
	addWeighted(SRC, alpha, mask, 1.0 - alpha, 0, SRC);

	// Show results
	imshow("Result[" + to_string(idx) + "]", SRC);
}



int main(int argc, char** argv)
{
	// Loads an image
	const char* filename = "Lane_center.JPG";
	LineDecting(filename, 1);
	filename = "Lane_changing.JPG";
	LineDecting(filename, 2);

	// Wait and Exit
	waitKey();
	return 0;
}