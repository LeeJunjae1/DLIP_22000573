#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	// Declare the output variables
	Mat dst, cdst, cdstP, canny, src;

	// Loads an image
//	const char* filename = "Lane_center.JPG";
	const char* filename = "Lane_changing.JPG";
	Mat SRC = imread(filename);
	src = SRC.clone();
	cvtColor(src, src, COLOR_BGR2GRAY);
	imshow("source",SRC);
	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		return -1;
	}

	// Edge detection
	Canny(src, dst, 200, 600, 3);

	imshow("canny", dst);
	rectangle(dst,Point (0,0),Point (dst.cols,440),Scalar(0),FILLED);
	rectangle(dst, Point(dst.cols/2+320, 0), Point(dst.cols, dst.rows), Scalar(0), FILLED);
	imshow("delete area",dst);

	// Copy edge results to the images that will display the results in BGR
	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	cdstP = cdst.clone();

	//// (Option 1) Standard Hough Line Transform
	//vector<Vec2f> lines;
	//HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);

	//// Draw the detected lines
	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	float rho = lines[i][0], theta = lines[i][1];
	//	Point pt1, pt2;
	//	double a = cos(theta), b = sin(theta);
	//	double x0 = a * rho, y0 = b * rho;
	//	pt1.x = cvRound(x0 + 1000 * (-b));
	//	pt1.y = cvRound(y0 + 1000 * (a));
	//	pt2.x = cvRound(x0 - 1000 * (-b));
	//	pt2.y = cvRound(y0 - 1000 * (a));
	//	line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	//}

	// (Option 2) Probabilistic Line Transform
	vector<Vec4i> linesP;
	HoughLinesP(dst, linesP, 1, CV_PI / 180, 10, 5, 10);

	double theta=0.0;
	double dx = 0.0;
	double dy = 0.0;
	double len = 0.0;

	double point_x_min = 0.0;
	double point_x_min1 = 0.0;
	double point_y_min = 0.0;
	double point_y_min1 = 0.0;
	int max_idx = 0;
	int min_idx = 0;

	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		dx = l[2] - l[0];
		dy = l[3] - l[1];
		len = sqrt(dx * dx + dy * dy);

		if (dx == 0) {
			theta = CV_PI / 2;  // 수직선일 경우 90도 (π/2 라디안)
		}
		else {
			theta = atan2(dx, dy) * 180 / CV_PI;  // atan2를 사용하면 dx=0도 안전하게 처리됨
		}

		if (theta > 90) {
			//line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
			if (i==0) {
				point_x_min1 = l[0];
				point_y_min1 = l[1];
				max_idx = i;
			}
			else {
				if (l[0]>point_x_min1&&l[1]>point_y_min1) {
					point_x_min1 = l[0];
					point_y_min1 = l[1];
					max_idx = i;
				}
			}
		}
		if (theta < 90) {
			if (i==0) {
				point_x_min = l[0];
				point_y_min = l[1];
				min_idx = i;

			}
			else {
				if (l[0] < point_x_min&&l[1]>point_y_min) {
					//line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 1, LINE_AA);
					point_x_min = l[0];
					point_y_min = l[1];
					min_idx = i;
				}
			}
		}
		//printf("dx: %f, degree : %f\n", dx, theta);
	//		circle(dst, Point(l[0], l[1]), 3, Scalar(155),-1);
	//circle(dst, Point(l[2], l[3]), 3, Scalar(255), -1);
	}

	double extend_left_x1, extend_left_x2 = 0.0;
	double extend_right_x1,extend_right_x2 = 0.0;
	double inter_x, inter_y = 0.0;
	double m1, m2, b1, b2 = 0;
	
	dx = linesP[max_idx][2] - linesP[max_idx][0];
	dy = linesP[max_idx][3] - linesP[max_idx][1];
	extend_left_x1 = (0 - linesP[max_idx][1]) / (dy / dx) + linesP[max_idx][0];
	m1 = dy / dx;
	b1 = linesP[max_idx][1] - m1 * linesP[max_idx][0];
	//Point left_1 = Point(extend_left_x1, 0);

	extend_left_x2 = (dst.rows - linesP[max_idx][1]) / (dy / dx) + linesP[max_idx][0];
	Point left_2 = Point(extend_left_x2, dst.rows);
	


	dx = linesP[min_idx][2] - linesP[min_idx][0];
	dy = linesP[min_idx][3] - linesP[min_idx][1];
	extend_right_x1 = (0 - linesP[min_idx][1]) / (dy / dx) + linesP[min_idx][0];
	m2 = dy / dx;
	b2 = linesP[min_idx][1] - m2 * linesP[min_idx][0];
	//Point right_1 = Point(extend_right_x1, 0);

	extend_right_x2 = (dst.rows - linesP[min_idx][1]) / (dy / dx) + linesP[min_idx][0];
	Point right_2 = Point(extend_right_x2, dst.rows);

	inter_x = (b2-b1) / (m1-m2);
	inter_y = m1 * inter_x + b1;

	line(SRC, Point(inter_x, inter_y), left_2, Scalar(0, 0, 255), 3, LINE_AA);
	line(SRC, Point(inter_x, inter_y), right_2, Scalar(0, 255, 0), 3, LINE_AA);

	circle(SRC, Point(inter_x, inter_y), 10, Scalar(0, 200, 255));
	line(SRC, Point(inter_x, inter_y),Point(inter_x, dst.rows),Scalar(255,0,0),1,LINE_AA );
	


	//line(cdstP, Point(linesP[max_idx][0], linesP[max_idx][1]), Point(linesP[max_idx][2], linesP[max_idx][3]), Scalar(0, 0, 255), 3, LINE_AA);
	//line(cdstP, Point(linesP[min_idx][0], linesP[min_idx][1]), Point(linesP[min_idx][2], linesP[min_idx][3]), Scalar(0, 255, 0), 3, LINE_AA);


	//// Draw the lines
	//for (size_t i = 0; i < linesP.size(); i++)
	//{	
	//	
	//	
	//	Vec4i l = linesP[i];
	//	dx = l[2] - l[0];
	//	dy = l[3] - l[1];
	//	len = sqrt(dx*dx+dy*dy);
	//	
	//	if (dx == 0) {
	//		theta = CV_PI / 2;  // 수직선일 경우 90도 (π/2 라디안)
	//	}
	//	else {
	//		theta = atan2(dx, dy)*180/CV_PI;  // atan2를 사용하면 dx=0도 안전하게 처리됨
	//	}
	//	if (theta > 90) {
	//		line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
	//	}
	//	if (theta < 90) {
	//		point_x_min = l[0];
	//		line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 1, LINE_AA);
	//	}

	//	printf("dx: %f, degree : %f\n", dx,theta);
	//	circle(dst, Point(l[0], l[1]), 3, Scalar(155),-1);
	//	circle(dst, Point(l[2], l[3]), 3, Scalar(255), -1);
	//	printf("len: %f\n",len);
	//}

	// Show results
	imshow("point ", dst);
	//imshow("Source", src);
	//imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
	imshow("Result", SRC);

	// Wait and Exit
	waitKey();
	return 0;
}