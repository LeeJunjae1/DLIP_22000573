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
	Mat SRC = imread(src_img);//이미지 받기
	src = SRC.clone();//받은 이미지 복사하기
	
	cvtColor(src, src, COLOR_BGR2GRAY);//grayscale로 변경하기
	imshow("source[" + to_string(idx) + "]", SRC);//원본 이미지 출력하기
	imshow("gray scale[" + to_string(idx) + "]", src);//gray scale 결과 출력하기
	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		return ;
	}

	// Edge detection
	Canny(src, dst, 200, 600, 3);

	imshow("canny[" + to_string(idx) + "]", dst);//canny 결과 출력

	//관심영역만 남기기 위해 관심없는 부분을 검은색 사각형으로 덮어씀
	rectangle(dst, Point(0, 0), Point(dst.cols, 440), Scalar(0), FILLED);//사진의 위쪽 부분
	rectangle(dst, Point(dst.cols / 2 + 320, 0), Point(dst.cols, dst.rows), Scalar(0), FILLED);//차량의 오른쪽 부분
	imshow("delete area[" + to_string(idx) + "]", dst);//관심있는 부분만 남긴 결과 출력

	// Copy edge results to the images that will display the results in BGR
	cvtColor(dst, cdstP, COLOR_GRAY2BGR);//BRG로 변환하기, 색을 표현하기 위해
	//cdstP = cdst.clone();

	vector<Vec4i> linesP;//line 검출을 위한 벡터
	HoughLinesP(dst, linesP, 1, CV_PI / 180, 10, 5, 10);//line 검출 실시, 이때 line의 두점을 알아냄

	double theta = 0.0;//두점의 각도를 구하기 위해
	double dx = 0.0;//x방향 차이를 보기 위해
	double dy = 0.0;//y방향 차리을 보기 위해
//	double len = 0.0;

	double point_x_min = 0.0;//차선의 오른쪽 부분 x 좌표 검출하기 위한 변수
	double point_x_min1 = 0.0;//차선 왼쪽부분의 x 좌표를 검출하기 위한 변수
	double point_y_min = 0.0;//차선 오른쪽 부분 y좌표를 검출하기 위한 변수
	double point_y_min1 = 0.0;//차선 왼쪽 부분의 y좌표를 검출하기 위한 변수
	int max_idx = 0;//여러 개의 라인 중 왼쪽 차선 line index
	int min_idx = 0;//여러 개의 라인 중 오른쪽 차선 line index

	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		dx = l[2] - l[0], dy = l[3] - l[1];//dx, dy곗산
		//len = sqrt(dx * dx + dy * dy);

		if (dx == 0) {
			theta = CV_PI / 2;  // 수직선일 경우 90도 (π/2 라디안)
		}
		else {
			theta = atan2(dx, dy) * 180 / CV_PI;  // atan2를 사용하면 dx=0도 안전하게 처리됨
		}

		if (theta > 90) {
			//왼쪽 차선 영역, 이때 line중 차량과 제일 가까운 부분 즉 안쪽 line을 검출하기 위한 과정
			if (i == 0) {
				//임의의 값 설정
				point_x_min1 = l[0];
				point_y_min1 = l[1];
				max_idx = i;
			}
			else {
				//x좌표가 큰 경우 안쪽라인일 가능성이 높음 이때 x좌표가 큰 동시에 y좌표도 크면 안쪽 차선임
				if (l[0] > point_x_min1 && l[1] > point_y_min1) {
					point_x_min1 = l[0];//현재 x좌표값 저장
					point_y_min1 = l[1];//현재 y좌표값 저장
					max_idx = i;//현재 인덱스 저장
				}
			}
		}
		if (theta < 90) {
			//오른쪽 차선 영역, 이때 line중 차량과 제일 가까운 부분 즉 안쪽 line을 검출하기 위한 과정
			if (i == 0) {
				//임의의 값 설정
				point_x_min = l[0];
				point_y_min = l[1];
				min_idx = i;

			}
			else {
				//x좌표가 작은 경우 안쪽라인일 가능성이 높음 이때 x좌표가 작은 동시에 y좌표도 크면 안쪽 차선임
				if (l[0] < point_x_min && l[1]>point_y_min) {
					point_x_min = l[0];//현재 x좌표값 저장
					point_y_min = l[1];//현재 y좌표값 저장
					min_idx = i;//현재 인덱스 저장
				}
			}
		}
	}

	double extend_left_x1, extend_left_x2 = 0.0;//왼쪽 연장선 x좌표
	double extend_right_x1, extend_right_x2 = 0.0;//오른쪽 연장선 x좌표
	double inter_x, inter_y = 0.0;//교점 x, y좌표
	double m1, m2, b1, b2 = 0;//m1, m2: 기울기, b1, b2: y절편

	//왼쪽 직선 부분
	dx = linesP[max_idx][2] - linesP[max_idx][0];//dx계산
	dy = linesP[max_idx][3] - linesP[max_idx][1];//dy계산
	//extend_left_x1 = (0 - linesP[max_idx][1]) / (dy / dx) + linesP[max_idx][0];//y값이 
	m1 = dy / dx;//기울기 계산
	b1 = linesP[max_idx][1] - m1 * linesP[max_idx][0];//y절편 계산

	extend_left_x2 = (dst.rows - linesP[max_idx][1]) / (dy / dx) + linesP[max_idx][0];//y좌표가 최대값일 때 x값 구하기
	Point left_2 = Point(extend_left_x2, dst.rows);//왼쪽 직선의 제일 밑 부분의 좌표


	//오른쪽 직선 부분
	dx = linesP[min_idx][2] - linesP[min_idx][0];//dx 계산
	dy = linesP[min_idx][3] - linesP[min_idx][1];//dy 계산
	//extend_right_x1 = (0 - linesP[min_idx][1]) / (dy / dx) + linesP[min_idx][0];
	m2 = dy / dx;//기울기 계산
	b2 = linesP[min_idx][1] - m2 * linesP[min_idx][0];//y절편 계산

	extend_right_x2 = (dst.rows - linesP[min_idx][1]) / (dy / dx) + linesP[min_idx][0];//y좌표가 최대값일 때 x값 구하기
	Point right_2 = Point(extend_right_x2, dst.rows);//오른쪽 직선의 제일 밑 부분의 좌표

	inter_x = (b2 - b1) / (m1 - m2);//교점의 x 좌표
	inter_y = m1 * inter_x + b1;//교점의 y 좌표

	line(SRC, Point(inter_x, inter_y), left_2, Scalar(0, 0, 255), 3, LINE_AA);//왼쪽 차선, 빨간색으로 표현
	line(SRC, Point(inter_x, inter_y), right_2, Scalar(0, 255, 0), 3, LINE_AA);//오른쪽 차선, 초록색으로 표현

	circle(SRC, Point(inter_x, inter_y), 10, Scalar(0, 200, 255),2);//교점 부분, 차량의 중심부분
	line(SRC, Point(inter_x, inter_y), Point(inter_x, dst.rows), Scalar(255, 0, 0), 1, LINE_AA);//교점과 차량을 연결한 직선 이것을 기준으로 차선을 똑바로 가고 있는지 판단 가능함
	
	//삼각형 그리기
	vector<vector<Point>> fillVec = { {Point(inter_x, inter_y),right_2,left_2 } };
	double alpha = 0.8;
	
	// 원본 이미지와 같은 크기의 mask 이미지 만들기 (검정 배경, 3채널)
	Mat mask = Mat::zeros(SRC.size(), SRC.type());
	fillPoly(mask, fillVec, Scalar(100, 100, 255));

	//가중치를 줘 삼각형을 투명하게 만들기
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