#ifndef Plate_h
#define Plate_h

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>


using namespace std;
using namespace cv;
using namespace cv::ml;

class Plate{
public:
	Plate();
	Plate(Mat img, Rect pos);
	string str();
	Rect position;
	Mat plateImg;
	std::vector<char> chars;
	std::vector<Rect> charsPos;
};


#endif