#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

bool verifySizes(RotatedRect mr);
Mat histeq(Mat in);

int main(int argc, char* argv[])
{
	Mat img_gray = imread("../3.JPG", IMREAD_GRAYSCALE);
	Mat input = imread("../3.JPG");

	//高斯滤波 去除环境噪声，否则会有许多垂直边缘
	blur(img_gray, img_gray, Size(5, 5));

	imshow("source", img_gray);
	//使用Sobel寻找垂直边缘
	Mat img_sobel;
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	imshow("sobel", img_sobel);

	//二值化阈值选择
	Mat img_threshold;
	threshold(img_sobel, img_threshold, 0, 255, THRESH_OTSU + THRESH_BINARY );
	imshow("binary", img_threshold);

	//形态学操作
	Mat element = getStructuringElement(MORPH_RECT , Size(17, 3));
	morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);
	imshow("morphologEx", img_threshold);

	//查找可能是车牌的轮廓
	std::vector<std::vector<Point> > contours;
	findContours(img_threshold,
		contours,
		RETR_EXTERNAL ,
		CHAIN_APPROX_NONE 
		);
	
	std::vector<std::vector<Point> >::iterator itc = contours.begin();
	std::vector<RotatedRect> rects;

	//移除不满足长宽比的区域
	while(itc!= contours.end()){
		//创建物体的边界
		RotatedRect mr = minAreaRect(Mat(*itc));
		if(!verifySizes(mr)){
			itc = contours.erase(itc);
		}else{
			++itc;
			rects.push_back(mr);
		}

	}

	Mat result;
	 input.copyTo(result);
	// drawContours(result, contours,
	// 	-1,
	// 	Scalar(0,0,255),
	// 	3);

	// imshow("result",result);
	
	for (int i=0; i< rects.size(); i++)
	{

		//画出每个矩形的中心
		circle(result, rects[i].center, 3, Scalar(0, 255, 0), -1);
		//得到宽度和高度之间的最小尺寸
		float minSize = (rects[i].size.width < rects[i].size.height)?rects[i].size.width:rects[i].size.height;
		minSize = minSize - minSize * 0.5;
		//initialize rand and get 5 points around center floodfill algorithm
		srand(time(NULL));
		//initialize floodfill parameters and variables
		Mat mask;
		mask.create(input.rows+2, input.cols +2, CV_8UC1);
		mask = Scalar::all(0);
		int loDiff = 30;
		int upDiff = 30;
		int connectivity = 4;
		int newMaskVal = 255;
		int NumSeeds = 10;
		Rect ccomp;
		int flags = connectivity + (newMaskVal << 8) + CV_FLOODFILL_FIXED_RANGE +  CV_FLOODFILL_MASK_ONLY;
		for (int j=0; j<NumSeeds; j++){
			Point seed;
			seed.x = rects[i].center.x + rand()%(int)minSize - (minSize/2);
			seed.y = rects[i].center.y + rand()%(int)minSize - (minSize/2);
			int area = floodFill(input, mask, seed, Scalar(255, 0, 0), &ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags);
		}
		std::vector<Point> pointsInterest;
		Mat_<uchar>::iterator itMask = mask.begin<uchar>();
		Mat_<uchar>::iterator end = mask.end<uchar>();
		for(; itMask != end; ++itMask)
			if (*itMask == 255)
			{
				pointsInterest.push_back(itMask.pos());
			}
		RotatedRect minRect = minAreaRect(pointsInterest);

		if(verifySizes(minRect)){
			//rotated rectangle drawing
			Point2f rect_points[4]; minRect.points(rect_points);
			for(int j=0; j<4; j++)
				line(result, rect_points[j], rect_points[(j+1)%4], Scalar(0, 0, 255), 1, 8);
			//得到旋转矩阵
			float r = (float)minRect.size.width / (float)minRect.size.height;
			float angle = minRect.angle;
			if(r<1)
				angle = 90+angle;

			Mat rotmat= getRotationMatrix2D(minRect.center, angle,1); 
			//create and rotate imgage
			Mat img_rotated;
			warpAffine(input, img_rotated, rotmat, input.size(), CV_INTER_CUBIC);
			
			//crop image
			Size rect_size = minRect.size;
			if(r<1)
				swap(rect_size.width, rect_size.height);
			Mat img_crop;
			getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

			Mat resultResized;
			resultResized.create(33, 144, CV_8UC3);
			resize(img_crop, resultResized, resultResized.size(), 0, 0, CV_INTER_CUBIC);
			//Equalize croped image
			Mat grayResult;
			cvtColor(resultResized, grayResult, CV_BGR2GRAY);
			blur(grayResult, grayResult, Size(3,3));
			grayResult = histeq(grayResult);
			if(1){  
                stringstream ss(stringstream::in | stringstream::out); 
                ss << "haha" << "_" << i << ".jpg"; 
                imwrite(ss.str(), grayResult); 
            }
			//imshow("gray", grayResult);
		}

		//name = sprintf("%d",i);
	// 	sprintf(name, "%d", i);
	// 	imshow(name, result);
	 }
	 cout << rects.size()<<endl;
	 imshow("result", result);

	char key;
	while(1)
	{
		key = waitKey(1);
		if(key == 'q' || key == 'Q')
			break;

	}

	return 0;
}

bool verifySizes(RotatedRect mr){
	float error = 0.4;
	//Spain car plate size:52x11 aspect4.7272
	float aspect = 4.7272;
	//设置最大最小区域，其他区域舍弃
	int min = 5*aspect*5;
	int max = 125*aspect*125;
	//Get only patchs that match to a respect ratio.
	float rmin = aspect - aspect * error;
	float rmax = aspect + aspect * error;

	int area = mr.size.height * mr.size.width;
	float r = (float)mr.size.width/(float)mr.size.height;
	if(r<1)
		r = (float)mr.size.height/(float)mr.size.width;

	if((area < min || area > max) || (r < rmin || r > rmax)){
		return false;
	}
	else{
		return true;
	}
}
//直方图均衡化
Mat histeq(Mat in){
	Mat out(in.size(), in.type());
	if(in.channels() == 3){
		Mat hsv;
		std::vector<Mat> hsvSplit;
		cvtColor(in, hsv, CV_BGR2HSV);
		split(hsv, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, hsv);
		cvtColor(hsv, out, CV_HSV2BGR);
	}else if(in.channels()==1){
		equalizeHist(in, out);
	}
	return out;
}
