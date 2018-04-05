#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/ml.hpp"
#include <iostream>
#include "./src/Plate.h"

#define HORIZONTAL    1  
#define VERTICAL    0 

using namespace std;
using namespace cv;
using namespace cv::ml;

bool verifySize(Mat r);
Mat preprocessChar(Mat in);
//Mat features(Mat in, int sizeData, int count);
Mat features(Mat in, int sizeData);
Mat getVisualHistogram(Mat *hist, int type);
void drawVisualFeatures(Mat character, Mat hhist, Mat vhist, Mat lowData, int count);
Mat ProjectedHistogram(Mat img, int t);
int classify(Mat f);
void trian (Mat TrainingData, Mat classes, int nlaysers);
Ptr<ANN_MLP> ann = ANN_MLP::create();
//ANN_MLP ann_c;

const char strCharacters[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',\
						'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',\
						 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'};
const int numCharacters = 30;

int main(int argc, char * argv[])
{
	Mat input = imread("../1.jpg", IMREAD_GRAYSCALE );
	Plate mplate;

	//Read file storage
	FileStorage fs;
	fs.open("../OCR.xml", FileStorage::READ);
	Mat TrainingData;
	Mat Classes;
	fs["TrainingDataF5"] >> TrainingData;
	fs["classes"] >> Classes;
	//训练神经网络
 	trian(TrainingData, Classes, 10);
 	// cout<< TrainingData.size()<< endl;
 	// cout<< TrainingData.cols <<endl;
 	// cout<< TrainingData.rows << endl;

 // 	int character = classify(TrainingData.row(5));
 // 	 Mat output;
 // 	 ann->predict(TrainingData.row(1), output);
 // 	 cout << character<< endl;
 // 	 cout<< output << endl;
 // 	 Point maxLoc;
	// double maxVal;
	// minMaxLoc(output, NULL, &maxVal, NULL, &maxLoc);
	// cout<< maxVal<< endl;
	// cout<< maxLoc.x<< endl;
 	// cout<< output.size()<< endl;
 	// cout<< output.row(0)<< endl;


	// char res[20];
	int i = 0;
	//std::vector<char> v;
	//二值化图像
	Mat img_threshold;
	threshold(input, img_threshold, 60, 255, THRESH_BINARY_INV );
	imshow("二值化", img_threshold);

	Mat img_contours;
	img_threshold.copyTo(img_contours);

	//找到可能特征轮廓
	std::vector< vector<Point> > contours;
	findContours(img_contours,
		contours,
		RETR_EXTERNAL ,
		CHAIN_APPROX_NONE );
	//cout<<contours.size()<<endl;
	//cout<<<<endl;
	//在白色图像上画出蓝色轮廓
	 Mat result;
	 input.copyTo(result);
	 cvtColor(result, result, COLOR_GRAY2RGB);
	// drawContours(result, contours,
	// 	-1,
	// 	Scalar(0, 0, 255),
	// 	1
	// 	);
	// imshow("contours", result);

	std::vector<std::vector<Point> > ::iterator itc = contours.begin();
	while(itc != contours.end())
	{
		//计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的
		Rect mr = boundingRect(Mat(*itc));
		//crop image
		Mat auxRoi(img_threshold, mr);
		//cout << itc->size() << endl;
		if (verifySize(auxRoi))
		{
			auxRoi = preprocessChar(auxRoi);

			// sprintf(res, "train_data_%d.jpg", i);
			 i++;
			// imwrite(res, auxRoi);
			// rectangle(result, mr, Scalar(0, 0, 255), 2);

			//对每一个小方块，提取直方图特征
			Mat f = features(auxRoi, 5);
//			cout << auxRoi.size() << endl;
			cout << f.size() << endl;

			int character = classify(f);
			//cout << character<< endl;
			mplate.chars.push_back(strCharacters[character]);
			mplate.charsPos.push_back(mr);
		}
		
		
		++itc;
	}

	// imwrite("reault1.jpg", result);
	// imshow("car_plate", result);


	char key;
	 string licensePlate = mplate.str();
	 cout<< licensePlate << endl;
	while(1)
	{
		key = waitKey(1);
		if(key == 'q' || key == 'Q')
			break;
	}
	return 0;
}

bool verifySize(Mat r)
{
	//字符尺寸 45*77
	float aspect = 45.0f/77.0f;
	float charAspect=(float)r.cols/(float)r.rows; 
//	float charAspect　= (float)r.cols/(float)r.rows;
	float error = 0.35;
	float minHeight = 15;
	float maxHeight = 28;
	//数字1的宽高比为０．２
	float minAspect = 0.2;
	float maxAspect = aspect + aspect * error;
	//像素面积
	float area = countNonZero(r);
	//bb面积
	float bbArea = r.cols * r.rows;
	//像素面积
	float percPixels = area/bbArea;

	if (percPixels < 0.8 && charAspect>minAspect && charAspect< maxAspect && r.rows >= minHeight && r.rows < maxHeight)
	{
		return true;
	}
	else 
		return false;

}

Mat preprocessChar(Mat in)
{
	//remap image
	int h = in.rows;
	int w = in.cols;
	int charSize = 20; //统一字符大小
	Mat transformMat = Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = m/2 - w/2;
	transformMat.at<float>(1, 2) = m/2 - h/2;

	Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

	Mat out;
	resize(warpImage, out, Size(charSize, charSize));

	return out;
}

//Mat features(Mat in, int sizeData, int count)
Mat features(Mat in, int sizeData)
{
	//直方图特征
	Mat vhist = ProjectedHistogram(in, VERTICAL);
	Mat hhist = ProjectedHistogram(in, HORIZONTAL);

	//low data feature
	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData));

	//画出直方图
	//drawVisualFeatures(in, hhist, vhist, lowData, count);


	//last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols + lowData.cols*lowData.cols;

	Mat out = Mat::zeros(1, numCols, CV_32F);
	//assign values to feature, ANN的样本特征为水平、垂直直方图和低分辨率图像所组成的矢量
	int j = 0;
	for(int i=0; i<vhist.cols; i++)
	{
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for(int i=0; i < hhist.cols; i++)
	{
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	for(int x=0; x<lowData.rows; x++)
	{
		for(int y=0; y<lowData.rows; y++)
		{
			out.at<float>(j)=(float)lowData.at<unsigned char>(x,y);
			j++;
		}
	}
	return out;
}

//创建累加的直方图，img为二值图，　t是方向：水平或竖直
Mat ProjectedHistogram(Mat img, int t)
{
	int sz = (t)?img.rows:img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for(int j=0; j<sz; j++)
	{
		Mat data = (t)?img.row(j):img.col(j);
		mhist.at<float>(j) = countNonZero(data); //统计这一行或一列中，非零元素的个数，并保存到mhisｔ中
	}
	//normalize histogram
	double min, max;
	minMaxLoc(mhist, &min, &max);

	if(max>0)
		mhist.convertTo(mhist, -1, 1.0f/max, 0); //用mhist直方图中的最大值，归一化直方图

	return mhist;
}

void drawVisualFeatures(Mat character, Mat hhist, Mat vhist, Mat lowData, int count)
{
	Mat img(121, 121, CV_8UC3, Scalar(0, 0, 0));
	Mat ch;
	Mat ld;
	char res[20];

	cvtColor(character, ch, CV_GRAY2RGB);

	resize(lowData,ld, Size(100, 100), 0, 0, INTER_NEAREST); //将ld从１５＊１５扩大到１００＊１００
	cvtColor(ld, ld, CV_GRAY2RGB);

	Mat hh = getVisualHistogram(&hhist, HORIZONTAL);
	Mat hv = getVisualHistogram(&vhist, VERTICAL);

	//Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height)
	Mat subImg = img(Rect(0, 101, 20, 20)); //ch:20*20
	ch.copyTo(subImg);

	subImg = img(Rect(21, 101, 100, 20));  //hh:hist.cols*100
	hh.copyTo(subImg);

	subImg = img(Rect(0, 0, 20,100)); //hv:hist.cols*100
	hv.copyTo(subImg);

	subImg = img(Rect(21, 0, 100, 100)); //ld:hist.cols*100
	ld.copyTo(subImg);

	line(img, Point(0, 100), Point(121, 100), Scalar(0, 0, 255));
	line(img, Point(20, 0), Point(20, 121), Scalar(0, 0, 255));

	sprintf(res, "hist%d.jpg", count);
	imshow(res, img);
	imwrite(res, img);

	cvWaitKey(0);
}

Mat getVisualHistogram(Mat *hist, int type)
{
	int size = 100;
	Mat imHist;

	if (type == HORIZONTAL)
	{
		imHist.create(Size(size, hist->cols), CV_8UC3);
	}
	else
	{
		imHist.create(Size(hist->cols, size), CV_8UC3);
	}

	imHist = Scalar(55, 55, 55);

	for (int i = 0; i < hist->cols; ++i)
	{
		float value = hist->at<float>(i);
		int maxval = (int)(value*size);

		Point pt1;
		Point pt2, pt3, pt4;

		if(type == HORIZONTAL)
		{
			pt1.x = pt3.x = 0;
			pt2.x = pt4.x = maxval;
			pt1.y = pt2.y = i;
			pt3.y = pt4.y = i+1;

			line(imHist, pt1, pt2, Scalar(220, 220, 220), 1, LINE_8, 0);
			line(imHist, pt3, pt4, Scalar(34, 34, 34), 1, LINE_8, 0);

			pt3.y = pt4.y = i+2;
			line(imHist, pt3, pt4, Scalar(44, 44, 44), 1, LINE_8, 0);
			pt3.y = pt4.y = i+3;
			line(imHist, pt3, pt4, Scalar(50, 50, 50), 1, LINE_8, 0);

		}		
		else
		{
			pt1.x = pt2.x = i;
			pt3.x = pt4.x = i+1;
			pt1.y = pt3.y = 100;
			pt2.y = pt4.y = 100- maxval;

			line(imHist, pt1, pt2, Scalar(220, 220, 220), 1, LINE_8, 0);
			line(imHist, pt3, pt4, Scalar(34, 34, 34), 1, LINE_8, 0);

			pt3.y = pt4.y = i+2;
			line(imHist, pt3, pt4, Scalar(44, 44, 44), 1, LINE_8, 0);
			pt3.y = pt4.y = i+3;
			line(imHist, pt3, pt4, Scalar(50, 50, 50), 1, LINE_8, 0);				
		}	
	}
	return imHist;
}

void trian (Mat TrainingData, Mat classes, int nlaysers)
{
	// Mat layers(1, 3, CV_32SC1);
	// layers.at<int>(0) = TrainingData.cols;
	// layers.at<int>(1) = nlaysers;
	// layers.at<int>(2) = 30;
	Mat layerSizes =  (Mat_<int>(1, 3) << TrainingData.cols, nlaysers, 30);
	ann->setLayerSizes(layerSizes);
	ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, FLT_EPSILON));
	ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001);
	
	//Prepare trainClasses
	//Create a mat with n trained data by m classes
	Mat trainClasses;
	trainClasses.create(TrainingData.rows, 30, CV_32FC1);
	for(int i = 0 ; i<trainClasses.rows; i++)
	{
		for(int k = 0; k < trainClasses.cols; k++)
		{
			//If clsaa of data i is same as a k class
			if ( k == classes.at<int>(i))
				trainClasses.at<float>(i, k) = 1;
			else 
				trainClasses.at<float>(i, k) = 0;
		}
	}
	//Mat weights(1, TrainingData.rows, CV_32FC1, Scalar::all(1) );
	Ptr<TrainData> tData = TrainData::create(TrainingData, ROW_SAMPLE, trainClasses);
	//learn classifier
	ann->train(tData);
	//ann_c.train(TrainingData, trainClasses, weights);
	// Mat output;
	// ann->calcError(tData, false, output);
	// cout<< output<< endl;
}
int classify(Mat f)
{
	int result = -1;
	Mat output (1, 30, CV_32FC1);
	ann->predict(f, output);
	//cout<< output<< endl;
	Point maxLoc;
	double maxVal;
	minMaxLoc(output, NULL, &maxVal, NULL, &maxLoc);
	//cout<< maxLoc<< endl;

	return maxLoc.x;
}