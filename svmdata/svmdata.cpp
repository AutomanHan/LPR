//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char * argv[])
{

	String Plate_path = "../pos";
	String Noplate_path = "../neg";
	bool addPath = false;

	std::vector<String> name_plate;
	std::vector<String> name_Noplate;

	//读取目录下所有文件名
	glob(Plate_path, name_plate, addPath);
	glob(Noplate_path, name_Noplate, addPath);

	int numPlates = name_plate.size();
	int numNoPlates = name_Noplate.size();
	
	Mat classes;    //numPlates + numNoPlates, 1, CV_32FC1
	Mat trainingData;  //numPlates+numNoPlates, imageWidth+imageHeight, CV_32FC1

	Mat trainingImages;
	std::vector<int> trainingLabels;

	for(int i = 0; i < numPlates; i++)
	{
		Mat img = imread(name_plate[i], 0);
		img = img.reshape(1, 1);
		trainingImages.push_back(img);
		trainingLabels.push_back(1);
	}

	for(int i = 0; i < numNoPlates; i++)
	{
		Mat img = imread(name_Noplate[i], 0);
		img = img.reshape(1, 1);
		trainingImages.push_back(img);
		trainingLabels.push_back(0);
	}

	Mat(trainingImages).copyTo(trainingData);

	trainingData.convertTo(trainingData, CV_32FC1);
	Mat(trainingLabels).copyTo(classes);
	classes.convertTo(classes, CV_32SC1);

	FileStorage fs("SVM.xml", FileStorage::WRITE);
	fs << "trainingData" << trainingData;
	fs << "classes" << classes;
	fs.release();



	return 0;
}