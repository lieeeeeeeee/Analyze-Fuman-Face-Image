#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "UserFunctions.h"

const int BOW_GRID_X = 8;
const int BOW_GRID_Y = 8;
float BOW_SCALE = 4.0f;
const float BOW_ANGLE = 0.0f;
const int BOW_N_VOCAB = 200;

using namespace std;
using namespace cv;
using std::vector;

map< int, vector<Mat> > load_images(
	string name)
{
	ifstream data_list(name);
	if (data_list.fail()) {
		cerr << "cannot open file: " << name << endl;
		return map< int, vector<Mat> >();
	}

	int i = 0;
	map< int, vector<Mat> > images;
	while (!data_list.eof() && i++ < 500000) {
		string image_file_name;
		int label;
		data_list >> image_file_name >> label;
		Mat image = imread(image_file_name);
		images[label].push_back(image);
	}

	return images;
}


void sample_datasets_from_images(
	map< int, vector<Mat> >& images,
	size_t n_train,
	size_t n_test,
	vector<Mat>& train_images,
	Mat& train_labels,
	vector<Mat>& test_images,
	Mat& test_labels)
{
	train_images.clear();
	test_images.clear();

	vector<int> vec_train_labels;
	vector<int> vec_test_labels;

	map< int, vector<Mat> >::iterator it;
	for (it = images.begin(); it != images.end(); ++it) {
		int l = it->first;
		vector<Mat>& image_set = it->second;
		random_shuffle(image_set.begin(), image_set.end());

		size_t n_images = image_set.size();
		for (size_t i = 0; i < n_train && i < n_images; ++i) {
			train_images.push_back(image_set[i]);
			vec_train_labels.push_back(l);
		}
		for (size_t i = n_train, j = 0; i < n_images && j < n_test; ++i, ++j) {
			test_images.push_back(image_set[i]);
			vec_test_labels.push_back(l);
		}
	}

	train_labels = Mat(vec_train_labels, true);
	test_labels = Mat(vec_test_labels, true);
}


Mat extract_pixels_as_features(
	vector<Mat>& images)
{
	const int W = 10;
	const int H = 10;

	Mat X(images.size(), W * H * 3, CV_32FC1);

	for (size_t i = 0; i < images.size(); ++i) {
		Mat resized(H, W, CV_8UC3);
		resize(images[i], resized, resized.size());
		Mat x = resized.reshape(1, 1);
		x.copyTo(X.row(i));
	}

	return X;
}


Mat extract_histogram(
	vector<Mat>& images)
{
	const int channels[] = { 0, 1, 2 };
	const int hsize[] = { 5, 5, 5 };
	const float range[] = { 0.0f, 256.0f };
	const float* ranges[] = { range, range, range };

	Mat X(images.size(), hsize[0] * hsize[1] * hsize[2], CV_32FC1);

	for (size_t i = 0; i < images.size(); ++i) {
		Mat hist;
		calcHist(&images[i], 1, channels, Mat(), hist, 3, hsize, ranges);
		hist /= images[i].size().width * images[i].size().height;
		MatConstIterator_<float> hit = hist.begin<float>();
		for (int j = 0; hit != hist.end<float>(); ++j, ++hit) {
			X.at<float>(i, j) = *hit;
		}

	}

	return X;
}


Mat train_bow(
	vector<Mat>& images)
{
	Ptr<SIFT> sift = SIFT::create();
	BOWKMeansTrainer trainer(BOW_N_VOCAB);

	for (size_t i = 0; i < images.size(); ++i) {
		int gw = images[i].size().width / BOW_GRID_X - 1;
		int gh = images[i].size().height / BOW_GRID_Y - 1;

		vector<KeyPoint> keypoints;
		for (int x = 1; x < gw; ++x) {
			for (int y = 1; y < gh; ++y) {
				keypoints.push_back(KeyPoint(
					float(x * BOW_GRID_X),
					float(y * BOW_GRID_Y),
					BOW_SCALE,
					BOW_ANGLE,
					0.0f,
					0
				));
			}
		}

		Mat desc;
		Mat gray;
		cvtColor(images[i], gray, COLOR_BGR2GRAY);
		sift->compute(gray, keypoints, desc);
		trainer.add(desc);
	}
	return trainer.cluster();
}


Mat extract_bow(
	Mat& vocab,
	vector<Mat>& images)
{
	Ptr<SIFT> sift = SIFT::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	BOWImgDescriptorExtractor extractor(sift, matcher);
	extractor.setVocabulary(vocab);

	Mat X(images.size(), extractor.descriptorSize(), extractor.descriptorType());

	for (size_t i = 0; i < images.size(); ++i) {
		int gw = images[i].size().width / BOW_GRID_X - 1;
		int gh = images[i].size().height / BOW_GRID_Y - 1;

		vector<KeyPoint> keypoints;
		for (int x = 1; x < gw; ++x) {
			for (int y = 1; y < gh; ++y) {
				keypoints.push_back(KeyPoint(
					float(x * BOW_GRID_X),
					float(y * BOW_GRID_Y),
					BOW_SCALE,
					BOW_ANGLE,
					0.0f,
					0
				));
			}
		}

		Mat x;
		Mat gray;
		vector< vector<int> > indices;
		cvtColor(images[i], gray, COLOR_BGR2GRAY);
		extractor.compute(gray, keypoints, x, &indices);
		x.copyTo(X.row(i));
	}

	return X;
}


Ptr<ml::SVM> train_linear_svm(
	Mat& X,
	Mat& y)
{
	Ptr<ml::TrainData> data = ml::TrainData::create(
		X, ml::ROW_SAMPLE, y
	);

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setC(1000);
	svm->train(data);

	return svm;
}


Ptr<ml::SVM> train_histogram_intersection_kernel_svm(
	Mat& X,
	Mat& y)
{
	Ptr<ml::TrainData> data = ml::TrainData::create(
		X, ml::ROW_SAMPLE, y
	);

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::INTER);
	svm->setC(1000);
	svm->train(data);

	return svm;
}

Ptr<ml::SVM> train_radial_basis_function_kernel_svm(
	Mat& X,
	Mat& y)
{
	Ptr<ml::TrainData> data = ml::TrainData::create(
		X, ml::ROW_SAMPLE, y
	);

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::RBF);
	svm->setC(1000);
	svm->train(data);

	return svm;
}


vector<float> evaluate_accuracy(
	Ptr<ml::StatModel> classifier,
	Mat& test_samples,
	Mat& test_labels
) {
	Mat labels;
	test_labels.convertTo(labels, CV_32F);


	Ptr<ml::TrainData> data = ml::TrainData::create(
		test_samples, ml::ROW_SAMPLE, labels
	);
	Mat res;
	float err = classifier->calcError(data, false, res);
	int testNum = test_labels.rows;
	vector<float> vec(testNum, 0);

	for (size_t n = 0; n < testNum; ++n) {
		float correct = labels.at<float>(n, 0);
		float answer = res.at<float>(n, 0);
		cout << "correct: " << correct << ", answer: " << answer << endl;
		if (correct == answer) vec[n] = 1.0f;
	}
	return vec;
	
}


vector<float> imgLearnig(int feature, int kernel, int traing, int test, string dataPath) {
	Mat train_samples, train_labels, test_samples, test_labels;
	Ptr<ml::SVM> svm;
	BOW_SCALE = 4.0;

	//srand(time(0)); // 画像をランダムに選出する場合はコメントを解除する

	{
		map< int, vector<Mat> > images;
		//images = load_images("dataset/"+ dataPath +".txt");
		images = load_images("dataset/"+dataPath+".txt");

		vector<Mat> train_images, test_images;
		sample_datasets_from_images(
			images, traing, test,
			train_images, train_labels,
			test_images, test_labels
		);
		Mat vocab;
		switch (feature) {
		case 1: /* Pixel values */
			train_samples = extract_pixels_as_features(train_images);
			test_samples  = extract_pixels_as_features(test_images);
			break;
		case 2: /* Color histogram */
			train_samples = extract_histogram(train_images);
			test_samples  = extract_histogram(test_images);
			break;
		case 3: /* Bag of Words */
			vocab = train_bow(train_images);
			train_samples = extract_bow(vocab, train_images);
			test_samples = extract_bow(vocab, test_images);
			break;
		default:
			return {};
		}
	}
	switch (kernel) {
	case 1: /* Linear SVM */
		svm = train_linear_svm(train_samples, train_labels);
		svm->save("svm.yml");
		break;
	case 2: /* SVM with histogram intersection kernel */
		svm = train_histogram_intersection_kernel_svm(train_samples, train_labels);
		svm->save("svm.yml");
		break;
	case 3: /* SVM with radial basis function kernel */
		svm = train_radial_basis_function_kernel_svm(train_samples, train_labels);
		svm->save("svm.yml");					   
		break;
	default:
		return {};
	}

	Ptr<ml::SVM> saved_svm = Algorithm::load<ml::SVM>("svm.yml");
	vector<float> vec = evaluate_accuracy(saved_svm, test_samples, test_labels);

	system("pause");

	return vec;
}
