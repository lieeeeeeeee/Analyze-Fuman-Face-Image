#include <Windows.h>
#include <direct.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <locale> 
#include <codecvt> 
#include <cstdio>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "UserFunctions.h"

using namespace std;
using namespace cv;
using std::vector;

string ROOT_PATH = "C:\\ëÂäw\\èÓïÒââèKC\\ProjectCpp\\ProjectCpp\\dataset\\101_ObjectCategories\\";


vector<string> getImgNames(string folderPath) {
	vector<string> imgNames;

	HANDLE hFind;
	WIN32_FIND_DATA fd;
	wstring searchName = wstring(folderPath.begin(), folderPath.end()) + L"*.jpg";
	hFind = FindFirstFile(searchName.c_str(), &fd);

	if (hFind == INVALID_HANDLE_VALUE) {
		return imgNames;
	}
	do {
		wstring_convert<codecvt_utf8<wchar_t>> conv;
		string filename = conv.to_bytes(fd.cFileName);
		imgNames.push_back(filename);
	} while (FindNextFile(hFind, &fd));
	FindClose(hFind);
	return imgNames;
}

void saveImg(string rowImgPath, string outputImgPath, string outputFolderName, int index) {
	vector<string> imgNames = getImgNames(rowImgPath);
	string txtRoot= "C:\\ëÂäw\\èÓïÒââèKC\\ProjectCpp\\ProjectCpp\\dataset\\";
	string txtFileName = txtRoot + "caltech101_" + outputFolderName + ".txt";
	
	ofstream txtFile(txtFileName, ios::app);

	if (imgNames.size() == 0) {
		throw std::runtime_error("cannot find images");
	}

	for (int i = 0; i < imgNames.size(); i++) {
		string imgPath = rowImgPath + imgNames[i];
		string outputImgPath = outputImgPath + imgNames[i];
		string learnigDatasPath = "C:/opencv/build/etc/haarcascades/";
		string learnigDatasName = "haarcascade_frontalface_alt.xml";
		Mat rowImg = imread(imgPath);
		CascadeClassifier faceDetector;

		if (!faceDetector.load(learnigDatasPath + learnigDatasName)) {
			throw std::runtime_error("cannot load classifier");
		}
		vector<Rect> faces;
		faceDetector.detectMultiScale(rowImg, faces);

		if (faces.size() == 0) {
			imwrite(outputImgPath, rowImg);
			return;
		}

		for (int j = 0; j < faces.size(); j++) {
			Rect face = faces[j];
			Mat faceImg = rowImg(face);
			imwrite(outputImgPath, faceImg);
			txtFile << outputImgPath + " " + to_string(index) << std::endl;
		}
	}
	txtFile.close();
}

vector<string> getFolderNames(string folderPath) {
	vector<string> folderNames;
	HANDLE hFind;
	WIN32_FIND_DATA fd;
	wstring searchName = wstring(folderPath.begin(), folderPath.end()) + L"*";
	hFind = FindFirstFile(searchName.c_str(), &fd);
	if (hFind == INVALID_HANDLE_VALUE) {
		return folderNames;
	}
	do {
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			wstring_convert<codecvt_utf8<wchar_t>> conv;
			string filename = conv.to_bytes(fd.cFileName);
			if (filename != "." && filename != "..") {
				folderNames.push_back(filename);
			}
		}
	} while (FindNextFile(hFind, &fd));
	FindClose(hFind);
	return folderNames;
}
void faceExtract(string outputPath) {
	string rowImgsFolderPath = ROOT_PATH + "RowImgs\\";
	string outputImgFolderPath = ROOT_PATH + outputPath + "\\";
	vector<string> folderNames = getFolderNames(rowImgsFolderPath);

	_mkdir(outputImgFolderPath.c_str());

	cout << rowImgsFolderPath << endl;
	cout << outputImgFolderPath << endl;

	if (folderNames.size() == 0) {
		saveImg(rowImgsFolderPath, outputImgFolderPath, outputPath, 1);
	}
	else {
		for (int i = 0; i < folderNames.size(); i++) {
			string rowImgPath = rowImgsFolderPath + folderNames[i] + "\\";
			string outputImgPath = outputImgFolderPath + folderNames[i] + "\\";
			cout << rowImgPath << endl;
			cout << outputImgPath << endl;

			saveImg(rowImgPath, outputImgPath, outputPath, i);
		}
	}
}