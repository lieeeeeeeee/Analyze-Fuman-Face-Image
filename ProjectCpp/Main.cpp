#include "UserFunctions.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include<string>
#include<vector>

using namespace std;
using std::cout;

vector<string> featuresStr = {
	"画素値",
	"ヒストグラム",
	"Bag of Words"
};
vector<string> kernelsStr = {
	"線形",
	"ヒストグラム交差",
	"放射基底関数"
};

void outputCsv(vector<string> headers, vector<vector<int>> values, vector<int> indexes, string name) {
	string fileName = name + ".csv";
	ofstream outputfile(fileName);
	int headersSize = size(headers);
	int valuesSize = size(values);
	int valueSize = size(values[0]);

	outputfile << endl;
	outputfile << "" << "," << flush;
	for (int i = 0; i < headersSize; i++) {
		outputfile << headers[i] << "," << flush;
	}
	outputfile << endl;

	for (int i = 0; i < valueSize; i++) {
		outputfile << indexes[i] << "," << flush;
		for (int j = 0; j < valuesSize; j++) {
			outputfile << values[j][i] << "," << flush;
		}
		outputfile << endl;
	}

	outputfile.close();
}
vector<int> userSelect(string title, vector<string> item) {
	int length = size(item);
	int j = 0;
	string input;
	vector<int> vec(length, 0);
	cout << "--------------------------------" << endl;
	cout << "* 使用する"+ title + "を選択してください" << endl;

	for (int i = 0; i < length; i++) {
		do {
			cout << item[i] + ": (Y/N) >> " << flush;
			cin >> input;
		} while (input != "Y" && input != "N");
		if (input == "Y") vec[j++] = i+1;
	}
	return vec;
}
void faceExtractSetup() {
	try {
		faceExtract("Face");
	}
	catch (const std::exception& e) {
		cout << e.what() << endl;
	}
}
void imgLearnigSetup() {
	int learnigNum = 4;
	int testNum = 10;
	int eachItems = 3;

	vector<string> headers;
	vector<vector<int>> valuesAll;
	vector<vector<int>> valuesEach;
	vector<int> indexes;
	vector<int> feature, kernel, course;
	

	string fileName;

	feature = userSelect("特徴量", featuresStr);
	kernel = userSelect("カーネル", kernelsStr);
	course = userSelect("コース", { "学習回数自動変更コース" });

	cout << "--------------------------------" << endl;
	cout << "csvファイルの名前を入力してください" << endl;
	cout << ">> " << flush;
	cin >> fileName;
	cout << "--------------------------------" << endl;
	if (course[0] == 1) {
		int featureIndex = feature[0];
		int kernelIndex = kernel[0];
		if (featureIndex == 0) return;
		if (kernelIndex == 0) return;
		int maxTestNum = 32;
		vector<int> valueAll;
		string header = featuresStr[featureIndex - 1] + " X " + kernelsStr[kernelIndex - 1];
		for (int i = 2; i <= maxTestNum; i *= 2) {
			int correctNumAll = 0;
			float correctRateAll = 0;
			
			vector<int> correctNumEach(eachItems, 0);
			vector<int> correctRateEach(eachItems+1, 0);
			int eachIndex = 0;
			learnigNum = i;
			vector<float> vec = imgLearnig(featureIndex, kernelIndex, learnigNum, testNum, "caltech101");

			cout << header << endl;
			cout << i << "回学習中..." << endl;

			int testNum = size(vec);
			
			if (testNum == 0) {
				cout << "学習に失敗しました" << endl; return;
			}
			for (int k = 0; k < testNum; k++) {
				if (k % 10 == testNum - 1) {
					correctRateEach[eachIndex] = (float)correctNumEach[eachIndex] / (float)testNum * 100;
					eachIndex++;
				}
				if (vec[k] == 1) {
					correctNumEach[eachIndex]++;
					correctNumAll++;
				}
				correctRateAll = (float)correctNumAll / (float)(k + 1);
			}
			indexes.push_back(i);
			valueAll.push_back(correctRateAll * 100);
			valuesEach.push_back(correctRateEach);
		}
		cout << header << endl;
		headers.push_back(header);
		valuesAll.push_back(valueAll);

	}
	else {
		for (int i = 0; i < size(feature); i++) {
			int featureIndex = feature[i];
			if (featureIndex == 0) break;
			for (int j = 0; j < size(kernel); j++) {
				vector<int> valueAll;
				vector<int> correctNumEach(eachItems, 0);
				vector<int> correctRateEach(eachItems+1, 0);
				int correctNumAll = 0;
				int eachIndex = 0;
				int kernelIndex = kernel[j];
				float correctRateAll = 0;
				if (kernelIndex == 0) break;
				string header = featuresStr[featureIndex - 1] + " X " + kernelsStr[kernelIndex - 1];

				cout << header << endl;
				cout << "10回学習中..." << endl;

				vector<float> vec = imgLearnig(featureIndex, kernelIndex, learnigNum, testNum, "caltech101");
				int testTotal = size(vec);

				if (testTotal == 0) {
					cout << "学習に失敗しました" << endl; continue;
				}
				indexes.resize(testTotal);
				cout << "--------------------------------" << endl;
				headers.push_back(header);

				for (int k = 0; k < testTotal; k++) {
					
					if (vec[k] == 1) {
						correctNumEach[eachIndex]++;
						correctNumAll++;
					}
					if (k % 10 == testNum - 1) {
						correctRateEach[eachIndex] = (float)correctNumEach[eachIndex] / (float)testNum * 100;
						eachIndex++;
					}
					correctRateAll = (float)correctNumAll / (float)(k + 1);
					if (k == testTotal-1)  correctRateEach[eachIndex] = correctRateAll * 100;
							
					valueAll.push_back(correctRateAll * 100);
					indexes[k] = k;
				}
				valuesAll.push_back(valueAll);
				valuesEach.push_back(correctRateEach);
			}
		}
	}
	outputCsv(headers, valuesAll, indexes, fileName + "-all");
	outputCsv(headers, valuesEach, {1,2,3,4}, fileName + "-each");
}

int main(void) {
	int user_input;

	cout << "--------------------------------" << endl;
	cout << "使用したい機能を選択してください" << endl;
	cout << "1: 画像から顔を抽出する" << endl;
	cout << "2: 画像を学習し分類する" << endl;

	do {
		cout << "(1 or 2) >> " << flush;
		cin >> user_input;
		//if user_input is not integer, user_input is 0
		if (cin.fail()) {
			cin.clear();
			cin.ignore();
			user_input = 0;
		}
	} while (user_input != 1 && user_input != 2);

	switch (user_input) {
	case 1:
		
		faceExtractSetup();
		

		break;
	case 2:
		try {
			imgLearnigSetup();
		}
		catch (const std::exception& e) {
			cout << e.what() << endl;
		}
		break;
	default:
		break;
	}

	return 0;
}


