#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include "rsa_face_detection.hpp"


int main(){
	std::string img_list = "./image/list.txt";
	std::ifstream list(img_list);
	std::string  img_name;
	RsaFaceDetector *detector = new RsaFaceDetector(3);
	while(list >> img_name){
		cv::Mat img = cv::imread(img_name);
		std::vector<Face> faces = detector->detect(img);	
		for(int i = 0; i < faces.size(); ++i){
			std::cout << "Face detected after NMS: " 
				<< faces[i].bbox[0] << " " 
				<< faces[i].bbox[1] << " " 
				<< faces[i].bbox[2] << " " 
				<< faces[i].bbox[3] << " " 
				<< "Score : " << faces[i].score << "\n";
			cv::rectangle(img, cv::Point(faces[i].bbox[0], faces[i].bbox[1]),
					cv::Point(faces[i].bbox[2], faces[i].bbox[3]),
					cv::Scalar(0,0,255), 1, 1, 0);
			std::cout << "Key points: ";
			for(int j = 0; j < 5; ++j){
				cv::circle(img, faces[i].key_points[j], 1, cv::Scalar(0,255,0), 2);
				std::cout << faces[i].key_points[j].x << " " 
					<< faces[i].key_points[j].y << " ";
			}
			std::cout << std::endl;
		}
		cv::imshow("test", img);
		cv::waitKey(100);
	}
	delete detector;

	return 0;
}

