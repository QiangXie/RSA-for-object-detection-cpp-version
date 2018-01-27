#ifndef _RSA_FACE_DETECTION_HPP_
#define _RSA_FACE_DETECTION_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <algorithm>
#include <cmath>
#include <Eigen/LU>
#include <Eigen/Dense>
#include "caffe/caffe.hpp"
#include "config.hpp"
#include "gpu_nms.hpp"

struct Face{
	std::vector<double> bbox;
	std::vector<cv::Point2f> key_points;
	float score;
};

bool comp(const Face & a, const Face & b);

Eigen::MatrixXd findNonreflectiveSimilarity(const cv::Point2f uv[], const cv::Point2f xy[]);

cv::Mat getSimilarityTransform(const cv::Point2f uv[], const cv::Point2f xy[]);

void getTripPoints(std::vector<cv::Point2f> &dst_rect, cv::Point2f src_key_point[]);

class RsaFaceDetector{
	public:
		explicit RsaFaceDetector(int gpu_id);
		std::vector<Face> detect(cv::Mat image);
		void sfnProcess(const cv::Mat & img);
		void rsaProcess(void);
		void lrnProcess(std::vector<Face> &faces_out);
	private:
		std::string sfn_net_def = SFN_NET_DEF;
		std::string sfn_net_weight = SFN_NET_WEIGHT;
		std::string rsa_net_def = RSA_NET_DEF;
		std::string rsa_net_weight = RSA_NET_WEIGHT;
		std::string lrn_net_def = LRN_NET_DEF;
		std::string lrn_net_weight = LRN_NET_WEIGHT;
		int gpu_id_;
		std::vector<std::shared_ptr<caffe::Blob<float> > > trans_featmaps;
		caffe::Blob<float> * sfn_net_output;
		caffe::Blob<float> * input_layer;
		caffe::Blob<float> * rsa_input_layer;
		caffe::Blob<float> * lrn_input_layer;
		std::vector<cv::Mat> input_channels;
		double resize_factor;
		std::vector<float> anchor_box_len;
		double thresh_score;
		double stride;
		double anchor_center;
		std::vector<int> scale;
		std::shared_ptr<caffe::Net<float> > sfn_net;
		std::shared_ptr<caffe::Net<float> > rsa_net;
		std::shared_ptr<caffe::Net<float> > lrn_net;
};

#endif

