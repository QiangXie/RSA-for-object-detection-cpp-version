#ifndef _RSA_FACE_DETECTION_HPP_
#define _RSA_FACE_DETECTION_HPP_

struct Face{
	std::vector<double> bbox;
	std::vector<cv::Point2f> key_points;
	float score;
};

bool comp(const Face & a, const Face & b);

Eigen::MatrixXd findNonreflectiveSimilarity(const cv::Point2f uv[], const cv::Point2f xy[]);

cv::Mat getSimilarityTransform(const cv::Point2f uv[], const cv::Point2f xy[]);

class RsaFaceDetector{
	public:
		explicit RsaFaceDetector();
		std::vector<Face> detect(cv::Mat image);
	private:
		int gpu_id;
		shared_ptr<Net<float> > net_res_pool2;
		shared_ptr<Net<float> > rsa_net;
		shared_ptr<Net<float> > detec_net;
};


#endif

