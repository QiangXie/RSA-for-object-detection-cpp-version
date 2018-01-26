#include "rsa_face_detection.hpp"

using namespace caffe;

bool comp(const Face & a, const Face & b){
	return a.score > b.score;
}


Eigen::MatrixXd findNonreflectiveSimilarity(const cv::Point2f uv[], const cv::Point2f xy[]){

	Eigen::MatrixXd X(10,4);
	Eigen::MatrixXd U(10,1);
	for(int i = 0; i < 5; i++){
		X(i,0) = xy[i].x;
		X(i,1) = xy[i].y;
		X(i,2) = 1.0;
		X(i,3) = 0.0;
		X(i+5,0) = xy[i].y;
		X(i+5,1) = -xy[i].x;
		X(i+5,2) = 0.0;
		X(i+5,3) = 1.0;

		U(i,0) = uv[i].x;
		U(i+5,0) = uv[i].y;
	}

	Eigen::MatrixXd r = X.jacobiSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(U);
	double sc = r(0,0);
	double ss = r(1,0);
	double tx = r(2,0);
	double ty = r(3,0);

	Eigen::MatrixXd Tinv(3,3);
	Tinv(0,0) = sc;
	Tinv(0,1) = -ss;
	Tinv(0,2) = 0.0;
	Tinv(1,0) = ss;
	Tinv(1,1) = sc;
	Tinv(1,2) = 0.0;
	Tinv(2,0) = tx;
	Tinv(2,1) = ty;
	Tinv(2,2) = 1.0;
	Eigen::MatrixXd T = Tinv.inverse();
	T(0,2) = 0.0;
	T(1,2) = 0.0;
	T(2,2) = 1.0;

	return T;
}

cv::Mat getSimilarityTransform(const cv::Point2f uv[], const cv::Point2f xy[]){
	Eigen::MatrixXd trans1 = findNonreflectiveSimilarity(uv, xy);
	cv::Point2f xy_new[5];

	for(int i = 0; i < 5; ++i){
		xy_new[i].x = -xy[i].x;
		xy_new[i].y = xy[i].y;
	}

	Eigen::MatrixXd trans2r = findNonreflectiveSimilarity(uv, xy_new);

	Eigen::MatrixXd TreflectY(3,3);
	TreflectY(0,0) = -1.0;
	TreflectY(0,1) = 0.0;
	TreflectY(0,2) = 0.0;
	TreflectY(1,0) = 0.0;
	TreflectY(1,1) = 1.0;
	TreflectY(1,2) = 0.0;
	TreflectY(2,0) = 0.0;
	TreflectY(2,1) = 0.0;
	TreflectY(2,2) = 1.0;

	Eigen::MatrixXd trans2 = trans2r * TreflectY;
	Eigen::MatrixXd trans1_inv = trans1.inverse();
	Eigen::MatrixXd trans2_inv = trans2.inverse();
	for(int i = 0; i < trans1_inv.rows() - 1; ++i){
		trans1_inv(i,trans1.cols() - 1) = 0;
		trans2_inv(i,trans1.cols() - 1) = 0;
	}
	trans1(trans1_inv.rows() - 1, trans1.cols() - 1) = 1;
	trans2(trans2_inv.rows() - 1, trans1.cols() - 1) = 1;

	

	Eigen::MatrixXd matrix_uv(5,3),matrix_xy(5,3);
	for(int i = 0; i < 5; ++i){
		matrix_uv(i,0) = uv[i].x;
		matrix_uv(i,1) = uv[i].y;
		matrix_uv(i,2) = 1;
		matrix_xy(i,0) = xy[i].x;
		matrix_xy(i,1) = xy[i].y;
		matrix_xy(i,2) = 1;
	}

	Eigen::MatrixXd trans1_block = trans1.block<3,2>(0,0);
	Eigen::MatrixXd trans2_block = trans2.block<3,2>(0,0);


	double norm1 = (matrix_uv * trans1_block - matrix_xy.block<5,2>(0,0)).norm();
	double norm2 = (matrix_uv * trans2_block - matrix_xy.block<5,2>(0,0)).norm();

	cv::Mat M(2, 3, CV_64F);
	double* m = M.ptr<double>();

	
	if(norm1 <= norm2){
		m[0] = trans1_inv(0,0);
		m[1] = trans1_inv(1,0);
		m[2] = trans1_inv(2,0);
		m[3] = trans1_inv(0,1);
		m[4] = trans1_inv(1,1);
		m[5] = trans1_inv(2,1);
	}
	else{
		m[0] = trans2_inv(0,0);
		m[1] = trans2_inv(1,0);
		m[2] = trans2_inv(2,0);
		m[3] = trans2_inv(0,1);
		m[4] = trans2_inv(1,1);
		m[5] = trans2_inv(2,1);
	}

	return M;
}

void getTripPoints(std::vector<cv::Point2f> &dst_rect, cv::Point2f src_key_point[]){
	cv::Point2f dst_key_point[5];
	dst_key_point[0] = cv::Point2f(0.2, 0.2);
	dst_key_point[1] = cv::Point2f(0.8, 0.2);
	dst_key_point[2] = cv::Point2f(0.5, 0.5);
	dst_key_point[3] = cv::Point2f(0.3, 0.75);
	dst_key_point[4] = cv::Point2f(0.7, 0.75);
	cv::Mat warp_mat = getSimilarityTransform(src_key_point, dst_key_point);

	std::vector<cv::Point2f> src_rect;
	src_rect.push_back(cv::Point2f(0.5, 0.5));
	src_rect.push_back(cv::Point2f(0, 0));
	src_rect.push_back(cv::Point2f(1.0, 0));
	for(int h = 0; h < 3; ++h){
		dst_rect[h].x = src_rect[h].x * warp_mat.ptr<double>(0)[0] 
			+ src_rect[h].y * warp_mat.ptr<double>(0)[1] 
			+ warp_mat.ptr<double>(0)[2];
		dst_rect[h].y = src_rect[h].x * warp_mat.ptr<double>(1)[0]
			+ src_rect[h].y * warp_mat.ptr<double>(1)[1]
			+ warp_mat.ptr<double>(1)[2];
	}
}

RsaFaceDetector::RsaFaceDetector(int gpu_id){
	this->gpu_id_ = gpu_id;
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(this->gpu_id_);

	this->sfn_net.reset(new Net<float>(this->sfn_net_def, TEST));
	this->sfn_net->CopyTrainedLayersFrom(this->sfn_net_weight);

	this->rsa_net.reset(new Net<float>(this->rsa_net_def, TEST));
	this->rsa_net->CopyTrainedLayersFrom(this->rsa_net_weight);

	this->lrn_net.reset(new Net<float>(this->lrn_net_def, TEST));
	this->lrn_net->CopyTrainedLayersFrom(this->lrn_net_weight);
	this->input_layer = sfn_net->input_blobs()[0];
	this->rsa_input_layer = rsa_net->input_blobs()[0];
	this->lrn_input_layer = lrn_net->input_blobs()[0];


	this->anchor_box_len.push_back(ANCHOR_BOX[2] - ANCHOR_BOX[0]);
	this->anchor_box_len.push_back(ANCHOR_BOX[3] - ANCHOR_BOX[1]);
	this->thresh_score = THRESH_SCORE;
	this->stride = STRIDE;
	this->anchor_center = ANCHOR_CENTER;
	for(int i = 5; i >= 1; --i){
		this->scale.push_back(i);
	}
}

void RsaFaceDetector::sfnProcess(const cv::Mat & img){
	int width = img.cols;
	int height = img.rows;
	int depth = img.channels();

	if(width > height){
		this->resize_factor = static_cast<double>(width) / static_cast<double>(MAX_IMG);
		height = static_cast<int>(MAX_IMG / static_cast<float>(width) * height);
		width = MAX_IMG;
	}
	else{
		this->resize_factor = static_cast<double>(height) / static_cast<double>(MAX_IMG);
		width = static_cast<int>(MAX_IMG / static_cast<float>(height) * width);
		height = MAX_IMG;
	}
	cv::Size input_geometry(width, height);

	this->input_layer->Reshape(1, depth, height, width);
	this->sfn_net->Reshape();
	float * input_data = input_layer->mutable_cpu_data();
	
	this->input_channels.clear();
	for(int i = 0; i < this->input_layer->channels(); ++i){
		cv::Mat channel(height, width, CV_32FC1, input_data);
		this->input_channels.push_back(channel);
		input_data += width * height;
	}
	cv::Mat img_resized;
	if(img.size() != input_geometry){
		cv::resize(img, img_resized, input_geometry);
	}
	else{
		img_resized = img;
	}
	cv::Mat img_float;
	img_resized.convertTo(img_float, CV_32FC3);
	cv::Mat mean_mat(input_geometry, CV_32FC3, cv::Scalar(127.0, 127.0, 127.0));
	cv::Mat img_normalized;
	cv::subtract(img_float, mean_mat, img_normalized);
	cv::split(img_normalized, this->input_channels);
	CHECK(reinterpret_cast<float*> (input_channels.at(0).data) 
			== sfn_net->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";

	this->sfn_net->Forward();
	this->sfn_net_output = sfn_net->output_blobs()[0];
}

void RsaFaceDetector::rsaProcess(void){
	trans_featmaps.clear();
	std::shared_ptr<Blob<float> > trans_featmap_ori(new Blob<float>);
	trans_featmap_ori->CopyFrom(*(this->sfn_net_output), false, true);

	trans_featmaps.push_back(trans_featmap_ori);
	int diffcnt;

	std::shared_ptr<Blob<float> > trans_featmap(new Blob<float>);
	std::shared_ptr<Blob<float> >  in_featmap(new Blob<float> );
	for(int i = 1; i < scale.size(); ++i){
		diffcnt = scale[i - 1] - scale[i];
		in_featmap->CopyFrom(*(trans_featmaps[i - 1]), false, true);
		for(int j = 0; j < diffcnt; ++j){
			this->rsa_input_layer->CopyFrom(*in_featmap, false, true);
			rsa_net->Reshape();
			rsa_net->Forward();
			in_featmap->CopyFrom(*rsa_net->output_blobs()[0], false, true);
		}
		trans_featmaps.push_back(std::shared_ptr<Blob<float> >(new Blob<float>));
		trans_featmaps[trans_featmaps.size() - 1]->CopyFrom(*rsa_net->output_blobs()[0], false, true);
	}

}

void RsaFaceDetector::lrnProcess(std::vector<Face> &faces_out){
	std::vector<std::vector<cv::Point2f> > pts_all;
	std::vector<std::vector<double> > rects_all;
	std::vector<float> valid_score_all;
	shared_ptr<Blob<float> > blob_rpn_cls;
	shared_ptr<Blob<float> > blob_rpn_reg;

	for(int i = 0; i < this->trans_featmaps.size(); ++i){
		lrn_input_layer->CopyFrom(*trans_featmaps[i], false, true);
		lrn_net->Reshape();
		lrn_net->Forward();
		blob_rpn_cls = lrn_net->blob_by_name("rpn_cls");
		blob_rpn_reg = lrn_net->blob_by_name("rpn_reg");

		int fmwidth = blob_rpn_cls->shape(3);
		int fmheight = blob_rpn_cls->shape(2);

		std::vector<float> valid_score;
		valid_score.clear();
		std::vector<std::vector<int> > valid_index;
		valid_index.clear();

		for(int x_ = 0; x_ < fmwidth; x_++){
			for(int y_ = 0; y_ < fmheight; y_++){
				if(blob_rpn_cls->data_at(0,0,y_,x_) > this->thresh_score){
					std::vector<int> index(2);
					index[0] = x_;
					index[1] = y_;
					valid_index.push_back(index);
					valid_score.push_back(blob_rpn_cls->data_at(0,0,y_,x_));
					valid_score_all.push_back(blob_rpn_cls->data_at(0,0,y_,x_));
				}	
			}
		}

		std::vector<std::vector<cv::Point2f> > pts_out(valid_index.size(), std::vector<cv::Point2f>(5, cv::Point2f(0,0)));
		std::vector<std::vector<double> > rects(valid_index.size(), std::vector<double>(4,0.0));
		for(int j = 0; j < valid_index.size(); ++j){
			std::vector<float> anchor_center_now(2);
			anchor_center_now[0] = valid_index[j][0]*this->stride + this->anchor_center;
			anchor_center_now[1] = valid_index[j][1]*this->stride + this->anchor_center;
			for(int h = 0; h < 5; h++){
				float anchor_point_now_x = anchor_center_now[0] + *(ANCHOR_PTS + h*2) * anchor_box_len[0];
				float anchor_point_now_y = anchor_center_now[1] + *(ANCHOR_PTS + h*2 + 1) * anchor_box_len[0];
				float pts_delta_x = blob_rpn_reg->data_at(0, 2*h, valid_index[j][1], valid_index[j][0]) *
					anchor_box_len[0];
				float pts_delta_y = blob_rpn_reg->data_at(0, 2*h+1, valid_index[j][1], valid_index[j][0]) *
					anchor_box_len[0];
				pts_out[j][h].x = pts_delta_x + anchor_point_now_x;
				pts_out[j][h].y = pts_delta_y + anchor_point_now_y;
			}

			std::vector<cv::Point2f> dst_rect(3, cv::Point2f(0, 0));
			cv::Point2f srcFivePoints[5];
			for(int h = 0; h < 5; ++h){
				srcFivePoints[h] = pts_out[j][h];
			}
			getTripPoints(dst_rect, srcFivePoints);
			double scale_double = pow(2, this->scale[i]-5);
			double rect_width = sqrt(pow((dst_rect[1].x - dst_rect[2].x), 2) 
					+ pow((dst_rect[1].y - dst_rect[2].y), 2));
			rects[j][0] = round((dst_rect[0].x - rect_width/2) / scale_double * resize_factor);
			rects[j][1] = round((dst_rect[0].y - rect_width/2) / scale_double * resize_factor);
			rects[j][2] = round((dst_rect[0].x + rect_width/2) / scale_double * resize_factor);
			rects[j][3] = round((dst_rect[0].y + rect_width/2) / scale_double * resize_factor);
			rects_all.push_back(rects[j]);
			for(int h = 0; h < 5; h++){
				pts_out[j][h].x = round(pts_out[j][h].x / scale_double * resize_factor);
				pts_out[j][h].y = round(pts_out[j][h].y / scale_double * resize_factor);
			}
			pts_all.push_back(pts_out[j]);
		}
	}

	float *boxes = new float[pts_all.size()*5];
	int *keep = new int[pts_all.size()*5];
	std::vector<Face> faces;

	for(int i = 0; i < pts_all.size(); ++i){
		Face face;
		face.bbox = rects_all[i];
		face.key_points = pts_all[i];
		face.score = valid_score_all[i];
		faces.push_back(face);
	}
	std::sort(faces.begin(), faces.end(), comp);
	for(int i = 0; i < faces.size(); ++i){
		boxes[i*5+0] = faces[i].bbox[0];
		boxes[i*5+1] = faces[i].bbox[1];
		boxes[i*5+2] = faces[i].bbox[2];
		boxes[i*5+3] = faces[i].bbox[3];
		boxes[i*5+4] = faces[i].score;
	}
	int num_out;
	_nms(keep, &num_out, boxes, faces.size(), 5, NMS_THRESH, GPU_ID);
	for(int i = 0; i < num_out; ++i){
		faces_out.push_back(faces[*(keep+i)]);
	}
	delete [] boxes;
	delete [] keep;
}

std::vector<Face> RsaFaceDetector::detect(cv::Mat image){
	this->sfnProcess(image);
	this->rsaProcess();
	std::vector<Face> out;
	lrnProcess(out);

	return out;
}
