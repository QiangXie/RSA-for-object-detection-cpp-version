#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_
#include <string>

const std::string SFN_NET_DEF = "../predict/model/res_pool2/test.prototxt";
const std::string SFN_NET_WEIGHT = "../predict/output/ResNet_3b_s16/tot_wometa_1epoch";
const std::string RSA_NET_DEF = "../predict/model/ResNet_3b_s16_fm2fm_pool2_deep/test.prototxt";
const std::string RSA_NET_WEIGHT = "../predict/output/ResNet_3b_s16_fm2fm_pool2_deep/65w";
const std::string LRN_NET_DEF = "../predict/model/ResNet_3b_s16_f2r/test.prototxt";
const std::string LRN_NET_WEIGHT = "../predict/output/ResNet_3b_s16/tot_wometa_1epoch";
const float ANCHOR_BOX[] = {-44.754833995939045, -44.754833995939045, 44.754833995939045, 44.754833995939045};
const float ANCHOR_PTS[] = {-0.1719448, -0.2204161, 0.1719447, -0.2261145, -0.0017059, -0.0047045, -0.1408936, 0.2034478, 0.1408936, 0.1977626};
//?Why define two thresh value
const float NMS_THRESH = 0.6;
const float THRESH_CLS = 3.0;
const float THRESH_SCORE = 8.0;
const float ANCHOR_CENTER = 7.5;
const float STRIDE = 16.0;
const int GPU_ID = 3;
const int MAX_IMG = 2048;
const int MIN_IMG = 64;




#endif
