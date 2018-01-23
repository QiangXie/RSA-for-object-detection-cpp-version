PROJECT := rsa_face_detection

CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_61,code=compute_61
ORIGIN := \$$ORIGIN

RELEASE_BUILD_DIR := .build_release
BUILD_DIR := $(RELEASE_BUILD_DIR)
BUILD_DIR_LINK := build

THIRD_PARTY_DIR := third_party
SRC_DIR := src
MAIN_SRC_DIR := $(SRC_DIR)/main
UTIL_SRC_DIR := $(SRC_DIR)/util
INCLUDE_DIR := include
LIB_DIR := lib

CUDA_DIR := /usr/local/cuda-8.0
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include

NVCC_FLAGS := -G 


MAIN_CXX_SRCS := $(shell find $(MAIN_SRC_DIR) -name "*.cpp")
UTIL_CXX_SRCS := $(shell find $(UTIL_SRC_DIR) -name "*.cpp")
UTIL_CU_SRCS := $(shell find src/util -name "*.cu")
MAIN_CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${MAIN_CXX_SRCS:.cpp=.o})
UTIL_CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${UTIL_CXX_SRCS:.cpp=.o})
CXX_OBJS := $(MAIN_CXX_OBJS) $(UTIL_CXX_OBJS)
CU_OBJS := $(addprefix $(BUILD_DIR)/cuda/, ${UTIL_CU_SRCS:.cu=.o})
OBJS := $(CXX_OBJS) $(CU_OBJS)
DEPS := ${OBJS:.o=.d}

EXECUTABLE_PROGRAM := ${MAIN_CXX_OBJS:.o=.bin}

ALL_BUILD_DIRS := $(sort $(BUILD_DIR) $(BUILD_DIR)/cuda/src/util $(addprefix $(BUILD_DIR)/, $(UTIL_SRC_DIR)) \
	$(addprefix $(BUILD_DIR)/, $(MAIN_SRC_DIR)))

CAFFE_DIR := $(THIRD_PARTY_DIR)/CaffeMex_v2
CAFFE_INCLUDE_DIR := $(CAFFE_DIR)/include
CAFFE_PB_DIR := $(CAFFE_DIR)/build/src
CAFFE_LIB_DIR := $(CAFFE_DIR)/build/lib

EIGEN_INCLUDE_DIR := /usr/include/eigen3
OPENCV_INCLUDE_DIR := /usr/local/include
OPENCV_LIB_DIR := /usr/local/lib

GLOG_LIB_DIR := /usr/lib/x86_64-linux-gnu 

CXX := /usr/bin/g++

GDB_FLAG := -g 

INCLUDE_DIRS := $(CAFFE_INCLUDE_DIR) $(INCLUDE_DIR) $(CUDA_INCLUDE_DIR) $(CAFFE_PB_DIR) $(OPENCV_INCLUDE_DIR) $(EIGEN_INCLUDE_DIR)
LIBRARY_DIRS := $(CAFFE_LIB_DIR) $(LIB_DIR) $(OPENCV_LIB_DIR) $(GLOG_LIB_DIR) $(CUDA_DIR)/lib64
LIBRARIES := caffe boost_system opencv_core opencv_imgproc opencv_highgui glog cudart cublas curand

WARNINGS := -Wall -Wno-sign-compare -Wno-uninitialized
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)
CXXFLAGS := -std=c++11 $(GDB_FLAG) -MMD -MP -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)  
LDFLAGS := $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(foreach library,$(LIBRARIES),-l$(library))
NVCCFLAGS += -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)

.PHONY: all clean

all: bin

bin: $(EXECUTABLE_PROGRAM)

$(BUILD_DIR_LINK): $(BUILD_DIR)/.linked

$(BUILD_DIR)/.linked:
	@ mkdir -p $(BUILD_DIR)
	@ $(RM) -r $(BUILD_DIR_LINK)
	@ ln -s $(BUILD_DIR) $(BUILD_DIR_LINK)
	@ touch $@

$(ALL_BUILD_DIRS): | $(BUILD_DIR_LINK)
	@ mkdir -p $@

$(BUILD_DIR)/%.o: %.cpp | $(ALL_BUILD_DIRS)
	@ echo CXX $@
	@ $(CXX) $< $(CXXFLAGS) -c -o $@

$(EXECUTABLE_PROGRAM): %.bin : %.o $(OBJS)
	@ echo CXX/LD -o $@
	@ $(CXX)  $(OBJS) -o $@ $(LINKFLAGS) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../../../$(CAFFE_LIB_DIR) -Wl,-rpath,/usr/local/cuda-8.0/lib64
	@ cd ./bin ; rm -rf demo ; ln -s ../$(EXECUTABLE_PROGRAM) demo

$(BUILD_DIR)/cuda/%.o: %.cu | $(ALL_BUILD_DIRS)
	@ echo NVCC $<
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} -odir $(@D)
	@ $(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 


clean:
	@ rm -rf $(BUILD_DIR)
	@ rm -rf $(BUILD_DIR_LINK)
	@ rm ./bin/demo

-include $(DEPS)
