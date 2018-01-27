# RSA for object detection c++ version

![](result.jpg) 

C++ version codebase for *Recurrent Scale Approximation for Object Detection in CNN* published at **ICCV 2017**, [[arXiv]](https://arxiv.org/abs/1707.09531). The codebase was written according to the [[matlab version code]](https://github.com/sciencefans/RSA-for-object-detection). There is no training code provided in matlab version project，so, here we just offer test code.

## How to use

Ensure opencv,Eigen and cuda have been installed.

####1.clone cpp-rsa-net

    git clone https://github.com/QiangXie/RSA-for-object-detection-cpp-version.git


####2.make some dirs

    mkdir bin && mkdir third_party


####3.install and compile CaffeMex_v2 with matlab interface

    cd third_party
    git clone https://github.com/sciencefans/CaffeMex_v2.git
    cp Makefile.config.example Makefile.config 
    make all -j32

####4.modify Makefile to customize your own config, and compile RSA for object detection c++ version code

    cd RSA-for-object-detection-cpp-version
    make

####5.run demo

    ./bin/demo


## About speed
190ms per image with TitanX，compare with Matlab version has a wide gap，still in progress.

