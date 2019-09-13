#!/usr/bin/env bash
#/bin/bash
/usr/local/cuda-10.1/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /data/lirh/anaconda3/envs/tensorflow3/lib/python3.6/site-packages/tensorflow/include  -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/lirh/anaconda3/envs/tensorflow11/lib/python3.6/site-packages/tensorflow/include  -I /usr/local/cuda-8.0/include -I /home/lirh/anaconda11/envs/tensorflow3/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/home/lirh/anaconda3/envs/tensorflow11/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
