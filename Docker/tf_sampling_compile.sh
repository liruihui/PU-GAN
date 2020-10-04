#!/usr/bin/env bash
#/bin/bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /data/lirh/anaconda3/envs/tensorflow3/lib/python3.6/site-packages/tensorflow/include  -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -O2 -I /usr/local/cuda-9.0/include -L/usr/local/cuda-9.0/lib64 ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -lcudart -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0