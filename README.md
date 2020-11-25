# PU-GAN: a Point Cloud Upsampling Adversarial Network
by [Ruihui Li](https://liruihui.github.io/), [Xianzhi Li](https://nini-lxz.github.io/), [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), [Daniel Cohen-Or](https://www.cs.tau.ac.il/~dcor/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction 

This repository is for our ICCV 2019 paper '[PU-GAN: a Point Cloud Upsampling Adversarial Network](https://liruihui.github.io/publication/PU-GAN/)'. The code is modified from [3PU](https://github.com/yifita/3PU) and [PU-Net](https://github.com/yulequan/PU-Net). 

### Docker
A Dockerfile is provided to help you relief the pain of configurate training environment. 

See the instructions in [here](./Docker).

### Installation
This repository is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators. 

For installing tensorflow, please follow the official instructions in [here](https://www.tensorflow.org/install/install_linux). The code is tested under TF1.11 (higher version should also work) and Python 3.6 on Ubuntu 16.04.

For compiling TF operators, please check `tf_xxx_compile.sh` under each op subfolder in `code/tf_ops` folder. Note that you need to update `nvcc`, `python` and `tensoflow include library` if necessary. 

### Note
When running the code, if you have `undefined symbol: _ZTIN10tensorflow8OpKernelE` error, you need to compile the TF operators. If you have already added the `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` but still have ` cannot find -ltensorflow_framework` error. Please use 'locate tensorflow_framework
' to locate the tensorflow_framework library and make sure this path is in `$TF_LIB`.

### Usage

1. Clone the repository:

   ```shell
   https://github.com/liruihui/PU-GAN.git
   cd PU-GAN
   ```
   
2. Compile the TF operators
   Follow the above information to compile the TF operators. 
   
3. Train the model:
    First, you need to download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/open?id=13ZFDffOod_neuF3sOM0YiqNbIJEeSKdZ) and put it in folder `data/train`.
    Then run:
   ```shell
   cd code
   python pu_gan.py --phase train
   ```

4. Evaluate the model:
    First, you need to download the pretrained model from [GoogleDrive](https://drive.google.com/open?id=12kWoB0-_tflq65RNpJEnNGTTwPXa6IOH), extract it and put it in folder 'model'.
    Then run:
   ```shell
   cd code
   python pu_gan.py --phase test
   ```
   You will see the input and output results in the folder `data/test/output`.
   
5. The training and testing mesh files can be downloaded from [GoogleDrive](https://drive.google.com/open?id=1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC).

### Evaluation code
We provide the code to calculate the uniform metric in the evaluation code folder. In order to use it, you need to install the CGAL library. Please refer [this link](https://www.cgal.org/download/linux.html) and  [PU-Net](https://github.com/yulequan/PU-Net) to install this library.
Then:
   ```shell
   cd evaluation_code
   cmake .
   make
   ./evaluation Icosahedron.off Icosahedron.xyz
   ```
The second argument is the mesh, and the third one is the predicted points.

## Citation

If PU-GAN is useful for your research, please consider citing:

    @inproceedings{li2019pugan,
         title={PU-GAN: a Point Cloud Upsampling Adversarial Network},
         author={Li, Ruihui and Li, Xianzhi and Fu, Chi-Wing and Cohen-Or, Daniel and Heng, Pheng-Ann},
         booktitle = {{IEEE} International Conference on Computer Vision ({ICCV})},
         year = {2019}
     }


### Questions

Please contact 'lirh@cse.cuhk.edu.hk'

