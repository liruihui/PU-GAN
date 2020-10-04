
## Step to Training

```bash
git clone https://github.com/liruihui/PU-GAN.git
cd PU-GAN/Docker
docker build -t tensorflow/pu-gan -f Dockerflie .
```

### Replace shell scripts under this directory to original ones
```bash
cp tf_approxmatch_compile.sh ../tf_ops/approxmatch/tf_approxmatch_compile.sh
cp tf_grouping_compile.sh ../tf_ops/grouping/tf_grouping_compile.sh
cp tf_interpolate_compile.sh ../tf_ops/interpolation/tf_interpolate_compile.sh
cp tf_nndistance_compile.sh ../tf_ops/nn_distance/tf_nndistance_compile.sh
cp tf_sampling_compile.sh ../tf_ops/sampling/tf_sampling_compile.sh

cp compile.sh ../tf_ops/compile.sh
```

### Start training
```bash
cd ../
docker run -it --rm \
    -e DISPLAY=unix$DISPLAY \
    -v $(pwd):/workspace/ \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /workspace \
    --name pu-gan-runtime \
    --gpus all \
    --shm-size 8G \
    tensorflow/pu-gan

cd tf_ops
bash compile.sh
cd ../
python pu_gan.py --phase train
```
