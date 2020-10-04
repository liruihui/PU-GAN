
cd approxmatch
echo 'approxmatch...'
bash tf_approxmatch_compile.sh
cd ../grouping
echo 'grouping...'
bash tf_grouping_compile.sh
cd ../interpolation
echo 'interpolation...'
bash tf_interpolate_compile.sh
cd ../nn_distance
echo 'nn_distance...'
bash tf_nndistance_compile.sh
cd ../sampling
echo 'sampling...'
bash tf_sampling_compile.sh
