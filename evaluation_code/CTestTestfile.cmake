# CMake generated Testfile for 
# Source directory: /home/lirh/pointcloud/GAN/PointGAN/PU-GAN/evaluation_code
# Build directory: /home/lirh/pointcloud/GAN/PointGAN/PU-GAN/evaluation_code
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(compilation_of__evaluation "/usr/local/bin/cmake" "--build" "/home/lirh/pointcloud/GAN/PointGAN/PU-GAN/evaluation_code" "--target" "evaluation")
set_tests_properties(compilation_of__evaluation PROPERTIES  FIXTURES_SETUP "evaluation" LABELS "Distance_2_Tests")
add_test(execution___of__evaluation "/home/lirh/pointcloud/GAN/PointGAN/PU-GAN/evaluation_code/evaluation")
set_tests_properties(execution___of__evaluation PROPERTIES  DEPENDS "compilation_of__evaluation" FIXTURES_REQUIRED "Distance_2_Tests;evaluation" LABELS "Distance_2_Tests" WORKING_DIRECTORY "/home/lirh/pointcloud/GAN/PointGAN/PU-GAN/evaluation_code/__exec_test_dir")
add_test(Distance_2_Tests_SetupFixture "/usr/local/bin/cmake" "-E" "copy_directory" "/home/lirh/pointcloud/GAN/PointGAN/PU-GAN/evaluation_code" "/home/lirh/pointcloud/GAN/PointGAN/PU-GAN/evaluation_code/__exec_test_dir")
set_tests_properties(Distance_2_Tests_SetupFixture PROPERTIES  FIXTURES_SETUP "Distance_2_Tests" LABELS "Distance_2_Tests")
add_test(Distance_2_Tests_CleanupFixture "/usr/local/bin/cmake" "-E" "remove_directory" "/home/lirh/pointcloud/GAN/PointGAN/PU-GAN/evaluation_code/__exec_test_dir")
set_tests_properties(Distance_2_Tests_CleanupFixture PROPERTIES  FIXTURES_CLEANUP "Distance_2_Tests" LABELS "Distance_2_Tests")
