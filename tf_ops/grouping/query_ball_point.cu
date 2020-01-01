#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include <string>
#include <vector>
#include "cuPrintf.cuh"
#include "cuPrintf.cu"

using namespace std;
using namespace std;
float randomf(){
    return (rand()+0.5)/(RAND_MAX+1.0);
}
static double get_time(){
    timespec tp;
    clock_gettime(CLOCK_MONOTONIC,&tp);
    return tp.tv_sec+tp.tv_nsec*1e-9;
}
// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample)
__global__ void query_ball_point_gpu(int b, int n, int m, const float* radius, int nsample, const float *xyz1, const float *xyz2, int *idx) {
    for (int i=0;i<b;++i) {
        for (int j=0;j<m;++j) {
            int cnt = 0;
            for (int k=0;k<n;++k) {
                if (cnt == nsample)
                    break; // only pick the FIRST nsample points in the ball
	            float x2=xyz2[j*3+0];
	            float y2=xyz2[j*3+1];
	            float z2=xyz2[j*3+2];
	            float x1=xyz1[k*3+0];
	            float y1=xyz1[k*3+1];
	            float z1=xyz1[k*3+2];
		        float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
                if (d<radius[0]) {
                    if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                        for (int l=0;l<nsample;++l)
                            idx[j*nsample+l] = k;
                    }
                    idx[j*nsample+cnt] = k;
                    cnt+=1;
                }
            }
        }
        xyz1+=n*3;
        xyz2+=m*3;
        idx+=m*nsample;
    }
}


// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    for (int i=0;i<b;++i) {
        for (int j=0;j<m;++j) {
            for (int k=0;k<nsample;++k) {
                int ii = idx[j*nsample+k];
                for (int l=0;l<c;++l) {
                    out[j*nsample*c+k*c+l] = points[ii*c+l];
                }
            }
        }
        points+=n*c;
        idx+=m*nsample;
        out+=m*nsample*c;
    }
}

// input: grad_out (b,m,nsample,c), idx (b,m,nsample), 
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
    for (int i=0;i<b;++i) {
        for (int j=0;j<m;++j) {
            for (int k=0;k<nsample;++k) {
                int ii = idx[j*nsample+k];
                for (int l=0;l<c;++l) {
                     grad_points[ii*c+l] += grad_out[j*nsample*c+k*c+l];
                }
            }
        }
        idx+=m*nsample;
        grad_out+=m*nsample*c;
        grad_points+=n*c;
    }
}

