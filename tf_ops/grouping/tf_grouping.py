import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so.so'))
def query_ball_point(radius, nsample, xyz, new_xyz):
    '''
    Input:
        nsample: int32, number of points selected in each ball region
        xyz: (batch_size, ndataset, 3) float32 array, input points
        new_xyz: (batch_size, npoint, 3) float32 array, query points
        radius: (batch_size), ball search radius
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    # return grouping_module.query_ball_point(radius, nsample, xyz1, xyz2)
    if np.isscalar(radius):
        radius = tf.expand_dims(tf.constant(radius), axis=0)
        batch_size = xyz.get_shape()[0].value
        radius = tf.tile(radius, [batch_size])
        idx, pts_cnt = grouping_module.query_ball_point(
            xyz, new_xyz, radius, nsample)
    else:
        idx, pts_cnt = grouping_module.query_ball_point(
            xyz, new_xyz, radius, nsample)
    return idx, pts_cnt

ops.NoGradient('QueryBallPoint')
def select_top_k(k, dist):
    '''
    Input:
        k: int32, number of k SMALLEST elements selected
        dist: (b,m,n) float32 array, distance matrix, m query points, n dataset points
    Output:
        idx: (b,m,n) int32 array, first k in n are indices to the top k
        dist_out: (b,m,n) float32 array, first k in n are the top k
    '''
    return grouping_module.selection_sort(dist, k)
ops.NoGradient('SelectionSort')
def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return grouping_module.group_point(points, idx)
@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]

# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keep_dims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keep_dims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D

# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_cross_matrix_general(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keep_dims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keep_dims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D

# A shape is (N, P, C)
def find_duplicate_columns(A):
    N = A.shape[0]
    P = A.shape[1]
    indices_duplicated = np.ones((N, 1, P), dtype=np.int32)
    for idx in range(N):
        _, indices = np.unique(A[idx], return_index=True, axis=0)
        indices_duplicated[idx, :, indices] = 0
    return indices_duplicated


# add a big value to duplicate columns
def prepare_for_unique_top_k(D, A):
    indices_duplicated = tf.py_func(find_duplicate_columns, [A], tf.int32)
    D += tf.reduce_max(D)*tf.cast(indices_duplicated, tf.float32)


# return shape is (N, P, K, 2)
def knn_point_2(k, points, queries, sort=True, unique=True):
    """
    points: dataset points (N, P0, K)
    queries: query points (N, P, K)
    return indices is (N, P, K, 2) used for tf.gather_nd(points, indices)
    distances (N, P, K)
    """
    #import pdb
    #pdb.set_trace()
    with tf.name_scope("knn_point"):
        batch_size = tf.shape(queries)[0]
        point_num = tf.shape(queries)[1]

        D = batch_distance_matrix_general(queries, points)
        if unique:
            prepare_for_unique_top_k(D, points)
        distances, point_indices = tf.nn.top_k(-D, k=k, sorted=sort)  # (N, P, K)
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
        return -distances, indices

def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    # b = xyz1.get_shape()[0].value
    # n = xyz1.get_shape()[1].value
    # c = xyz1.get_shape()[2].value
    # m = xyz2.get_shape()[1].value
    # xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    # xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    xyz1 = tf.expand_dims(xyz1,axis=1)
    xyz2 = tf.expand_dims(xyz2,axis=2)
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    #dist = tf.sqrt(tf.abs(dist))
    # outi, out = select_top_k(k, dist)
    # idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    # val = tf.slice(out, [0,0,0], [-1,-1,k])

    val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx

if __name__=='__main__':
    knn=True
    import numpy as np
    import time
    np.random.seed(100)
    pts = np.random.random((32,512,64)).astype('float32')
    tmp1 = np.random.random((32,512,3)).astype('float32')
    tmp2 = np.random.random((32,128,3)).astype('float32')
    with tf.device('/gpu:1'):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        radius = 0.1 
        nsample = 64
        if knn:
            _, idx = knn_point(nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)
        else:
            idx, _ = query_ball_point(radius, nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)
            #grouped_points_grad = tf.ones_like(grouped_points)
            #points_grad = tf.gradients(grouped_points, points, grouped_points_grad)
    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(100):
            ret = sess.run(grouped_points)
        print(time.time() - now)
        print(ret.shape, ret.dtype)
        print(ret)
    
    
