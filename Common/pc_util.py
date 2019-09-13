""" Utility functions for processing point clouds.

Author: Charles R. Qi, Hao Su
Date: November 2016
"""

import os
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from open3d import *
import open3d
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Draw point cloud
from Common.eulerangles import euler2mat

# Point cloud IO
import numpy as np
import plyfile

from sklearn.neighbors import NearestNeighbors


# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b,:,:]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    return vol

#a = np.zeros((16,1024,3))
#print point_cloud_to_volume_batch(a, 12, 1.0, False).shape

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points


def extract_knn_patch(queries, pc, k):
    """
    queries [M, C]
    pc [P, C]
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    knn_idx = knn_search.kneighbors(queries, return_distance=False)
    k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    return k_patches

def get_knn_idx(queries, pc, k):
    """
    queries [M, C]
    pc [P, C]
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    knn_idx = knn_search.kneighbors(queries, return_distance=False)
    return knn_idx

def get_pairwise_distance(batch_features):
    """Compute pairwise distance of a point cloud.

    Args:
      batch_features: numpy (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    og_batch_size = len(batch_features.shape)

    if og_batch_size == 2: #just two dimension
        batch_features = np.expand_dims(batch_features, axis=0)


    batch_features_transpose = np.transpose(batch_features, (0, 2, 1))

    #batch_features_inner = batch_features@batch_features_transpose
    batch_features_inner = np.matmul(batch_features,batch_features_transpose)

    #print(np.max(batch_features_inner), np.min(batch_features_inner))


    batch_features_inner = -2 * batch_features_inner
    batch_features_square = np.sum(np.square(batch_features), axis=-1, keepdims=True)


    batch_features_square_tranpose = np.transpose(batch_features_square, (0, 2, 1))

    return batch_features_square + batch_features_inner + batch_features_square_tranpose

def get_knn_dis(queries, pc, k):
    """
    queries [M, C]
    pc [P, C]
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    dis,knn_idx = knn_search.kneighbors(queries, return_distance=True)
    #k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    return dis

def normalize_point_cloud(input):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
    input = input / furthest_distance
    return input, centroid, furthest_distance

def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02, is_2D=False):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    if is_2D:
        chn = 2
    else:
        chn = 3
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data[:, :, chn:] = 0
    jittered_data += batch_data
    return jittered_data

def downsample_points(pts, K):
    # if num_pts > 8K use farthest sampling
    # else use random sampling
    if pts.shape[0] >= 2*K:
        sampler = FarthestSampler()
        return sampler(pts, K)
    else:
        return pts[np.random.choice(pts.shape[0], K,
            replace=(K<pts.shape[0])), :]


class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        farthest_pts = np.zeros((k, 3), dtype=np.float32)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts

# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(
                3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir,
                       c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim3d(xlim)
            ax.set_ylim3d(ylim)
            ax.set_zlim3d(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                      xlim=(-0.55, 0.55), ylim=(-0.55, 0.55), zlim=(-0.55, 0.55)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 3))

    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        # color = pcd[:, 0]
        color = 'k'
        ax = fig.add_subplot(
            1, len(pcds), j + 1, projection='3d')
        ax.view_init(0, 270)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir,
                   c=color, s=size, vmin=-1, vmax=0.5)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim3d(xlim)
        ax.set_ylim3d(ylim)
        ax.set_zlim3d(zlim)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def read_pcd(filename, count=None):
    points = read_point_cloud(filename)
    points = np.array(points.points).astype(np.float32)
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = downsample_points(points, count)
    return points


def save_pcd(filename, points):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    write_point_cloud(filename, pcd)


def read_ply_with_color(file, count=None):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'], loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'], loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)
    colors = None
    if 'red' in loaded['vertex'].data.dtype.names:
        colors = np.vstack([loaded['vertex'].data['red'], loaded['vertex'].data['green'], loaded['vertex'].data['blue']])
        if 'alpha' in loaded['vertex'].data.dtype.names:
            colors = np.concatenate([colors, np.expand_dims(loaded['vertex'].data['alpha'], axis=0)], axis=0)
        colors = colors.transpose(1, 0)
        colors = colors.astype(np.float32) / 255.0

    points = points.transpose(1, 0)
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = downsample_points(points, count)
    return points, colors


def read_ply(file, count=None):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'], loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'], loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)

    points = points.transpose(1, 0)
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = downsample_points(points, count)
    return points


def save_ply_with_face_property(points, faces, property, property_max, filename, cmap_name="Set1"):
    face_num = faces.shape[0]
    colors = np.full(faces.shape, 0.5)
    cmap = cm.get_cmap(cmap_name)
    for point_idx in range(face_num):
        colors[point_idx] = cmap(property[point_idx] / property_max)[:3]
    save_ply_with_face(points, faces, filename, colors)


def save_ply_with_face(points, faces, filename, colors=None):
    vertex = np.array([tuple(p) for p in points], dtype=[
                      ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces = np.array([(tuple(p),) for p in faces], dtype=[
                     ('vertex_indices', 'i4', (3, ))])
    descr = faces.dtype.descr
    if colors is not None:
        assert len(colors) == len(faces)
        face_colors = np.array([tuple(c * 255) for c in colors],
                               dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        descr = faces.dtype.descr + face_colors.dtype.descr

    faces_all = np.empty(len(faces), dtype=descr)
    for prop in faces.dtype.names:
        faces_all[prop] = faces[prop]
    if colors is not None:
        for prop in face_colors.dtype.names:
            faces_all[prop] = face_colors[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(
        vertex, 'vertex'), plyfile.PlyElement.describe(faces_all, 'face')], text=False)
    ply.write(filename)


def load(filename, count=None):
    if filename[-4:] == ".ply":
        points = read_ply(filename, count)[:, :3]
    elif filename[-4:] == ".pcd":
        points = read_pcd(filename, count)[:, :3]
    else:
        points = np.loadtxt(filename).astype(np.float32)
        if count is not None:
            if count > points.shape[0]:
                # fill the point clouds with the random point
                tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
                tmp[:points.shape[0], ...] = points
                tmp[points.shape[0]:, ...] = points[np.random.choice(
                    points.shape[0], count - points.shape[0]), :]
                points = tmp
            elif count < points.shape[0]:
                # different to pointnet2, take random x point instead of the first
                # idx = np.random.permutation(count)
                # points = points[idx, :]
                points = downsample_points(points, count)
    return points

def save_ply(points, filename, colors=None, normals=None):
    vertex = np.core.records.fromarrays(points.transpose(1,0),names='x, y, z',formats='f4, f4, f4')
    num_vertex = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.core.records.fromarrays(normals.transpose(1,0),names='nx, ny, nz',formats='f4, f4, f4')
        assert len(vertex_normal) == num_vertex
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        assert len(colors) == num_vertex
        if colors.max() <= 1:
            colors = colors*255
        if colors.shape[1] == 4:
            vertex_color = np.core.records.fromarrays(colors.transpose(1,0),names='red, green, blue, alpha',formats='u1, u1, u1, u1')
        else:
            vertex_color = np.core.records.fromarrays(colors.transpose(1,0),names='red, green, blue',formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(num_vertex, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_property(points, property, filename, property_max=None, normals=None, cmap_name='Set1'):
    point_num = points.shape[0]
    colors = np.full([point_num, 3], 0.5)
    cmap = cm.get_cmap(cmap_name)
    if property_max is None:
        property_max = np.amax(property)
    for point_idx in range(point_num):
        colors[point_idx] = cmap(property[point_idx] / property_max)[:3]
    save_ply(points, filename, colors, normals)


# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def draw_point_cloud(input_points, canvasSize=500, space=240, diameter=10,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0,1,2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    canvasSizeX = canvasSize
    canvasSizeY = canvasSize

    image = np.zeros((canvasSizeX, canvasSizeY))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter-1)/2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
                disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSizeX/2 + (x*space)
        yc = canvasSizeY/2 + (y*space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc
        #image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
        image[px, py] = image[px, py] * 0.7 + dv * 0.3

    val = np.max(image)+1e-8
    val = np.percentile(image,99.9)
    image = image / val
    mask = image==0

    image[image>1.0]=1.0
    image = 1.0-image
    #image = np.expand_dims(image, axis=-1)
    #image = np.concatenate((image*0.3+0.7,np.ones_like(image), np.ones_like(image)), axis=2)
    #image = colors.hsv_to_rgb(image)
    image[mask]=1.0


    return image

def point_cloud_three_views(points,diameter=5):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    # img1 = draw_point_cloud(points, xrot=90/180.0*np.pi,  yrot=0/180.0*np.pi, zrot=0/180.0*np.pi,diameter=diameter)
    # img2 = draw_point_cloud(points, xrot=180/180.0*np.pi, yrot=0/180.0*np.pi, zrot=0/180.0*np.pi,diameter=diameter)
    # img3 = draw_point_cloud(points, xrot=0/180.0*np.pi,  yrot=-90/180.0*np.pi, zrot=0/180.0*np.pi,diameter=diameter)
    # image_large = np.concatenate([img1, img2, img3], 1)
    try:
        img1 = draw_point_cloud(points, zrot=110 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,diameter=diameter)
        img2 = draw_point_cloud(points, zrot=70 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,diameter=diameter)
        img3 = draw_point_cloud(points, zrot=180.0 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,diameter=diameter)
        image_large = np.concatenate([img1, img2, img3], 1)
    except Exception as e:
        image_large = np.zeros((500, 1500), dtype=np.float32)

    return image_large


from PIL import Image
def point_cloud_three_views_demo():
    """ Demo for draw_point_cloud function """
    points = read_ply('../third_party/mesh_sampling/piano.ply')
    im_array = point_cloud_three_views(points)
    img = Image.fromarray(np.uint8(im_array*255.0))
    img.save('piano.jpg')


def pyplot_draw_point_cloud(points, output_filename=None):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if output_filename:
        savefig(output_filename)

def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)


if __name__=="__main__":
    point_cloud_three_views_demo()
