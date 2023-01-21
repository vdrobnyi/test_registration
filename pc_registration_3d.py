from ctypes.wintypes import PPOINT
import open3d as o3d
import numpy as np
import laspy

# import PCLKeypoint

from tqdm import tqdm
from sklearn.decomposition import PCA
import torch
from matplotlib import cm
import copy

from scipy.spatial import KDTree


axes = o3d.geometry.TriangleMesh.create_coordinate_frame()


# Visualise NumPy (xyz, normals)
def draw_sample(points, normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    o3d.visualization.draw_geometries([pcd, axes])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


# This function is only used to make the keypoints look better on the rendering
def keypoints_to_spheres(keypoints, color=[1.0, 0.75, 0.0]):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color(color)
    return spheres


# Reads las file, returns Open3D PointCloud
def read_las(file):
    with laspy.open(file) as fh:
        las = fh.read()
    
    xyz = las.xyz
    rgb = np.stack([las.red, las.green, las.blue]).transpose() / 65535
    
    # # Read semantics
    # semantic_id = np.array(las.semantic_id, dtype=int)
    # # One hot encoding
    # semantic_oh = np.eye(1 + max(semantic_id))[semantic_id]
    
    # # Move to origin
    # xyz_mean = las.xyz.mean(axis=0)
    
    # # Z - axis collinear with gravity 
    # # assumes that PC varies less along gravity direction
    # xyz = PCA().fit_transform(las.xyz)

    # Downsample
    reduce_factor = 5
    ind = np.array(list(range(0, len(las.xyz), reduce_factor)))

    # semantic_oh = semantic_oh[ind]

    # Create Open3D object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[ind])
    pcd.colors = o3d.utility.Vector3dVector(rgb[ind])
    pcd.estimate_normals()

    return pcd


def scale_pcd(pcd, scale):
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * scale)
    return pcd


# Samples PointClouds of radius 'radius' around 'query_points'
def sample_pcds(pcd, query_points, radius=1):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    tree = KDTree(points)
    ind = tree.query_ball_point(query_points, r=radius)

    out_pcds = []
    out_kps = []

    for j, i in enumerate(ind):
        # Ignore points that doesn't have many neighbours
        if len(i) > 300:
            out_kps.append(j)
            # Normalize sampled point cloud
            p = points[i] - query_points[j]
            p = p / np.abs(p).max(axis=0)
            # Downsample if necessary
            index = np.random.choice(p.shape[0], 1000, replace=True)
            # (xyz, normals, index from original point cloud)
            out_pcds.append((p[index], normals[i][index], i[index]))
            
    return out_pcds, out_kps


def inference(encoder, pcd, normals):
    p = torch.tensor(np.concatenate([pcd, normals], axis=1)).unsqueeze(0).float()
    points_encoded, features = encoder.net(p)
    return points_encoded, features


# Visualise input and output of point encoder. Should ireally be similar
def check_encoder(pcd, keypoints):
    from point_encoder import PointEncoder
    encoder = PointEncoder("net_4.ptr", 64)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    tree = KDTree(points)
    ind = tree.query_ball_point(np.asarray(keypoints.points), r=2)
    kpn = np.asarray(keypoints.points)

    encoder.net.eval()

    for j, i in enumerate(ind):
        if len(i) > 300:
            p = points[i] - kpn[j]
            p = p / np.abs(p).max(axis=0)
            index = np.random.choice(p.shape[0], 1000, replace=True)
            draw_sample(p[index], normals[i][index])
            
            points_encoded, features = inference(encoder, p[index], normals[i][index])
            draw_sample(points_encoded.squeeze().transpose(0, 1).detach().numpy(), normals[i])


def encode_keypoints(pcd, keypoints):
    from point_encoder import PointEncoder

    encoder = PointEncoder("net_4.ptr", 64)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    tree = KDTree(points)
    ind = tree.query_ball_point(np.asarray(keypoints.points), r=2)
    kpn = np.asarray(keypoints.points)

    encoder.net.eval()

    out_keypoints = []
    out_features = []
    with torch.no_grad():
        for j, i in enumerate(ind):
            if len(i) > 300:
                # Normalize sampled point cloud
                p = points[i] - kpn[j]
                p = p / np.abs(p).max(axis=0)
                # Downsample if necessary
                index = np.random.choice(p.shape[0], 1000, replace=True)
                
                _, features = inference(encoder, p[index], normals[i][index])
                out_keypoints.append(kpn[j])
                out_features.append(features.cpu().detach().numpy()[0])
    
    return np.array(out_keypoints), np.transpose(np.array(out_features))


# For each key_point2 find all points from 'key_points1' that is closer than 'threshold' in feature space
def match_dists(key_points1, features1, key_points2, features2, threshold=0.5):
    tree1 = KDTree(np.transpose(features1))
    dist, ind = tree1.query(np.transpose(features2))
    dist, ind = dist[dist < threshold], ind[dist < threshold]
    return dist, ind


# Alligns two point clouds. Assumes that they are from the same area
def align(file1, file2):
    voxel_size = 0.01
    keypoint_radius = 1
    keypoint_radius_nm = 0.5
    kp = "iss"

    print("reading file 1...")
    pcd1 = read_las(in_file1)
    pcd1.paint_uniform_color([0.2, 0.2, 0.8])
    # o3d.visualization.draw_geometries([pcd1])
    print("preprocess file 1...")
    # pcd1 = pcd1.voxel_down_sample(voxel_size)
    # cl1, ind = pcd1.remove_radius_outlier(nb_points=3, radius=2)
    # cl1.estimate_normals()
    cl1=pcd1

    print("reading file 2...")
    pcd2 = read_las(in_file2)
    # pcd2 = scale_pcd(pcd2, 0.75)
    pcd2.paint_uniform_color([0.2, 0.8, 0.0])
    print("preprocess file 2...")
    # pcd2 = pcd2.voxel_down_sample(voxel_size)
    # cl2, ind = pcd2.remove_radius_outlier(nb_points=3, radius=2)
    # cl2.estimate_normals()
    cl2=pcd2

    # Detecting keypoints
    if kp == "iss":
        print("keypoints file 1...")
        keypoints1 = o3d.geometry.keypoint.compute_iss_keypoints(cl1, salient_radius=keypoint_radius, non_max_radius=keypoint_radius_nm)
        print("keypoints file 2...")
        keypoints2 = o3d.geometry.keypoint.compute_iss_keypoints(cl2, salient_radius=keypoint_radius, non_max_radius=keypoint_radius_nm)
    elif kp == "harris":
        exit(239)
        # print("keypoints file 1...")
        # cl1_np = np.asarray(cl1.points)
        # keypoints1_np = PCLKeypoint.keypointSift(cl1_np)
        # keypoints1 = o3d.geometry.PointCloud()
        # keypoints1.points = o3d.utility.Vector3dVector(keypoints1_np)
        # print("keypoints file 2...")
        # cl2_np = np.asarray(cl2.points)
        # keypoints2_np = PCLKeypoint.keypointSift(cl2_np)
        # keypoints2 = o3d.geometry.PointCloud()
        # keypoints2.points = o3d.utility.Vector3dVector(keypoints2_np)

    print("descripting keypoints...")
    keypoints1, descriptors1 = encode_keypoints(cl1, keypoints1)
    keypoints2, descriptors2 = encode_keypoints(cl2, keypoints2)

    # Prepare Open3d objects for matching
    kps1 = o3d.geometry.PointCloud()
    kps1.points = o3d.utility.Vector3dVector(keypoints1)
    ds1 = o3d.pipelines.registration.Feature()
    ds1.data = descriptors1

    kps2 = o3d.geometry.PointCloud()
    kps2.points = o3d.utility.Vector3dVector(keypoints2)
    ds2 = o3d.pipelines.registration.Feature()
    ds2.data = descriptors2

    # Find transformation
    print("ransac...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            kps1, kps2, ds1, ds2, True,
            0.6,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000000, 0.999))
    print(result)
    print(result.transformation)

    draw_registration_result(cl1, cl2, result.transformation)

    # Visualise close keypoints from 2 point clouds
    _, corr = match_dists(keypoints1, descriptors1, keypoints2, descriptors2)
    lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(kps2, kps1, list(zip(range(len(keypoints1)), corr)))
    lines.paint_uniform_color(np.array([1., 0., 0.]))
    o3d.visualization.draw_geometries([cl1, cl2, keypoints_to_spheres(kps1), keypoints_to_spheres(kps2, [1.0, 0., 0.]), axes, lines])


def rot_matr(angle, mirror):
        angle = angle * np.pi / 180
        mat = np.zeros((3, 3))
        co = np.cos(angle)
        si = np.sin(angle)
        mat[0, 0] = co
        mat[1, 1] = co
        mat[0, 1] = si
        if mirror == 0:
            mat[1, 0] = -si
        else:
            mat[1, 0] = si
        mat[2, 2] = 1
        return mat


# Prepares data to train auto-encoder
def prepare_data(files):
    voxel_size = 0.01
    keypoint_radius = 1
    keypoint_radius_nm = 0.5
    kp = "iss"

    pcds = []
    for file in files:
        pcd = read_las(file)
        pcd = pcd.voxel_down_sample(voxel_size)
        cl, ind = pcd.remove_radius_outlier(nb_points=3, radius=2)
        cl.estimate_normals()
        pcds.append(cl)

    keypoints = []
    if kp == "iss":
        for pcd in pcds:
            kps = o3d.geometry.keypoint.compute_iss_keypoints(pcd, salient_radius=keypoint_radius, non_max_radius=keypoint_radius_nm)
            keypoints.append(kps)

    out = []
    for i in range(len(pcds)):
        pcd = pcds[i]
        kps = keypoints[i]
        
        # Augment
        for mirror in (0, 1):
            for angle in np.array(list(range(0, 360, 20))):
                c = o3d.geometry.PointCloud(pcd).rotate(rot_matr(angle, mirror))
                
                lkp, _ = sample_pcds(c, np.asarray(kps.points), radius=10)
                out.extend(lkp)

    np.save("train_ae_10m.npy", out, allow_pickle=True)


if __name__ == "__main__":
    in_file1 = '/mnt/d/data/pcd1.las'
    in_file2 = '/mnt/d/data/pcd2.las'

    # prepare_data([in_file1, in_file2])
    align(in_file1, in_file2)
