import open3d as o3d
import numpy as np
from star.pytorch.star import STAR
import numpy as np
from numpy import newaxis
import pickle
import os
import torch

star = STAR(gender='female')
betas = np.array([
            np.array([ 0, 0, 0, 0,
                      0, 0, 0, 0,
                      0, 0])])
num_betas=10
batch_size=1
m = STAR(gender='male',num_betas=num_betas)

# Zero pose
poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
betas = torch.cuda.FloatTensor(betas)

trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
model = star.forward(poses, betas,trans)
shaped = model.v_shaped[-1, :, :]
numpy_array = model.squeeze().cpu().numpy()

# Print or use the NumPy array as needed
print(numpy_array)

path_smplx_meshes = 'smplx_meshes.npy'
smplx = np.load(path_smplx_meshes)[1]
print(smplx)
print((np.load(path_smplx_meshes)).shape)


point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(numpy_array)
o3d.io.write_point_cloud("output_cloud.ply", point_cloud_o3d)

# Convert NumPy array to Open3D PointCloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(numpy_array)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])




# Load PLY file
ply_file_path = "michal_skan.ply"
point_cloud = o3d.io.read_point_cloud(ply_file_path)

# Visualize the point cloud
#o3d.visualization.draw_geometries([point_cloud])



mesh = o3d.io.read_triangle_mesh(ply_file_path)
# Extract vertices from the mesh


# Extract vertices from the mesh
vertices = np.asarray(mesh.vertices)


#smplx = np.load(numpy_array)


# Convert NumPy array to Open3D PointCloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(vertices)

print(vertices)

new_array_1 = vertices[np.newaxis, :]

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])