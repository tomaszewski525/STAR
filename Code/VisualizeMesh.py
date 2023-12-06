import open3d as o3d
import numpy as np

# Load the NumPy structured array from a .npy file
point_cloud_data = np.load('michal_skan_star.npy', allow_pickle=True).item()
print(point_cloud_data)

# Extract star_verts
star_verts = point_cloud_data['star_verts'][0]

# Create an Open3D PointCloud object
point_cloud = o3d.geometry.PointCloud()

# Set the points in the PointCloud object
point_cloud.points = o3d.utility.Vector3dVector(star_verts)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])