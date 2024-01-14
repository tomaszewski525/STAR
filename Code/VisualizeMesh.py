import open3d as o3d
import numpy as np
"""
# Load the NumPy structured array from a .npy file
point_cloud_data = np.load('Data/michal_skan_star.npy', allow_pickle=True).item()

# Extract star_verts
star_verts = point_cloud_data['star_verts'][0]
scale = point_cloud_data['scale'][0]

# Create an Open3D PointCloud object
point_cloud = o3d.geometry.PointCloud()

# Set the points in the PointCloud object
point_cloud.points = o3d.utility.Vector3dVector(star_verts*scale)
"""

######### Load scanned mesh #########
scan_file_path = "Data/transformed_mesh.ply"
scan_mesh = o3d.io.read_triangle_mesh(scan_file_path)
scan_vertices = np.asarray(scan_mesh.vertices)
scan_pc = o3d.geometry.PointCloud()
scan_pc.points = o3d.utility.Vector3dVector(scan_vertices)


# Visualize the point cloud
o3d.visualization.draw_geometries([scan_mesh])