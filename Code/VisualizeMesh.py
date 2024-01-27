import open3d as o3d
import numpy as np
######### Load scanned mesh #########
scan_file_path = "Data/transformed_mesh.ply"
scan_mesh = o3d.io.read_triangle_mesh(scan_file_path)
scan_vertices = np.asarray(scan_mesh.vertices)
scan_pc = o3d.geometry.PointCloud()
scan_pc.points = o3d.utility.Vector3dVector(scan_vertices)


# Visualize the point cloud
o3d.visualization.draw_geometries([scan_mesh])