import open3d as o3d
import numpy as np

# Create a simple point cloud for demonstration
points = np.random.rand(100, 3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Create the visualizer
vis = o3d.visualization.Visualizer()

# Add the point cloud to the visualizer
vis.create_window()
vis.add_geometry(point_cloud)

# Run the visualization loop
vis.run()
vis.destroy_window()