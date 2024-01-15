import open3d as o3d
import numpy as np
import time
from PIL import Image
def image_to_3d_plane(image_width, image_height, focal_length_x, focal_length_y, principal_point_x, principal_point_y, depth=1.0):
    # Step 1: Construct the intrinsics matrix
    K = np.array([[focal_length_x, 0, principal_point_x],
                  [0, focal_length_y, principal_point_y],
                  [0, 0, 1]])

    # Step 2: Create 2D grid representing image coordinates
    u, v = np.meshgrid(np.arange(image_width), np.arange(image_height))
    u = u.flatten()
    v = v.flatten()

    # Step 3: Normalize coordinates
    normalized_coordinates = np.vstack([u, v, np.ones_like(u)])
    normalized_coordinates = np.linalg.inv(K) @ normalized_coordinates

    # Step 4: Direction vector
    direction_vectors = np.vstack([normalized_coordinates, np.ones_like(u)])

    # Step 5: Set a fixed depth
    depth_values = depth * np.ones_like(u)

    # Step 6: Combine direction vectors and depth values
    points_3d = np.vstack([direction_vectors, depth_values])

    # Convert to Open3D PointCloud
    cloud = o3d.geometry.PointCloud()
    print(np.transpose(points_3d))
    cloud.points = o3d.utility.Vector3dVector(np.transpose(points_3d))

    return cloud


# Function to create a 3D mesh from an image
def create_mesh_from_image(image_path, camera_params, fx, fy, cx, cy):
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Get image dimensions
    height, width, _ = img_array.shape
    print(width)
    print(height)
    # Create a 3D plane with dimensions equal to the image resolution
    plane = o3d.geometry.TriangleMesh.create_box(width, height, 1.0)
    #plane.scale(width / fx, height / fy, 1.0)

    # Map each pixel to a vertex on the 3D plane
    vertices = np.zeros((height, width, 3), dtype=float)
    uv_coords = np.zeros((height, width, 2), dtype=float)
    for i in range(height):
        for j in range(width):
            # Map pixel coordinates to 3D coordinates
            x = (j - cx) / fx
            y = (i - cy) / fy
            vertices[i, j, 0] = 0
            vertices[i, j, 1] = ((i - cy) / fy)  # Invert y-axis to match Open3D coordinates
            vertices[i, j, 2] = (j - cx) / fx  # Use image intensity for the z-coordinate

            # Map pixel coordinates to UV coordinates
            uv_coords[i, j, 0] = j / (width - 1)
            uv_coords[i, j, 1] = 1 - i / (height - 1)  # Invert y-axis to match UV coordinates

    # Flatten the vertices array
    vertices_flat = vertices.reshape(-1, 3)
    uv_coords_flat = uv_coords.reshape(-1, 2)

    # Create triangles by connecting the vertices in a grid pattern
    triangles = []
    triangle_material_ids = o3d.utility.IntVector()
    for i in range(height - 1):
        for j in range(width - 1):
            index = i * width + j
            triangles.append([index, index + 1, index + width])
            triangles.append([index + 1, index + width + 1, index + width])
            triangle_material_ids.append(0)

    # Set vertices and triangles for the mesh
    #print(triangles)

    plane.vertices = o3d.utility.Vector3dVector(vertices_flat)
    plane.triangles = o3d.utility.Vector3iVector(triangles)
    plane.triangle_uvs = o3d.utility.Vector2dVector(uv_coords_flat)
    plane.textures = [o3d.io.read_image(image_path)]
    plane.triangle_material_ids = triangle_material_ids
    print(plane)
    return plane




# Load the mesh
mesh_a = o3d.io.read_triangle_mesh("Data/michal_alligned.obj", True)

# Intrinsic parameters (modify as needed)
intrinsic_width = 640
intrinsic_height = 480
fx = 525.0
fy = 525.0
cx = 320.0
cy = 240.0
"""[0, 0, 3] Back
"translation": [0, 0, 3], "rotation": np.array([[-1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 1]])},  # Back
"""
"""[0, 0, 3]
                       np.array( [[0, 0, -1],
                       [0, 1, 0],
                       [1, 0, 0]] )
                       @ np.array(np.array([[-1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 1]])
Left side
"""
local_rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(90), 0, 0))
print(local_rotation_matrix)
# Create a list of camera positions (extrinsic parameters)

camera_positions = [
    {"translation": [0, 0, 3], "rotation": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])} , # Right side
    {"translation": [-3, 0, 3], "rotation": np.array([[-1, 0, 1], [0, 1, 0], [0, 0, -1]])@ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])} , # Front side
    {"translation": [0, 0, 3], "rotation": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])@ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])} , # Bottom
    {"translation": [0, 0, 3], "rotation": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])} ,  # Back
    {"translation": [0, 0, 3], "rotation": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]] )@ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])} ,  # Left side
    {"translation": [0, 0, 3], "rotation": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])}  # Top
]

# Visualization
vis = o3d.visualization.Visualizer()
vis.create_window(width=intrinsic_width, height=intrinsic_height)
point_clouds = []
for i, camera_pos in enumerate(camera_positions):
    camera_params = o3d.camera.PinholeCameraParameters()

    # Set intrinsic parameters
    camera_params.intrinsic.set_intrinsics(intrinsic_width, intrinsic_height, fx, fy, cx, cy)

    # Set extrinsic parameters
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = camera_pos["translation"]
    extrinsic[:3, :3] = camera_pos["rotation"]

    camera_params.extrinsic = extrinsic

    # Add geometry to the visualization
    vis.add_geometry(mesh_a)

    # Set the virtual camera parameters
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params, True)


    # Render and show the scene
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"camera_view_{i}.png")
    #mesh = create_mesh_from_image(f"camera_view_{i}.png", camera_params, fx, fy, cx, cy)
    #rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)
    # Define image size
    width = 640
    height = 480

    # Create a flat depth map with a constant depth value
    depth_value = 1.0  # Replace with your desired depth value
    depth_image = np.full((height, width), depth_value, dtype=np.float32)

    # Create Open3D Image object
    depth_o3d = o3d.geometry.Image(depth_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.io.read_image(f"camera_view_{i}.png"), depth_o3d, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_params.intrinsic, camera_params.extrinsic)
    print(pcd)
    #point_cloud = o3d.geometry.create_point_cloud_from_rgbd_image(o3d.io.read_image(f"camera_view_{i}.png"), camera_params.intrinsic, camera_params.extrinsic)
    pcd = image_to_3d_plane(width,height,fx,fy,cx,cy)
    #point_cloud = o3d.geometry.PointCloud()
    #point_cloud.points = o3d.utility.Vector3dVector(mesh.vertices)
    vis.remove_geometry(mesh_a)
    # Convert point cloud to triangle mesh
    #pcd.estimate_normals()
    #tri_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

    # Visualize the original point cloud and the resulting triangle mesh
    #o3d.visualization.draw_geometries([pcd, tri_mesh])
    #vis.add_geometry(pcd)
    #vis.add_geometry(tri_mesh)
    #vis.add_geometry(mesh_a)
    vis.add_geometry(pcd)
    # Visualize the scene
    vis.run()
    #point_clouds.append(tri_mesh)
    # Remove geometry for the next view
    #vis.remove_geometry(mesh_a)


#point_clouds.append(mesh_a)
#o3d.visualization.draw_geometries(point_clouds)
# Close the visualization window
#vis.destroy_window()