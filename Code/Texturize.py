import open3d as o3d
import numpy as np
import torch
from star.pytorch.star import STAR


def create_star_model(poses, betas, trans):
    ################### CREATE STAR MODEL ###################
    poses = torch.cuda.FloatTensor(poses)
    betas = torch.cuda.FloatTensor(betas)
    trans = torch.cuda.FloatTensor(trans)

    star_gender = 'male'
    star = STAR(gender=star_gender)
    # Save or visualize the resulting mesh
    d = star(poses, betas, trans)
    ######################################
    return d

def find_nearest_neibghour(target, source):
    distances = torch.norm(target[:, :, None, :] - source[:, None, :, :], dim=-1)

    # Find nearest neighbors
    min_distances, min_indices = torch.min(distances, dim=-1)

    return min_distances, min_indices

def create_star_mesh(d, star_colors):
    star_verts = d.cpu().numpy().squeeze()
    star_faces = d.f

    avatar = o3d.geometry.TriangleMesh()
    avatar.vertices = o3d.utility.Vector3dVector(star_verts)
    avatar.triangles = o3d.utility.Vector3iVector(star_faces)
    avatar.vertex_colors = o3d.utility.Vector3dVector(star_colors)
    return avatar


def texturize_scan_nnn(star_scan_poses_path='michal_skan_star.npy', context_mesh_path="transformed_mesh.ply", texturized_scan_save_path='michal_skan_star_colored.npy'):
    ################### LOAD STAR MODEL DATA ###################
    # Load the NumPy structured array from a .npy file
    point_cloud_data = np.load(star_scan_poses_path, allow_pickle=True).item()

    # Extract star_verts
    trans = point_cloud_data['trans']
    poses = point_cloud_data['poses']
    betas = point_cloud_data['betas']
    star_verts = point_cloud_data['star_verts'][0]
    scale = point_cloud_data['scale'][0]
    ######################################

    ################### CREATE STAR MODEL ###################
    d = create_star_model(poses, betas, trans)
    ######################################



    ################### LOAD CONTEXT MESH FOR TEXTURING ###################
    scan_mesh = o3d.io.read_triangle_mesh(context_mesh_path)
    scan_vertices = np.asarray(scan_mesh.vertices)
    scan_vertices_npy = scan_vertices[np.newaxis, :]


    scan_vertices_npy = torch.cuda.FloatTensor(scan_vertices_npy)
    ######################################


    ################### FIND NEAREST NEIBHOUR ###################
    min_distances, min_indices = find_nearest_neibghour(d, scan_vertices_npy)
    ######################################

    ################### FIND VERTEX COLOR BASED ON NEAREST NEIBHOUR ###################
    # Get color from scan_mesh for each nearest neighbor
    scan_vertex_colors = np.asarray(scan_mesh.vertex_colors)
    star_colors = scan_vertex_colors[min_indices.cpu().numpy()]
    star_colors = star_colors.squeeze()

    ######################################

    ################### CREATE TRIANGLE STAR MESH ###################
    avatar = create_star_mesh(d, star_colors)

    ######################################

    # Visualize the mesh
    o3d.visualization.draw_geometries([avatar])

    print(star_colors)
    print(scale)
    results = {'poses': poses, 'betas': betas, 'trans' : trans, 'star_verts' : star_verts,
               'scale' : scale, 'colors': star_colors}

    print('Saving the results %s.'%(texturized_scan_save_path))
    np.save(texturized_scan_save_path, results)

