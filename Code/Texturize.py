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


def find_k_nearest_neibghour(target, k, source):
    distances = torch.norm(target[:, :, None, :] - source[:, None, :, :], dim=-1)

    # Find nearest neighbors
    # Find k-nearest neighbors
    _, indices = torch.topk(distances, k, dim=-1, largest=False)

    return indices



def create_star_mesh(d, star_colors, translation):
    star_verts = d.cpu().numpy().squeeze()
    star_verts = star_verts + translation
    star_faces = d.f

    avatar = o3d.geometry.TriangleMesh()
    avatar.vertices = o3d.utility.Vector3dVector(star_verts)
    avatar.triangles = o3d.utility.Vector3iVector(star_faces)


    avatar.vertex_colors = o3d.utility.Vector3dVector(star_colors)
    return avatar

def create_star_mesh_obj(d, mesh, translation):
    star_verts = d.cpu().numpy().squeeze()
    star_verts = star_verts + translation
    star_faces = d.f

    avatar = o3d.geometry.TriangleMesh()
    avatar.vertices = o3d.utility.Vector3dVector(star_verts)
    avatar.triangles = o3d.utility.Vector3iVector(star_faces)
    avatar.textures = mesh.textures
    avatar.triangle_uvs = mesh.triangle_uvs
    avatar.triangle_material_ids = mesh.triangle_material_ids
    return avatar

def create_star_mesh_no_color(d, translation):
    star_verts = d.cpu().numpy().squeeze()
    star_verts = star_verts + translation
    star_faces = d.f

    avatar = o3d.geometry.TriangleMesh()
    avatar.vertices = o3d.utility.Vector3dVector(star_verts)
    avatar.triangles = o3d.utility.Vector3iVector(star_faces)


    return avatar

def create_star_mesh_no_color_scale(d, translation, scale):
    star_verts = d.cpu().numpy().squeeze()*scale
    star_verts = star_verts + translation
    star_faces = d.f

    avatar = o3d.geometry.TriangleMesh()
    avatar.vertices = o3d.utility.Vector3dVector(star_verts)
    avatar.triangles = o3d.utility.Vector3iVector(star_faces)


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
    avatar = create_star_mesh(d, star_colors, trans)

    ######################################

    # Visualize the mesh
    o3d.visualization.draw_geometries([avatar])


    results = {'poses': poses, 'betas': betas, 'trans' : trans, 'star_verts' : star_verts,
               'scale' : scale, 'colors': star_colors}

    print('Saving the results %s.'%(texturized_scan_save_path))
    np.save(texturized_scan_save_path, results)



def texturize_scan_k_nnn(star_scan_poses_path='michal_skan_star.npy', context_mesh_path="transformed_mesh.ply",
                       texturized_scan_save_path='michal_skan_star_colored.npy', k=1):
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
    avatar = create_star_mesh_no_color_scale(d, trans, scale)
    o3d.io.write_triangle_mesh("michal_star_2.ply", avatar)

    ################### LOAD CONTEXT MESH FOR TEXTURING ###################
    scan_mesh = o3d.io.read_triangle_mesh(context_mesh_path)
    scan_vertices = np.asarray(scan_mesh.vertices)
    scan_vertices_npy = scan_vertices[np.newaxis, :]

    scan_vertices_npy = torch.cuda.FloatTensor(scan_vertices_npy)
    ######################################

    ################### FIND NEAREST NEIBHOUR ###################
    indicies = find_k_nearest_neibghour(d, k, scan_vertices_npy)
    min_distances, min_indices = find_nearest_neibghour(d, scan_vertices_npy)
    ######################################

    ################### Get avarage vertex color ###################
    # Get color from scan_mesh for each nearest neighbor
    scan_vertex_colors = np.asarray(scan_mesh.vertex_colors)
    # Get color from vertex_colors for each nearest neighbor
    neighbor_colors = scan_vertex_colors[indicies.cpu().numpy()]

    # Calculate the average color along the last dimension (k dimension)
    average_color = np.mean(neighbor_colors, axis=-2)
    star_colors = average_color.squeeze()
    ######################################
    ################### CREATE TRIANGLE STAR MESH ###################
    avatar = create_star_mesh(d, star_colors, trans)

    ######################################

    # Visualize the mesh
    o3d.visualization.draw_geometries([avatar])

    results = {'poses': poses, 'betas': betas, 'trans': trans, 'star_verts': star_verts,
               'scale': scale, 'colors': star_colors}

    print('Saving the results %s.' % (texturized_scan_save_path))
    np.save(texturized_scan_save_path, results)




def attach_pointcloud_to_scan(star_scan_poses_path='michal_skan_star.npy', context_mesh_path="transformed_mesh.ply"):
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
    min_distances, min_indices = find_nearest_neibghour(scan_vertices_npy, d)
    ######################################

    ################### ATTACH ###################

    return min_indices



def calculate_new_scanned_verticies_positions(min_indicies, scanned_mesh_file_path, previous_star_verticies, current_star_verticies):

    ################### LOAD PLY MESH ###################
    scan_mesh = o3d.io.read_triangle_mesh(scanned_mesh_file_path)
    scan_vertices = np.asarray(scan_mesh.vertices)
    scan_vertices_npy = scan_vertices[np.newaxis, :]

    scan_verticies = torch.FloatTensor(scan_vertices)



    # Extract the corresponding vertices from cuurent_star_verticies_locations
    corresponding_current_vertices = current_star_verticies[:, min_indicies[0], :]
    corresponding_previous_vertices = previous_star_verticies[:, min_indicies[0], :]

    # Calculate the rigid transformation matrix
    # We'll use Procrustes analysis to find the optimal rotation, scaling, and translation
    A = corresponding_previous_vertices.unsqueeze(0)
    B = corresponding_current_vertices.unsqueeze(0)


    rot, trans = procrustes_analysis_batched(A, B)

    # Use Procrustes to find optimal transformation
    #scale, rot_matrix, translation = torch.linalg.svd(torch.bmm(B.transpose(1, 2), A.transpose(1, 2).transpose(2, 1)))

    # Apply the transformation to the vertices in scanned_mesh
    #transformed_scanned_mesh = torch.bmm(scan_verticies.unsqueeze(0), rot_matrix.transpose(1, 2)) * scale + translation
    #print(transformed_scanned_mesh)
    # transformed_scanned_mesh now contains the vertices of scanned_mesh transformed
    # based on the rigid transformation between previous and current star vertices.


def procrustes_analysis_batched(tensor1, tensor2):
    # Reshape tensors to 2D matrices
    matrix1 = tensor1.view(-1, 3)
    matrix2 = tensor2.view(-1, 3)

    # Calculate means
    mean1 = matrix1.mean(dim=0)
    mean2 = matrix2.mean(dim=0)

    # Center the matrices
    centered1 = matrix1 - mean1
    centered2 = matrix2 - mean2

    # Calculate covariance matrix
    cov_matrix = torch.matmul(centered2.t(), centered1)

    # Singular Value Decomposition
    _, _, V = torch.svd(cov_matrix)

    # Calculate rotation matrix
    rotation_matrix = torch.matmul(V, V.t())

    # Calculate translation vector
    translation_vector = mean1 - torch.matmul(rotation_matrix, mean2)

    return rotation_matrix, translation_vector