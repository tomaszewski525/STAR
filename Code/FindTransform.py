import numpy as np
from scipy.spatial.transform import Rotation
from star.pytorch.star import STAR
import numpy as np
from numpy import newaxis
import pickle
import os
import torch
import open3d as o3d

def allign_model(scan_path="michal_skan.ply", tranformed_scan_path="transformed_mesh.ply"):
    ######### Create STAR body model for icp transformation estimation #########
    star = STAR(gender='male')
    betas = np.array([
                np.array([ 0, 0, 0, 0,
                          0, 0, 0, 0,
                          0, 0])])
    num_betas=10
    batch_size=1
    m = STAR(gender='male',num_betas=num_betas)

    # Specify the path to your .npy file
    file_path = path_neutral_star = 'C:/Users/tfran/Desktop/Inzynierka/STAR/STAR/Code/star_poses.npy'
    data = np.load(file_path)


    # Zero pose
    poses = torch.cuda.FloatTensor(np.zeros((batch_size, 72)))
    betas = torch.cuda.FloatTensor(betas)

    trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
    model = star.forward(poses, betas,trans)
    shaped = model.v_shaped[-1, :, :]
    star_verticies = model.squeeze().cpu().numpy()


    ######### Convert star to point cloud #########
    star_pc = o3d.geometry.PointCloud()
    star_pc.points = o3d.utility.Vector3dVector(star_verticies)



    ######### Load scanned mesh #########
    scan_mesh = o3d.io.read_triangle_mesh(scan_path)
    scan_vertices = np.asarray(scan_mesh.vertices)
    scan_pc = o3d.geometry.PointCloud()
    scan_pc.points = o3d.utility.Vector3dVector(scan_vertices)

    ######### Before icp transform estimation #########
    o3d.visualization.draw_geometries([scan_mesh, star_pc])


    ######### Perform Generalized ICP registration #########
    star_pc.estimate_normals()
    scan_pc.estimate_normals()
    reg_gicp = o3d.pipelines.registration.registration_icp(
        scan_pc, star_pc, 0.2, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300)
    )

    ######### Apply estimated transform to scan_mesh
    scan_mesh.transform(reg_gicp.transformation)

    ######### After icp transform estimation #########
    o3d.visualization.draw_geometries([scan_mesh, star_pc])


    ######### Save tranformed mesh #########
    o3d.io.write_triangle_mesh(tranformed_scan_path, scan_mesh)