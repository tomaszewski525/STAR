from star.pytorch.star import STAR
import numpy as np
from numpy import newaxis
import pickle
import os
import torch
import open3d as o3d
from ChangeOptimPose import ModelManipulator
from losses import convert_scan_2_star
from Texturize import texturize_scan_nnn
class AvatarCreator:
    def __init__(self, scan_path, transformed_scan_save_path, star_default_poses_path = "manipulated_star_poses.npy"):
        print("init")

    def allign_scan_to_star(self, scan_path, star_default_poses_path, save_scan_path):
        ######### Create STAR body model for icp transformation estimation #########
        star = STAR(gender='male')
        betas = np.array([
            np.array([0, 0, 0, 0,
                      0, 0, 0, 0,
                      0, 0])])
        num_betas = 10
        batch_size = 1
        m = STAR(gender='male', num_betas=num_betas)

        # Specify the path to your .npy file
        file_path = star_default_poses_path
        data = np.load(file_path)

        # Zero pose
        poses = torch.cuda.FloatTensor(np.zeros((batch_size, 72)))
        betas = torch.cuda.FloatTensor(betas)

        trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
        model = star.forward(poses, betas, trans)
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
        o3d.io.write_triangle_mesh(save_scan_path, scan_mesh)


    def change_manually_star_pose(self, star_poses_save_file, context_scan_path):
        pose_manipulator = ModelManipulator(star_poses_save_file, context_scan_path)
        pose_manipulator.viusalize()

    def optimize_star_to_scan(self, scan_file_path, star_poses_path, path_save_star_parms):

        ### Clear unused GPU memory
        torch.cuda.empty_cache()

        scan_mesh = o3d.io.read_triangle_mesh(scan_file_path)
        scan_vertices = np.asarray(scan_mesh.vertices)
        scan_vertices_npy = scan_vertices[np.newaxis, :]

        star_gender = 'male'  # STAR Model Gender (options: male,female,neutral).
        MAX_ITER_EDGES = 100  # Number of LBFGS iterations for an on edges objective
        MAX_ITER_VERTS = 100  # Number of LBFGS iterations for an on vertices objective
        NUM_BETAS = 10

        opt_parms = {'MAX_ITER_EDGES': MAX_ITER_EDGES,
                     'MAX_ITER_VERTS': MAX_ITER_VERTS,
                     'NUM_BETAS': NUM_BETAS,
                     'GENDER': star_gender}
        ########################################################################################################################

        print('Loading the SMPL Meshes and ')

        np_poses, np_betas, np_trans, star_verts, np_scale = convert_scan_2_star(scan_vertices_npy,
                                                                                 star_poses_path, **opt_parms)
        results = {'poses': np_poses, 'betas': np_betas, 'trans': np_trans, 'star_verts': star_verts, 'scale': np_scale}
        print('Saving the results %s.' % (path_save_star_parms))
        np.save(path_save_star_parms, results)


    def texturize_star_nnn(self, star_scan_poses, transformed_scan_path, texturized_scan_poses_save_path):
        texturize_scan_nnn(star_scan_poses, transformed_scan_path, texturized_scan_poses_save_path)
