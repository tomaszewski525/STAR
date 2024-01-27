import subprocess

from PyNuitrack import py_nuitrack
import time
import sys, os
sys.path.append(os.getcwd())
import torch
import numpy as np
from star.pytorch.star import STAR
from scipy.spatial.transform import Rotation
import cv2
import open3d as o3d
import Texturize
import FindTransform
import ChangeOptimPose
import convert_scan_to_star
from avatar_creator import AvatarCreator

#scan_path = "Data/Michal_skan_alligned_manually.obj"
"""
scan_path = "Data/michal_skan.ply"
transformed_scan_path = "Data/michal_alligned_PLY.ply"
star_default_poses_path = "Data/star_poses.npy"
star_scan_poses_bef_path = "Data/manipulated_star_poses.npy"
star_scan_poses_aft_path='Data/michal_skan_star_PLY.npy'
"""
#transformed_scan_path = "Data/michal_alligned.obj"
#transformed_scan_path = "Data/Michal_Low_Poly_alligned.obj"



scan_path = "Data/Michal_skan_alligned_manually.obj"
transformed_scan_path = "Data/Michal_Low_Poly_alligned.obj"
star_default_poses_path = "Data/star_poses.npy"
star_scan_poses_bef_path = "Data/manipulated_star_poses.npy"
star_scan_poses_aft_path='Data/michal_skan_star.npy'



texturized_scan_poses_save_path = 'Data/michal_skan_star_colored.npy'

#mesh = o3d.io.read_triangle_mesh("Data\Michal_LowPoly_StarMesh_Texture.obj", True)
#o3d.visualization.draw_geometries([mesh])

avatar_creator = AvatarCreator(scan_path, transformed_scan_path, star_scan_poses_bef_path)

allign_model = False
change_optim_pose = False
optimize_model = False
texturize_model = False
visualize_model = False
apply_joints = True

attach_point_cloud_to_mesh = True



if allign_model:
    avatar_creator.allign_scan_to_star(scan_path, star_default_poses_path, transformed_scan_path)

if change_optim_pose:
    avatar_creator.change_manually_star_pose(star_scan_poses_bef_path, transformed_scan_path)

if False:
    # Load the NumPy structured array from a .npy file
    poses = np.load(star_scan_poses_bef_path, allow_pickle=True)

    star = STAR(gender='male')
    betas = np.array([
        np.array([0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0])])
    num_betas = 10
    batch_size = 1
    m = STAR(gender='male', num_betas=num_betas)

    # Zero pose
    all_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 72)))
    poses = torch.cuda.FloatTensor(poses)
    betas = torch.cuda.FloatTensor(betas)

    trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    model = star.forward(all_pose, betas, trans)
    shaped = model.v_shaped[-1, :, :]
    star_verticies = model.squeeze().cpu().numpy()

    ######### Convert star to point cloud #########
    star_pc = o3d.geometry.PointCloud()
    star_pc.points = o3d.utility.Vector3dVector(star_verticies)


    mesh_a = o3d.io.read_triangle_mesh('Data\Michal_Low_Poly_alligned.obj', True)
    #o3d.io.write_triangle_mesh("michal_star.ply", avatar)
    o3d.visualization.draw_geometries([star_pc])
    #mesh_a = o3d.io.read_triangle_mesh(transformed_scan_path, True)

mesh_a = o3d.io.read_triangle_mesh(transformed_scan_path, True)
print(mesh_a)
if optimize_model:
    avatar_creator.optimize_star_to_scan(transformed_scan_path, star_default_poses_path, star_scan_poses_aft_path)

if texturize_model:
    k = 5
    #Texturize.texturize_scan_nnn(star_scan_poses_aft_path, transformed_scan_path, texturized_scan_poses_save_path)
    Texturize.texturize_scan_k_nnn(star_scan_poses_aft_path, transformed_scan_path, texturized_scan_poses_save_path, k)

if visualize_model:
    # Load the NumPy structured array from a .npy file
    point_cloud_data = np.load(star_scan_poses_aft_path, allow_pickle=True).item()

    # Extract star_verts
    trans = point_cloud_data['trans']
    poses = point_cloud_data['poses']
    betas = point_cloud_data['betas']
    star_verts = point_cloud_data['star_verts'][0]
    scale = point_cloud_data['scale'][0]
    ######################################

    ################### CREATE STAR MODEL ###################
    d = Texturize.create_star_model(poses, betas, trans)
    ######################################
    ######### Convert star to point cloud #########
    star_pc = o3d.geometry.PointCloud()
    star_pc.points = o3d.utility.Vector3dVector(star_verts)

    d = Texturize.create_star_model(poses, betas, trans)

    mesh_a = o3d.io.read_triangle_mesh(transformed_scan_path, True)
    o3d.io.write_triangle_mesh("michal_Low_poly_star.ply", Texturize.create_star_mesh_no_color(d, trans))
    o3d.visualization.draw_geometries([star_pc, mesh_a])

#connection_inidices_of_star = Texturize.attach_pointcloud_to_scan(star_scan_poses_aft_path, transformed_scan_path)

# 1. Repair scan
# 2. Optimize model
# 3. Texturize model
# 4. Apply Joints rotations

if apply_joints:
    #################### OPTIMIZE MODEL ############################
    star_mesh = o3d.io.read_triangle_mesh("Data/Michal_LowPoly_star_texturized.obj", True)

    ####################### LOAD STAR MODEL ########################
    # Load the NumPy structured array from a .npy file
    point_cloud_data = np.load(texturized_scan_poses_save_path, allow_pickle=True).item()
    print(point_cloud_data)
    # Extract star_verts
    trans = point_cloud_data['trans']
    poses = point_cloud_data['poses']
    betas = point_cloud_data['betas']
    star_verts = point_cloud_data['star_verts'][0]
    scale = point_cloud_data['scale'][0]
    star_colors = point_cloud_data['colors']
    ################################################################


    ######################## CREATE VISUALIZER #########################
    # Create a visualization window
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    ####################################################################

    ######################## DATA #########################
    kin = [[0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15], [0, 3, 6, 9, 13, 16, 18, 20, 22],
           [0, 3, 6, 9, 14, 17, 19, 21, 23]]

    smpl_joints = {
        0: 'pelvis',
        1: 'left_hip',
        2: 'right_hip',
        3: 'spine1',
        4: 'left_knee',
        5: 'right_knee',
        6: 'spine2',
        7: 'left_ankle',
        8: 'right_ankle',
        9: 'spine3',
        10: 'left_foot',
        11: 'right_foot',
        12: 'neck',
        13: 'left_collar',
        14: 'right_collar',
        15: 'head',
        16: 'left_shoulder',
        17: 'right_shoulder',
        18: 'left_elbow',
        19: 'right_elbow',
        20: 'left_wrist',
        21: 'right_wrist',
        22: 'left_hand',
        23: 'right_hand'
    }
    nuitrack_to_smpl = {
        py_nuitrack.JointType.waist:[0],
        py_nuitrack.JointType.left_hip:[1],
        py_nuitrack.JointType.right_hip:[2],
        py_nuitrack.JointType.torso:[3, 6, 9],
        py_nuitrack.JointType.left_knee:[4],
        py_nuitrack.JointType.right_knee:[5],
        py_nuitrack.JointType.left_ankle:[7],
        py_nuitrack.JointType.right_ankle:[8],
        py_nuitrack.JointType.neck:[12],
        py_nuitrack.JointType.left_collar:[13],
        py_nuitrack.JointType.right_collar:[14],
        py_nuitrack.JointType.head:[15],
        py_nuitrack.JointType.left_shoulder:[16],
        py_nuitrack.JointType.right_shoulder:[17],
        py_nuitrack.JointType.left_elbow:[18],
        py_nuitrack.JointType.right_elbow:[19],
        py_nuitrack.JointType.left_wrist:[20],
        py_nuitrack.JointType.right_wrist:[21],
        py_nuitrack.JointType.left_hand:[22],
        py_nuitrack.JointType.right_hand:[23],
    }

    all_nuitrack_joint_types = [py_nuitrack.JointType.waist, py_nuitrack.JointType.left_hip, py_nuitrack.JointType.right_hip,
    py_nuitrack.JointType.torso, py_nuitrack.JointType.left_knee, py_nuitrack.JointType.right_knee, py_nuitrack.JointType.left_ankle,
    py_nuitrack.JointType.right_ankle, py_nuitrack.JointType.neck, py_nuitrack.JointType.left_collar, py_nuitrack.JointType.right_collar,
    py_nuitrack.JointType.head, py_nuitrack.JointType.left_shoulder, py_nuitrack.JointType.right_shoulder, py_nuitrack.JointType.left_elbow, py_nuitrack.JointType.right_elbow,
    py_nuitrack.JointType.left_wrist, py_nuitrack.JointType.right_wrist, py_nuitrack.JointType.left_hand, py_nuitrack.JointType.right_hand
    ]

    joints_to_ignore =[py_nuitrack.JointType.left_wrist, py_nuitrack.JointType.right_wrist, py_nuitrack.JointType.left_hand,  py_nuitrack.JointType.right_hand
    ]

    nuitrack_to_smpl_axis = {
        py_nuitrack.JointType.waist:[-1, 1, 1],
        py_nuitrack.JointType.left_hip:[-1, 1, 1],
        py_nuitrack.JointType.right_hip:[-1, 1, 1],
        py_nuitrack.JointType.torso:[-1, 1, 1],
        py_nuitrack.JointType.left_knee:[-1, 1, 1],
        py_nuitrack.JointType.right_knee:[-1, 1, 1],
        py_nuitrack.JointType.left_ankle:[-1, 1, 1],
        py_nuitrack.JointType.right_ankle:[-1, 1, 1],
        py_nuitrack.JointType.neck:[-1, 1, 1],
        py_nuitrack.JointType.left_collar:[1, 1, 1],
        py_nuitrack.JointType.right_collar:[1, 1, 1],
        py_nuitrack.JointType.head:[-1, 1, 1],
        py_nuitrack.JointType.left_shoulder:[1, -1, 1],
        py_nuitrack.JointType.right_shoulder:[1, -1, 1],
        py_nuitrack.JointType.left_elbow:[-1, -1, 1],
        py_nuitrack.JointType.right_elbow:[-1, -1, 1],
        py_nuitrack.JointType.left_wrist:[1, -1, 1],
        py_nuitrack.JointType.right_wrist:[1, -1, 1],
        py_nuitrack.JointType.left_hand:[1, -1, 1],
        py_nuitrack.JointType.right_hand:[1, -1, 1],
    }

    nuitrack_joint_memory = {
        py_nuitrack.JointType.waist: [],
        py_nuitrack.JointType.left_hip: [],
        py_nuitrack.JointType.right_hip: [],
        py_nuitrack.JointType.torso: [],
        py_nuitrack.JointType.left_knee: [],
        py_nuitrack.JointType.right_knee: [],
        py_nuitrack.JointType.left_ankle: [],
        py_nuitrack.JointType.right_ankle: [],
        py_nuitrack.JointType.neck: [],
        py_nuitrack.JointType.left_collar: [],
        py_nuitrack.JointType.right_collar: [],
        py_nuitrack.JointType.head: [],
        py_nuitrack.JointType.left_shoulder: [],
        py_nuitrack.JointType.right_shoulder: [],
        py_nuitrack.JointType.left_elbow: [],
        py_nuitrack.JointType.right_elbow: [],
        py_nuitrack.JointType.left_wrist: [],
        py_nuitrack.JointType.right_wrist: [],
        py_nuitrack.JointType.left_hand: [],
        py_nuitrack.JointType.right_hand: [],
    }
    timestamps = {
        py_nuitrack.JointType.waist: [],
        py_nuitrack.JointType.left_hip: [],
        py_nuitrack.JointType.right_hip: [],
        py_nuitrack.JointType.torso: [],
        py_nuitrack.JointType.left_knee: [],
        py_nuitrack.JointType.right_knee: [],
        py_nuitrack.JointType.left_ankle: [],
        py_nuitrack.JointType.right_ankle: [],
        py_nuitrack.JointType.neck: [],
        py_nuitrack.JointType.left_collar: [],
        py_nuitrack.JointType.right_collar: [],
        py_nuitrack.JointType.head: [],
        py_nuitrack.JointType.left_shoulder: [],
        py_nuitrack.JointType.right_shoulder: [],
        py_nuitrack.JointType.left_elbow: [],
        py_nuitrack.JointType.right_elbow: [],
        py_nuitrack.JointType.left_wrist: [],
        py_nuitrack.JointType.right_wrist: [],
        py_nuitrack.JointType.left_hand: [],
        py_nuitrack.JointType.right_hand: [],
    }

    def get_star_kinematic_chain(joint):
        connections = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],
                       [5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],
                       [12,15],[13,16],[14,17],[16,18],[17,19],[18,20],
                       [19,21],[20,22],[21,23]]

        kin = [[0,1,4,7,10], [0,2,5,8,11], [0,3,6,9,12,15], [0,3,6,9,13,16,18,20,22], [0,3,6,9,14,17,19,21,23]]

        joints_that_influence = []
        for chain in kin:
            for j in chain:
                if j == joint:
                    return joints_that_influence
                else:
                    joints_that_influence.append(j)

            joints_that_influence = []


        return joints_that_influence

    circular_buffer = {
        py_nuitrack.JointType.waist: [],
        py_nuitrack.JointType.left_hip: [],
        py_nuitrack.JointType.right_hip: [],
        py_nuitrack.JointType.torso: [],
        py_nuitrack.JointType.left_knee: [],
        py_nuitrack.JointType.right_knee: [],
        py_nuitrack.JointType.left_ankle: [],
        py_nuitrack.JointType.right_ankle: [],
        py_nuitrack.JointType.neck: [],
        py_nuitrack.JointType.left_collar: [],
        py_nuitrack.JointType.right_collar: [],
        py_nuitrack.JointType.head: [],
        py_nuitrack.JointType.left_shoulder: [],
        py_nuitrack.JointType.right_shoulder: [],
        py_nuitrack.JointType.left_elbow: [],
        py_nuitrack.JointType.right_elbow: [],
        py_nuitrack.JointType.left_wrist: [],
        py_nuitrack.JointType.right_wrist: [],
        py_nuitrack.JointType.left_hand: [],
        py_nuitrack.JointType.right_hand: [],
    }
    ###############################################################
    def convert_rot_mat_to_euler(rot_mat):
        ### first transform the matrix to euler angles
        r = Rotation.from_matrix(rot_mat)
        angles = r.as_rotvec()
        return angles

    def manipulate_joint(joint_num, new_value, poses):
        poses[0,joint_num*3] = new_value[0]
        poses[0,joint_num*3+1] = new_value[1]
        poses[0,joint_num*3+2] = new_value[2]
        return poses


    def get_joint_orientation(joint_num):
        x = poses[0, joint_num * 3]
        y = poses[0, joint_num * 3 + 1]
        z = poses[0, joint_num * 3 + 2]

        return [x,y,z]


    BUFFER_SIZE = 150
    TIME_THRESHOLD = 1.0


    def convert_to_relative_rotation(glob_joint, global_orientation):
        joints = get_star_kinematic_chain(glob_joint)

        for joint in joints:
            local_orientation = get_joint_orientation(joint)

            global_orientation[0] = global_orientation[0] - local_orientation[0]
            global_orientation[1] = global_orientation[1] - local_orientation[1]
            global_orientation[2] = global_orientation[2] - local_orientation[2]

        return global_orientation


    def add_data(orientation, joint_type):
        global circular_buffer, timestamps
        current_time = time.time()

        memory_values = circular_buffer[joint_type]
        memory_values.append((orientation, current_time))
        circular_buffer[joint_type] = memory_values


        timestamp_memory = timestamps[joint_type]
        timestamp_memory.append(current_time)
        timestamps[joint_type] = timestamp_memory


        i = 0
        last_index = len(timestamp_memory)-1
        indexes_to_pop = []
        for time_past in timestamp_memory:
            time_difference = timestamp_memory[last_index] - time_past
            if time_difference > TIME_THRESHOLD:
                indexes_to_pop.append(i)
            i = i+1

        # Remove elements from timestamp_memory based on indexes_to_pop
        for index in reversed(indexes_to_pop):
            timestamp_memory.pop(index)

        timestamps[joint_type] = timestamp_memory



        for index in reversed(indexes_to_pop):
            memory_values.pop(index)

        circular_buffer[joint_type] = memory_values



    def get_average_orientation(joint_type):
        global circular_buffer
        if not circular_buffer:
            return None  # No data available

        # Separate orientations and timestamps
        orientations, _ = zip(*circular_buffer[joint_type])

        # Calculate the average orientation
        average_orientation = np.mean(orientations, axis=0)
        return average_orientation




    #################### INIT NUITRACK ####################
    nuitrack = py_nuitrack.Nuitrack()
    nuitrack.init()

    devices = nuitrack.get_device_list()
    for i, dev in enumerate(devices):
        print(dev.get_name(), dev.get_serial_number())
        if i == 0:
            #dev.activate("ACTIVATION_KEY") #you can activate device using python api
            print(dev.get_activation())
            nuitrack.set_device(dev)
            print(nuitrack.get_version())
            print(nuitrack.get_license())
            nuitrack.create_modules()
            nuitrack.run()

    i = 0
    start_time = time.time()
    joint_avarage_time = 0.01
    start_postion = np.array([0,0,0], dtype=np.float32)
    translation = np.array([0,0,0], dtype=np.float32)

    while visualizer.poll_events():


        ######### Visualize ##########

        visualizer.clear_geometries()
        d = Texturize.create_star_model(poses, betas, trans)

        avatar = Texturize.create_star_mesh_obj(d, star_mesh, translation)

        #if i>0:
            #Texturize.calculate_new_scanned_verticies_positions(connection_inidices_of_star, transformed_scan_path, previous_star_verticies, d)
        #else:
         #   i+=1
        visualizer.add_geometry(avatar)
        visualizer.update_renderer()

        previous_star_verticies = d

        # GET NUITRACK JOINT VALUE
        key = cv2.waitKey(1)
        nuitrack.update()
        data = nuitrack.get_skeleton()
        for skel in data.skeletons:
            over_conf = 0
            under_conf = 0
            all_joints = all_nuitrack_joint_types.copy()
            if not np.array_equal([0,0,0],start_postion) :
                for el in skel[1:]:
                    if el.confidence > 0.7:
                        if el.type == py_nuitrack.JointType.waist:
                            translation = (start_postion - el.real)/1000

                        if el.type not in joints_to_ignore:
                            all_joints.remove(el.type)
                            joints = nuitrack_to_smpl[el.type]
                            axis_convert = nuitrack_to_smpl_axis[el.type]

                            for joint in joints:

                                # global orientation
                                orientation = convert_rot_mat_to_euler(el.orientation)
                                orientation = [axis_convert[0]*orientation[0], axis_convert[1]*orientation[1],
                                               axis_convert[2]*orientation[2]]

                                local_orientation = convert_to_relative_rotation(joint, orientation)
                                if el.type == py_nuitrack.JointType.left_shoulder or el.type == py_nuitrack.JointType.left_elbow:
                                    print(local_orientation)
                                    print(el.type)
                                add_data(local_orientation, el.type)
                                local_orientation = get_average_orientation(el.type)
                                poses = manipulate_joint(joint, local_orientation, poses)
                    else:
                        if el.type not in joints_to_ignore:
                            all_joints.remove(el.type)
                            joints = nuitrack_to_smpl[el.type]
                            axis_convert = nuitrack_to_smpl_axis[el.type]

                            for joint in joints:
                                orientation = convert_rot_mat_to_euler(el.orientation)
                                orientation = [0, 0,
                                               0]


                                add_data(orientation, el.type)
                                orientation = get_average_orientation(el.type)
                                poses = manipulate_joint(joint, orientation, poses)

            else:
                for el in skel[1:]:
                    if el.confidence > 0.7:
                        if np.array_equal([0,0,0],start_postion)  and el.type == py_nuitrack.JointType.waist:
                            start_postion = el.real