import subprocess

from PyNuitrack import py_nuitrack
import time
import sys, os
sys.path.append(os.getcwd())
import torch
import numpy as np

from scipy.spatial.transform import Rotation
import cv2
import open3d as o3d
import Texturize
import FindTransform
import ChangeOptimPose
import convert_scan_to_star
from avatar_creator import AvatarCreator

scan_path = "Data/michal_alligned.obj"
transformed_scan_path = "Data/michal_alligned.obj"
#transformed_scan_path = "Data/transformed_mesh.ply"

star_default_poses_path = "Data/star_poses.npy"
star_scan_poses_bef_path = "Data/manipulated_star_poses.npy"
star_scan_poses_aft_path='Data/michal_skan_star.npy'
texturized_scan_poses_save_path = 'Data/michal_skan_star_colored.npy'

avatar_creator = AvatarCreator(scan_path, transformed_scan_path, star_scan_poses_bef_path)

allign_model = False
change_optim_pose = False
optimize_model = True
texturize_model = False
visualize_model = False
apply_joints = False

attach_point_cloud_to_mesh = True

if allign_model:
    avatar_creator.allign_scan_to_star(scan_path, star_default_poses_path, transformed_scan_path)

if change_optim_pose:
    avatar_creator.change_manually_star_pose(star_scan_poses_bef_path, transformed_scan_path)

if optimize_model:
    avatar_creator.optimize_star_to_scan(transformed_scan_path, star_scan_poses_bef_path, star_scan_poses_aft_path)

if texturize_model:
    k = 2
    #Texturize.texturize_scan_nnn(star_scan_poses_aft_path, transformed_scan_path, texturized_scan_poses_save_path)
    Texturize.texturize_scan_k_nnn(star_scan_poses_aft_path, transformed_scan_path, texturized_scan_poses_save_path, k)


connection_inidices_of_star = Texturize.attach_pointcloud_to_scan(star_scan_poses_aft_path, transformed_scan_path)

# 1. Repair scan
# 2. Optimize model
# 3. Texturize model
# 4. Apply Joints rotations

if apply_joints:
    #################### OPTIMIZE MODEL ############################

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

    joints_to_ignore =[py_nuitrack.JointType.left_wrist, py_nuitrack.JointType.right_wrist
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
        py_nuitrack.JointType.head:[1, 1, 1],
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


    BUFFER_SIZE = 150
    TIME_THRESHOLD = 1.0



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
    joint_avarage_time = 0.5
    start_postion = np.array([0,0,0], dtype=np.float32)
    translation = np.array([0,0,0], dtype=np.float32)

    while visualizer.poll_events():


        ######### Visualize ##########
        #print(poses)
        visualizer.clear_geometries()
        d = Texturize.create_star_model(poses, betas, trans)

        avatar = Texturize.create_star_mesh(d, star_colors, translation)
        if i>0:
            Texturize.calculate_new_scanned_verticies_positions(connection_inidices_of_star, transformed_scan_path, previous_star_verticies, d)
        else:
            i+=1
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

                                orientation = convert_rot_mat_to_euler(el.orientation)
                                orientation = [axis_convert[0]*orientation[0], axis_convert[1]*orientation[1],
                                               axis_convert[2]*orientation[2]]


                                add_data(orientation, el.type)
                                orientation = get_average_orientation(el.type)
                                poses = manipulate_joint(joint, orientation, poses)
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