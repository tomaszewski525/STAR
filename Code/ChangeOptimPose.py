import numpy as np
from scipy.spatial.transform import Rotation
from star.pytorch.star import STAR
import numpy as np
from numpy import newaxis
import pickle
import os
import torch
import open3d as o3d
import keyboard

class ModelManipulator:
    def __init__(self, star_poses_save_file="star_poses.npy", scan_file_path="transformed_mesh.ply"):

        ######### Create STAR body model for icp transformation estimation #########
        self.star = STAR(gender='male')
        self.betas = np.array([
            np.array([0, 0, 0, 0,
                      0, 0, 0, 0,
                      0, 0])])
        self.num_betas = 10
        self.batch_size = 1
        self.m = STAR(gender='male', num_betas=self.num_betas)

        # Zero pose
        self.poses = torch.cuda.FloatTensor(np.zeros((self.batch_size, 72)))
        self.betas = torch.cuda.FloatTensor(self.betas)

        self.trans = torch.cuda.FloatTensor(np.zeros((self.batch_size, 3)))
        self.model = self.star.forward(self.poses, self.betas, self.trans)
        self.shaped = self.model.v_shaped[-1, :, :]
        self.star_verticies = self.model.squeeze().cpu().numpy()

        ######### Convert star to point cloud #########
        self.star_pc = o3d.geometry.PointCloud()
        self.star_pc.points = o3d.utility.Vector3dVector(self.star_verticies)

        self.scan_mesh = o3d.io.read_triangle_mesh(scan_file_path)
        self.scan_vertices = np.asarray(self.scan_mesh.vertices)
        self.scan_pc = o3d.geometry.PointCloud()
        self.scan_pc.points = o3d.utility.Vector3dVector(self.scan_vertices)

        self.pressed_number = ""
        self.axis = ""
        self.axis_new_value = 0


    def viusalize(self):
        # Create a visualization window
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()

        # Add the point cloud to the visualization
        visualizer.add_geometry(self.scan_mesh)
        visualizer.add_geometry(self.star_pc)

        def on_key_event(e):

            if e.name != "enter" and str(e.event_type) == "down" and e.name in ["1", "2", "3", "4", "5", "6", "7", "8",
                                                                                "9", "0"]:
                self.pressed_number = self.pressed_number + e.name
            if e.name == "enter":
                self.pressed_number = ""
            if e.name == "s":
                # Convert PyTorch tensor to NumPy array
                poses_np = self.poses.cpu().numpy()

                # Save the NumPy array to a .npy file
                # file_path_npy = 'C:/Users/tfran/Desktop/Inzynierka/STAR/STAR/Code/star_poses.npy'
                np.save(self.star_poses_save_file, poses_np)

            if e.name in ["x", "y", "z"]:
                self.axis = e.name
            num = 0
            if e.name == "p":
                print("add")
                if int(self.pressed_number) < 23:
                    if self.axis == "x":
                        num = 0
                    if self.axis == "y":
                        num = 1
                    if self.axis == "z":
                        num = 2
                    self.axis_new_value += 0.05
                    # Indices of the values you want to update
                    indices_to_update = [(int(self.pressed_number)) * 3 + num]
                    # New values to assign
                    new_values = torch.cuda.FloatTensor(np.arange(1))
                    # Update specific values
                    self.poses[:, indices_to_update] = self.axis_new_value

            if e.name == "o":
                if int(self.pressed_number) < 23:
                    if int(self.pressed_number) < 23:
                        if self.axis == "x":
                            num = 0
                        if self.axis == "y":
                            num = 1
                        if self.axis == "z":
                            num = 2
                        self.axis_new_value -= 0.05
                        # Indices of the values you want to update
                        indices_to_update = [(int(self.pressed_number)) * 3 + num]

                        # New values to assign
                        new_values = torch.cuda.FloatTensor(np.arange(1))

                        # Update specific values
                        self.poses[:, indices_to_update] = self.axis_new_value

            print(e.name)

        # Hook for key events
        keyboard.hook(on_key_event)

        # Run the visualization loop
        while visualizer.poll_events():
            print(self.pressed_number)
            model = self.star.forward(self.poses, self.betas, self.trans)
            shaped = model.v_shaped[-1, :, :]
            star_verticies = model.squeeze().cpu().numpy()

            ######### Convert star to point cloud #########
            self.star_pc = o3d.geometry.PointCloud()
            self.star_pc.points = o3d.utility.Vector3dVector(star_verticies)

            visualizer.clear_geometries()

            # Add the point cloud to the visualization
            visualizer.add_geometry(self.scan_mesh)
            visualizer.add_geometry(self.star_pc)
            visualizer.update_renderer()
            # Other processing can be done here




