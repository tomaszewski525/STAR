# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman
import matplotlib.pyplot as plt
import torch, os, csv
import numpy as np
from star.pytorch.star import STAR
from torch.autograd import Variable
import open3d as o3d
def get_vert_connectivity(num_verts, mesh_f):
    import scipy.sparse as sp
    vpv = sp.csc_matrix((num_verts,num_verts))
    def row(A):
        return A.reshape((1, -1))
    def col(A):
        return A.reshape((-1, 1))
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T
    return vpv

def get_verts_per_edge(num_verts,faces):
    import scipy.sparse as sp
    vc = sp.coo_matrix(get_vert_connectivity(num_verts, faces))
    def row(A):
        return A.reshape((1, -1))
    def col(A):
        return A.reshape((-1, 1))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]]
    return result

def edge_loss(d,smpl):
    vpe = get_verts_per_edge(6890,d.f)
    edges_for = lambda x: x[:,vpe[:,0],:] - x[:,vpe[:,1],:]
    edge_obj = edges_for(d) - edges_for(smpl)
    return edge_obj

def edge_loss_scan(star, scan):
    print(star)
    print(scan)

data = []
def verts_loss_scan(star_cp,scan_cp):
    # Calculate distances
    star_cp = star_cp
    scan_cp = scan_cp
    distances = torch.norm(star_cp[:, :, None, :] - scan_cp[:, None, :, :], dim=-1).float()


    # Find nearest neighbors
    #min_distances, min_indices = torch.min(distances, dim=-1)
    # Find k-nearest neighbors
    min_distances, min_indices = torch.topk(distances, k=1, largest=False, dim=-1)

    # Calculate total distance
    total_distance = torch.sum(min_distances)
    global data
    data.append(total_distance.item())
    print(total_distance)
    return total_distance

def verts_loss(d,smpl):
    return torch.sum((d-smpl)**2.0)

def v2v_loss(d,smpl):
    return torch.mean(torch.sqrt(torch.sum((d-smpl)**2.0,axis=-1)))


def convert_smpl_2_star(smpl,MAX_ITER_EDGES,MAX_ITER_VERTS,NUM_BETAS,GENDER):
    '''
        Convert SMPL meshes to STAR
    :param smpl:
    :return:
    '''

    #   In other words, it's assuming that smpl is a NumPy array or some other compatible data structure, and it's converting it to a PyTorch tensor residing on the GPU.
    smpl = torch.cuda.FloatTensor(smpl)
    batch_size = smpl.shape[0]

    if batch_size > 32:
        import warnings
        warnings.warn(
            'The Default optimization parameters (MAX_ITER_EDGES,MAX_ITER_VERTS) were tested on batch size 32 or smaller batches')

    # Init Star model
    star = STAR(gender=GENDER)

    # Init float tensor for star and wrap it with Variable for gradient calculations
    global_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    global_pose = Variable(global_pose, requires_grad=True)
    # 72 -3 becuase of joints number
    joints_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 72 - 3)))
    joints_pose = Variable(joints_pose, requires_grad=True)
    betas = torch.cuda.FloatTensor(np.zeros((batch_size, NUM_BETAS)))
    betas = Variable(betas, requires_grad=True)
    trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    trans = Variable(trans, requires_grad=True)

    # Optimizer parameters
    """
    Overall, it seems like this code is setting up an optimization process using the L-BFGS algorithm 
    to update the global_pose variable based on some objective function. The poses tensor is then formed
    by concatenating global_pose with another tensor, possibly for further use in the optimization process or subsequent computations."""

    learning_rate = 1e-1
    # Target of optimalization is global pose
    optimizer = torch.optim.LBFGS([global_pose], lr=learning_rate)
    poses = torch.cat((global_pose, joints_pose), 1)

    # Init star with torch
    d = star(poses, betas, trans)

    ########################################################################################################################
    # Fitting the model with an on edges objective first
    print('STAGE 1/2 - Fitting the Model on Edges Objective')
    for t in range(MAX_ITER_EDGES):

        # New star model
        poses = torch.cat((global_pose, joints_pose), 1)
        d = star(poses, betas, trans)

        # objective
        def edge_loss_closure():
            # square difference between edges
            loss = torch.sum(edge_loss_scan(d, smpl) ** 2.0)
            return loss

        # zero out gradients
        optimizer.zero_grad()

        # save gradients and calculate loss
        edge_loss_closure().backward()

        # next step, optimize global pose params
        optimizer.step(edge_loss_closure)

    # Optimize joints pose
    optimizer = torch.optim.LBFGS([joints_pose], lr=learning_rate)
    for t in range(MAX_ITER_EDGES):
        # New star model
        poses = torch.cat((global_pose, joints_pose), 1)
        d = star(poses, betas, trans)

        # objective function
        def edge_loss_closure():
            # square difference between edges
            loss = torch.sum(edge_loss(d, smpl) ** 2.0)
            return loss

        optimizer.zero_grad()
        edge_loss_closure().backward()
        optimizer.step(edge_loss_closure)

    ########################################################################################################################
    # Fitting the model with an on vertices objective
    print('STAGE 2/2 - Fitting the Model on a Vertex Objective')
    # Target of optimalization is joints_pose, global_pose, trans, betas
    optimizer = torch.optim.LBFGS([joints_pose, global_pose, trans, betas], lr=learning_rate)
    for t in range(MAX_ITER_VERTS):
        poses = torch.cat((global_pose, joints_pose), 1)
        d = star(poses, betas, trans)


        def vertex_closure():
            loss = torch.sum(verts_loss(d, smpl) ** 2.0)
            return loss

        optimizer.zero_grad()
        vertex_closure().backward()
        optimizer.step(vertex_closure)

    ########################################################################################################################
    np_poses = poses.detach().cpu().numpy()
    np_betas = betas.detach().cpu().numpy()
    np_trans = trans.detach().cpu().numpy()
    np_star_verts = d.detach().cpu().numpy()
    ########################################################################################################################

    return np_poses, np_betas, np_trans , np_star_verts

def convert_smplx_2_star(smplx,MAX_ITER_EDGES,MAX_ITER_VERTS,NUM_BETAS,GENDER):
    '''
        Convert SMPL-X meshes to STAR meshes

    :param smpl:
    :return:
    '''
    smplx = torch.cuda.FloatTensor(smplx)
    batch_size = smplx.shape[0]

    if batch_size > 32:
        import warnings
        warnings.warn('The Default optimization parameters (MAX_ITER_EDGES,MAX_ITER_VERTS) were tested on batch size 32 or smaller batches')

    star = STAR(gender=GENDER,num_betas=NUM_BETAS)

    global_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    global_pose = Variable(global_pose, requires_grad=True)

    joints_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 72 - 3)))
    joints_pose = Variable(joints_pose, requires_grad=True)

    betas = torch.cuda.FloatTensor(np.zeros((batch_size, NUM_BETAS)))
    betas = Variable(betas, requires_grad=True)

    trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    trans = Variable(trans, requires_grad=True)
    learning_rate = 1e-1
    optimizer = torch.optim.LBFGS([global_pose], lr=learning_rate)
    poses = torch.cat((global_pose, joints_pose), 1)
    d = star(poses, betas, trans)

    # Fitting the model with an on edges objective first
    print('STAGE 1/2 - Fitting the Model on Edges Objective')
    for t in range(MAX_ITER_EDGES):
        poses = torch.cat((global_pose, joints_pose), 1)
        d = star(poses, betas, trans)

        def edge_loss_closure():
            loss = torch.sum(edge_loss(d, smplx) ** 2.0)
            return loss

        optimizer.zero_grad()
        edge_loss_closure().backward()
        optimizer.step(edge_loss_closure)

    optimizer = torch.optim.LBFGS([joints_pose], lr=learning_rate)
    for t in range(MAX_ITER_EDGES):
        poses = torch.cat((global_pose, joints_pose), 1)
        d = star(poses, betas, trans)
        def edge_loss_closure():
            loss = torch.sum(edge_loss(d, smplx) ** 2.0)
            return loss
        optimizer.zero_grad()
        edge_loss_closure().backward()
        optimizer.step(edge_loss_closure)
    ########################################################################################################################
    # Fitting the model with an on vertices objective
    print('STAGE 2/2 - Fitting the Model on a Vertex Objective')
    optimizer = torch.optim.LBFGS([joints_pose, global_pose, trans, betas], lr=learning_rate)
    for t in range(MAX_ITER_VERTS):
        poses = torch.cat((global_pose, joints_pose), 1)
        d = star(poses, betas, trans)
        def vertex_closure():
            loss = torch.sum(verts_loss(d, smplx) ** 2.0)
            return loss
        optimizer.zero_grad()
        vertex_closure().backward()
        optimizer.step(vertex_closure)

    np_poses = poses.detach().cpu().numpy()
    np_betas = betas.detach().cpu().numpy()
    np_trans = trans.detach().cpu().numpy()
    np_star_verts = d.detach().cpu().numpy()

    return np_poses, np_betas, np_trans , np_star_verts , star.f



def convert_scan_2_star(smpl, star_poses_file_path, MAX_ITER_EDGES,MAX_ITER_VERTS,NUM_BETAS,GENDER):
    '''
        Convert SMPL meshes to STAR
    :param smpl:
    :return:
    '''

    # In other words, it's assuming that smpl is a NumPy array or some other compatible data structure, and it's converting it to a PyTorch tensor residing on the GPU.
    smpl = torch.cuda.FloatTensor(smpl)
    batch_size = smpl.shape[0]

    if batch_size > 32:
        import warnings
        warnings.warn(
            'The Default optimization parameters (MAX_ITER_EDGES,MAX_ITER_VERTS) were tested on batch size 32 or smaller batches')

    # Specify the path to your .npy file
    poses_data = np.load(star_poses_file_path)
    poses = torch.cuda.FloatTensor(poses_data)

    #poses = torch.cuda.FloatTensor(np.zeros((batch_size, 72)))
    poses = Variable(poses, requires_grad=True)
    # Init Star model
    star = STAR(gender=GENDER)
    all_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 72-3)))
    all_pose = Variable(all_pose, requires_grad=True)
    # Init float tensor for star and wrap it with Variable for gradient calculations
    hands_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 18)))
    hands_pose = Variable(hands_pose, requires_grad=True)
    shoulders_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 6)))
    shoulders_pose = Variable(shoulders_pose, requires_grad=True)

    global_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    global_pose = Variable(global_pose, requires_grad=True)
    # 72 -3 becuase of joints number
    joints_pose = torch.cuda.FloatTensor(np.zeros((batch_size, 72 - 27)))
    joints_pose = Variable(joints_pose, requires_grad=True)
    betas = torch.cuda.FloatTensor(np.zeros((batch_size, NUM_BETAS)))
    betas = Variable(betas, requires_grad=True)
    trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    trans = Variable(trans, requires_grad=True)

    scale = torch.cuda.FloatTensor(np.ones((batch_size, 3)))
    scale = Variable(scale, requires_grad=True)

    # Optimizer parameters
    """
    Overall, it seems like this code is setting up an optimization process using the L-BFGS algorithm 
    to update the global_pose variable based on some objective function. The poses tensor is then formed
    by concatenating global_pose with another tensor, possibly for further use in the optimization process or subsequent computations."""

    learning_rate = 0.01

    # Target of optimalization is global pose

    # Init star with torch
    d = star(poses, betas, trans)

    ########################################################################################################################
    # Fitting the model with an on edges objective first

    ########################################################################################################################
    # Fitting the model with an on vertices objective
    print('STAGE 2/2 - Fitting the Model on a Vertex Objective')
    # Target of optimalization is joints_pose, global_pose, trans, betas

    optimizer = torch.optim.Adam([poses, betas, trans, scale], lr=learning_rate)

    global data
    for t in range(MAX_ITER_VERTS):
        d = star(poses, betas, trans)

        #print(d)
        #print(scale)
        print(t)
        print(MAX_ITER_VERTS)
        #print(poses)
        #print(trans)
        #print(data)
        def vertex_closure():
            loss = torch.sum(verts_loss_scan(d*scale, smpl))
            return loss

        optimizer.zero_grad()
        vertex_closure().backward()
        optimizer.step(vertex_closure)

    # Create a plot
    #plt.plot(data)

    # Add labels and title
    #plt.xlabel('Iteracja')
    #plt.ylabel('Koszt')
    #plt.title('Optymalizacja')

    # Display the plot
    #plt.show()

    ########################################################################################################################
    np_poses = poses.detach().cpu().numpy()
    np_betas = betas.detach().cpu().numpy()
    np_trans = trans.detach().cpu().numpy()
    np_star_verts = d.detach().cpu().numpy()
    np_scale = scale.detach().cpu().numpy()
    ########################################################################################################################

    return np_poses, np_betas, np_trans , np_star_verts, np_scale