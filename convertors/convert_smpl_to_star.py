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
import numpy as np
import os
from losses import convert_smpl_2_star
import open3d as o3d
########################################################################################################################
path_smpl_meshes = ''      #Path SMPL Meshes, a numpy array of SMPL verticies (batch_size x 6890 x 3)
path_save_star_parms = 'dupa.npy' #Path to save the star paramters

star_gender = 'male'   #STAR Model Gender (options: male,female,neutral).
MAX_ITER_EDGES = 100    #Number of LBFGS iterations for an on edges objective
MAX_ITER_VERTS = 1500   #Number of LBFGS iterations for an on vertices objective
NUM_BETAS = 10



opt_parms = {'MAX_ITER_EDGES':MAX_ITER_EDGES ,
             'MAX_ITER_VERTS':MAX_ITER_VERTS,
             'NUM_BETAS':NUM_BETAS,
             'GENDER':star_gender}
########################################################################################################################
print('Loading the SMPL Meshes and ')

# Load PLY file
ply_file_path = "michal_skan.ply"
point_cloud = o3d.io.read_point_cloud(ply_file_path)
mesh = o3d.io.read_triangle_mesh(ply_file_path)
vertices = np.asarray(mesh.vertices)
smpl = vertices[np.newaxis, :]


np_poses , np_betas , np_trans , star_verts = convert_smpl_2_star(smpl,**opt_parms)
results = {'poses':np_poses,'betas':np_betas,'trans':np_trans,'star_verts':star_verts}
print('Saving the results %s.'%(path_save_star_parms))
np.save(path_save_star_parms,results)

