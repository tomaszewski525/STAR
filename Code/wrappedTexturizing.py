import open3d as o3d
import numpy as np
import torch
from scipy.spatial import Delaunay

def texture_mapping(mesh_A, mesh_B):
    # Compute face normals for Mesh A
    mesh_A.compute_triangle_normals()

    # Get vertices and triangle normals from Mesh A
    vertices_A = np.asarray(mesh_A.vertices)
    normals_A = np.asarray(mesh_A.triangle_normals)

    # Create KDTree for Mesh B for efficient searching
    kdtree_B = o3d.geometry.KDTreeFlann(mesh_B)

    # Initialize an empty texture for Mesh B
    texture_B = np.zeros((len(mesh_B.vertices), 2))
    print(np.asarray(mesh_B.triangles))

    vec_indexes = np.zeros((len(mesh_A.triangles),1))

    # Iterate over faces of Mesh A
    for i in range(len(mesh_A.triangles)):

        # Get the normal and a vertex from Mesh A
        normal_A = normals_A[i]
        vertex_A = vertices_A[mesh_A.triangles[i][0]]  # You can use any vertex from the triangle
        # Shoot a ray from the vertex along the normal
        _, index_B, _ = kdtree_B.search_knn_vector_3d(vertex_A + normal_A * 0.1, 1)

        vec_indexes[i] = index_B

        # zalozmy ze mam indeks face'a
        """
        for vector_index, num_matches in enumerate(matches_per_vector):
            if num_matches >1:
                print(f"Vector {vector_index}: {num_matches} matches")
        """

        """
        t = 0
        face_index = 0
        face_cords = []
        for vec in mesh_B.triangles:
            if index_B in vec:
                face_index = t
                face_cords = vec
                break
            t = t+1
        print(face_index)
        """

        #print(mesh_A.triangle_uvs[i]) # kordy tekstur

        # Use the intersection point to get the texture from Mesh A
        #texture_B[index_B[0]] = [mesh_A.triangle_uvs[i][0]]
    print(vec_indexes)
    print(np.asarray(mesh_B.triangles))
    indexes = torch.tensor(vec_indexes)
    vectors = torch.tensor(mesh_B.triangles)

    # Initialize an empty list to store the result vectors
    result_vectors = []

    # Convert the first structure to integers
    indexes = indexes.int()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indexes = indexes.to(device)
    vectors = vectors.to(device)

    # Iterate over each index
    for index in indexes:
        print(index)
        # Find the indices where the index is present in each vector
        indices = (vectors == index).nonzero(as_tuple=True)

        # Check if any match is found
        if indices[0].numel() > 0:
            # Extract the vectors corresponding to the indices
            result_vectors.append(vectors[indices[0]])
    print(result_vectors)

    #print(len(mesh_A.triangle_uvs))
    #print(np.asarray(mesh_A.triangle_uvs))
    #print(len(mesh_A.triangles))
    #print(np.asarray(mesh_A.triangles))


    # Assign the computed texture to Mesh B
    mesh_B.triangle_uvs = o3d.utility.Vector3dVector(texture_B)

# Load Mesh A and Mesh B
mesh_a = o3d.io.read_triangle_mesh("Data/michal_alligned.obj", True)
mesh_b = o3d.io.read_triangle_mesh("michal_star.ply")

# Texturize Mesh B with texture from Mesh A
result_mesh_b = texture_mapping(mesh_a, mesh_b)

# Save the result
o3d.io.write_triangle_mesh("path/to/result_mesh_b.obj", result_mesh_b)