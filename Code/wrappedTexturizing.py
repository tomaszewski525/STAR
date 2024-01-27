import open3d as o3d
import numpy as np
import torch
from scipy.spatial import Delaunay
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import LineString
import Texturize
def texture_mapping(mesh_A, mesh_B):
    # Compute face normals for Mesh A
    mesh_A.compute_triangle_normals()
    mesh_B.compute_triangle_normals()

    # Get vertices and triangle normals from Mesh A
    vertices_A = np.asarray(mesh_A.vertices)
    normals_A = np.asarray(mesh_A.triangle_normals)

    # Create KDTree for Mesh B for efficient searching
    kdtree_B = o3d.geometry.KDTreeFlann(mesh_B)

    # Initialize an empty texture for Mesh B
    texture_B = np.zeros((len(mesh_B.triangles)*3, 2))


    cube = o3d.t.geometry.TriangleMesh.from_legacy(mesh_B)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)



    rays = []
    #rays.append(mesh_B)
    #rays.append(mesh_A)
    # Iterate over faces of Mesh A\
    hitted_triangles = 0
    not_hitted_triangles = 0
    for i in range(len(mesh_A.triangles)):
        # Get the normal and a vertex from Mesh A
        normal_A = normals_A[i]
        vertex_A = vertices_A[mesh_A.triangles[i][0]]  # You can use any vertex from the triangle


        # Shoot a ray from the vertex along the normal
        s, index_B, t = kdtree_B.search_knn_vector_3d(vertex_A - normal_A * 0.1, 1)

        ray_line = o3d.geometry.LineSet()
        ray_line.points = o3d.utility.Vector3dVector([vertex_A, vertex_A - normal_A * 0.1])
        ray_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        ray_line.paint_uniform_color([1, 0, 0])  # Red color for rays

        # Add nth ray to visualization
       # if i%100==0:
           # rays.append(ray_line)



        point = np.asarray(mesh_B.vertices)[index_B]
        #print(point)
        normal = np.asarray(mesh_B.triangle_normals)[index_B]
        epsilon = 1e-6
        ray_d = np.hstack((point + epsilon * normal, -normal))
        ray = o3d.core.Tensor([ray_d], dtype=o3d.core.Dtype.Float32)
        # do ray casting with the ray
        primitive_id = scene.cast_rays(ray)["primitive_ids"].item()

        if primitive_id < 100000:
            # Trójkat ktory jest najblizej mesha
            t = 0
            hit_vertices = []
            for point in mesh_A.triangles[i]:
                #hit_vertices.append(mesh_B.vertices[point])
                texture_B[primitive_id*3+t] = mesh_A.triangle_uvs[point]
                t = t + 1

            # Create a new mesh for the hit triangles
            hit_mesh = o3d.geometry.TriangleMesh()
            hit_mesh.vertices = o3d.utility.Vector3dVector(hit_vertices)
            hit_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])

            # Paint the hit mesh green
            hit_mesh.paint_uniform_color([0, 1, 0])
            if i % 5 == 0:
                #print(ray_line)
                #print(hit_mesh)
                rays.append(hit_mesh)
                #rays.append(ray_line)

            hitted_triangles = hitted_triangles+1
        else:
            not_hitted_triangles = not_hitted_triangles + 1

    print(len(rays))

    ######### VISUALIZE HITS ###################
    # Visualization setup
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    # Add Mesh A to the visualization
    #rays.append(mesh_B)
    #o3d.visualization.draw_geometries(rays)


    # Assign the computed texture to Mesh B
    print(o3d.utility.Vector2dVector(texture_B))
    print(mesh_A.triangle_uvs)
    mesh_B.triangle_uvs = o3d.utility.Vector2dVector(texture_B)
    mesh_B.textures = mesh_A.textures
    return mesh_B, mesh_A

def transfer_texture(mesh_A, mesh_B):
    mesh_A.compute_triangle_normals()
    normals_A = np.asarray(mesh_A.triangle_normals)

    for i in range(len(mesh_A.triangles)):
        print(i)
        tri_indices_A = mesh_A.triangles[i]
        vertices_A = []

        normal = mesh_A.triangle_normals[i]
        for indice in tri_indices_A:
            vertices_A.append(mesh_A.vertices[indice])


        # Create a Shapely LineString for the ray
        ray = LineString([vertices_A[0], vertices_A[0] - normal*100])

        for i in range(len(mesh_B.triangles)):
            tri_indices_B = mesh_B.triangles[i]
            vertices_B = []
            for indice in tri_indices_B:
                vertices_B.append(mesh_B.vertices[indice])
            # Create a Shapely Polygon for the triangle
            triangle = Polygon(vertices_B)
            # Check for intersection
            if ray.intersects(triangle):

                #intersection_point = ray.intersection(triangle)
                print("Intersection")
                break;


def texture_mapping_inverted(scanned_mesh, star_mesh):

    # Compute face normals for Mesh A
    scanned_mesh.compute_triangle_normals()
    star_mesh.compute_triangle_normals()

    # Get vertices and triangle normals from Mesh A
    star_verticies = np.asarray(star_mesh.vertices)
    star_normals = np.asarray(star_mesh.triangle_normals)

    # creatae sacnned_mesh
    kdtree_B = o3d.geometry.KDTreeFlann(scanned_mesh)

    # Initialize an empty texture for Mesh B
    texture_B = np.zeros((len(star_mesh.triangles)*3, 2))


    cube = o3d.t.geometry.TriangleMesh.from_legacy(scanned_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    rays = []
    hitted_triangles = 0
    not_hitted_triangles = 0
    baden_trojkaty = [0, 100, 2, 4, 6, 8]
    for i in range(len(star_mesh.triangles)):
        star_verts = []
        for p in range(len(star_mesh.triangles[i])):


            # Get the normal and a vertex from Mesh A
            star_normal = star_normals[i]
            star_vertex = star_verticies[star_mesh.triangles[i][p]] # for each vertex in triangle
            star_verts.append(star_vertex)
            # Shoot a ray from the vertex along the normal
            s, index_B, t = kdtree_B.search_knn_vector_3d(star_vertex + star_normal * 0.1, 1)

            ray_line = o3d.geometry.LineSet()
            ray_line.points = o3d.utility.Vector3dVector([star_vertex, star_vertex - star_normal * 0.1])
            ray_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            ray_line.paint_uniform_color([1, 0, 0])  # Red color for rays

            # Get closest point
            #point = np.asarray(scanned_mesh.vertices)[index_B]
            #normal = np.asarray(scanned_mesh.triangle_normals)[index_B]
            epsilon = 1e-6
            ray_d = np.hstack((star_vertex + star_normal * 1, -star_normal))
            ray = o3d.core.Tensor([ray_d], dtype=o3d.core.Dtype.Float32)
            # do ray casting with the ray
            primitive_id = scene.cast_rays(ray)["primitive_ids"].item()
            threshold_text_distance = 0.08

            if primitive_id < 100000:
                # Trójkat ktory jest najblizej mesha
                t = 0
                vet = []
                for point in scanned_mesh.triangles[primitive_id]:
                    vet.append(scanned_mesh.vertices[point])
                    # if index_B == point:
                    if p == 0:
                        texture_B[i * 3 + p] = scanned_mesh.triangle_uvs[point]
                    if p == 1:
                        tex_1 = texture_B[i * 3 + 0]
                        distance = np.linalg.norm(tex_1 - scanned_mesh.triangle_uvs[point])
                        if distance > threshold_text_distance:
                            texture_B[i * 3 + p] = tex_1
                        else:
                            texture_B[i * 3 + p] = scanned_mesh.triangle_uvs[point]
                    if p == 2:
                        tex_2 = texture_B[i * 3 + 1]
                        distance = np.linalg.norm(tex_2 - scanned_mesh.triangle_uvs[point])
                        if distance > threshold_text_distance:
                            texture_B[i * 3 + p] = tex_2
                        else:
                            texture_B[i * 3 + p] = scanned_mesh.triangle_uvs[point]

                # Create a new mesh for the hit triangles
                hit_mesh = o3d.geometry.TriangleMesh()
                hit_mesh.vertices = o3d.utility.Vector3dVector(vet)
                hit_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])

                # Paint the hit mesh green
                hit_mesh.paint_uniform_color([0, 1, 0])
                if i  in baden_trojkaty:
                    # print(ray_line)
                    # print(hit_mesh)
                    rays.append(hit_mesh)
                    rays.append(ray_line)

                hitted_triangles = hitted_triangles+1
            else:
                ray_d = np.hstack((star_vertex - star_normal * 1, -star_normal))
                ray = o3d.core.Tensor([ray_d], dtype=o3d.core.Dtype.Float32)
                # do ray casting with the ray
                primitive_id = scene.cast_rays(ray)["primitive_ids"].item()
                if primitive_id < 100000:
                    # Trójkat ktory jest najblizej mesha
                    t = 0
                    vet = []
                    for point in scanned_mesh.triangles[primitive_id]:
                        vet.append(scanned_mesh.vertices[point])
                        # if index_B == point:
                        if p == 0:
                            texture_B[i * 3 + p] = scanned_mesh.triangle_uvs[point]
                        if p == 1:
                            tex_1 = texture_B[i * 3 + 0]
                            distance = np.linalg.norm(tex_1 - scanned_mesh.triangle_uvs[point])
                            if distance > threshold_text_distance:
                                texture_B[i * 3 + p] = tex_1
                            else:
                                texture_B[i * 3 + p] = scanned_mesh.triangle_uvs[point]
                        if p == 2:
                            tex_2 = texture_B[i * 3 + 1]
                            distance = np.linalg.norm(tex_2 - scanned_mesh.triangle_uvs[point])
                            if distance > threshold_text_distance:
                                texture_B[i * 3 + p] = tex_2
                            else:
                                texture_B[i * 3 + p] = scanned_mesh.triangle_uvs[point]
                    # Create a new mesh for the hit triangles
                    hit_mesh = o3d.geometry.TriangleMesh()
                    hit_mesh.vertices = o3d.utility.Vector3dVector(vet)
                    hit_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])

                    # Paint the hit mesh green
                    hit_mesh.paint_uniform_color([0, 1, 0])
                    if i  in baden_trojkaty:
                        # print(ray_line)
                        # print(hit_mesh)
                        rays.append(hit_mesh)
                        rays.append(ray_line)

                    hitted_triangles = hitted_triangles + 1

        # Create a new mesh for the hit triangles
        hit_mesh = o3d.geometry.TriangleMesh()
        hit_mesh.vertices = o3d.utility.Vector3dVector(star_verts)
        hit_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])

        # Paint the hit mesh green
        hit_mesh.paint_uniform_color([1, 0, 0])
        if i  in baden_trojkaty:
            # print(ray_line)
            # print(hit_mesh)
            rays.append(hit_mesh)

    # Add Mesh A to the visualization
    # rays.append(mesh_B)
    o3d.visualization.draw_geometries(rays)

    star_mesh.triangle_uvs = o3d.utility.Vector2dVector(texture_B)
    star_mesh.textures = scanned_mesh.textures
    return star_mesh, scanned_mesh


# Load Mesh A and Mesh B
mesh_a = o3d.io.read_triangle_mesh("Data/Michal_Low_Poly_alligned.obj", True)
mesh_b = o3d.io.read_triangle_mesh("michal_star.ply")

# Load the NumPy structured array from a .npy file
point_cloud_data = np.load('Data/michal_skan_star.npy', allow_pickle=True).item()

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
star_mesh = Texturize.create_star_mesh_no_color_scale(d, trans, scale)




star_mesh.triangle_material_ids = mesh_a.triangle_material_ids

#print(len(mesh_b.triangle_material_ids))

# Texturize Mesh B with texture from Mesh A
result_mesh_b, mesh_A = texture_mapping_inverted(mesh_a, star_mesh)
print(result_mesh_b)
result_mesh_b.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(result_mesh_b.triangles), dtype=np.int32))
o3d.visualization.draw_geometries([result_mesh_b])
# Save the result
#o3d.io.write_triangle_mesh("path/to/result_mesh_b.obj", result_mesh_b)






