import open3d as o3d

def convert_ply_to_obj(input_ply_path, output_obj_path):
    # Load PLY file
    mesh = o3d.io.read_triangle_mesh(input_ply_path)

    # Save as OBJ file
    o3d.io.write_triangle_mesh(output_obj_path, mesh)


def convert_ply_to_obj_material(input_ply_path, output_obj_path, output_mtl_path):
    # Load PLY file
    mesh = o3d.io.read_triangle_mesh(input_ply_path)

    # Create a basic material
    material = o3d.geometry.TriangleMesh.create_box().materials[0]

    # Save as OBJ file with embedded material information
    o3d.io.write_triangle_mesh(output_obj_path, mesh, write_materials=True)

    # Save a separate MTL file (you may customize this based on your material properties)
    with open(output_mtl_path, 'w') as mtl_file:
        mtl_file.write('newmtl material_name\n')
        mtl_file.write('Ka 1.000 1.000 1.000\n')
        mtl_file.write('Kd 1.000 1.000 1.000\n')
        mtl_file.write('Ks 0.000 0.000 0.000\n')
        mtl_file.write('Ns 0\n')
        mtl_file.write('illum 2\n')


# Replace 'input_file.ply', 'output_file.obj', and 'output_file.mtl' with your actual file paths
convert_ply_to_obj_material('Data/transformed_mesh.ply', 'output_file.obj', 'output_file.mtl')


# Replace 'input_file.ply' and 'output_file.obj' with your actual file paths
#convert_ply_to_obj('Data/transformed_mesh.ply', 'output_file.obj')