import open3d as o3d
import numpy as np
mesh_a = o3d.io.read_triangle_mesh("Data/michal_alligned.obj", True)

def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()

        return False

    key_to_callback = {}
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=480)

for i in range(1,6):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        print(ctr)
        
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([mesh_a],
                                                              rotate_view)


#custom_draw_geometry_with_rotation(mesh_a)