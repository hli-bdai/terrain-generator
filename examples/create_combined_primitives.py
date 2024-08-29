import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import inspect
import argparse
import copy

from terrain_generator.wfc.wfc import WFCSolver

from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import create_mesh_pattern, create_mesh_tile
from terrain_generator.utils import visualize_mesh
from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import MeshPartsCfg, MeshPattern

from terrain_generator.trimesh_tiles.primitive_course.steps import *


def generate_tiles(
    cfgs,
    mesh_name="result_mesh.stl",
    mesh_dir="results/result",
    height_offset=0.0,
    visualize=False  
):    
    floating_mesh = trimesh.Trimesh()
    result_mesh = trimesh.Trimesh()

    x_offset = 0.0
    y_offset = 0.0   
    mesh_id = 0
    for mesh_cfg in cfgs:        
        tile = create_mesh_tile(mesh_cfg)
        mesh = tile.get_mesh().copy()         
        dim = mesh_cfg.dim
        if mesh_id > 0:
            y_offset += dim[1]/2.0
        xy_offset = np.array([x_offset, y_offset, 0.0])
        mesh.apply_translation(xy_offset)
        if "floating" in mesh_cfg.name:
            floating_mesh += mesh
        else:
            result_mesh += mesh        
        
        mesh_id += 1
        y_offset += dim[1]/2.0

    # Rotation about z-axis for -90 degree for the exported mesh
    result_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2.0, [0, 0, 1]))
    min_bound = result_mesh.bounds[0]
    max_bound = result_mesh.bounds[1]
    center_x = min_bound[0]+(max_bound[0]-min_bound[0])/2.0
    center_offset = -np.array([center_x, 0.0, min_bound[2]-height_offset])
    result_mesh.apply_translation(center_offset)

    os.makedirs(mesh_dir, exist_ok=True)
    print("saving mesh to ", mesh_name)
    result_mesh.export(os.path.join(mesh_dir, mesh_name))
    if floating_mesh.vertices.shape[0] > 0:
        # get name before extension
        name, ext = os.path.splitext(mesh_name)
        if visualize:
            visualize_mesh(floating_mesh)
        floating_mesh.export(os.path.join(mesh_dir, name + "_floating" + ext))
    if visualize:
        visualize_mesh(result_mesh)   

def generate_stairs(dim, level, mesh_dir, stair_type='straignt_up', visualize=False):    
    # * types = {straight_up, straight_down, pyramid_up, pyramid_down, up_down, down_up, plus_up, plus_down}
    height_diff = level * 2.0

    # * sets the number of stairs by size of the array
    num_stairs = 7        

    cfgs = create_stairs(
        MeshPartsCfg(dim=dim), height_diff=height_diff, num_stairs=num_stairs, type=stair_type, noise=0.0)
    mesh_dir = os.path.join(mesh_dir, inspect.currentframe().f_code.co_name)
    generate_tiles(cfgs, mesh_name=f"mesh_{level:.1f}.obj", mesh_dir=mesh_dir, visualize=visualize)
    return cfgs

def generate_inclined_surfaces(dim, level, mesh_dir, height=0.0, visualize=False):
    width = 0.3                     # width of each block
    side_distance = 0.4             # distane between two blocks in the side direction
    side_std = 0.0
    height_std = 0.0
    incline_angle = np.pi/6.0       # angle of inclined surfaces
    cfgs = create_inclined_surfaces(MeshPartsCfg(dim=dim), width=width, height=height, side_distance=side_distance, 
                         side_std=side_std, height_std=height_std, angle=incline_angle)    
    mesh_dir = os.path.join(mesh_dir, inspect.currentframe().f_code.co_name)    
    generate_tiles(cfgs, mesh_name=f"mesh_{level:.1f}.obj", mesh_dir=mesh_dir, visualize=visualize)
    return cfgs

def generate_overhanging_boxes(dim, level, mesh_dir, height=0.0, visualize=False):
    cfgs = create_two_overhanging_boxes(MeshPartsCfg(dim=dim), height_offset=height, side_std=0.0, height_std=0.0, aligned=True)    
    mesh_dir = os.path.join(mesh_dir, inspect.currentframe().f_code.co_name)    
    generate_tiles(cfgs, mesh_name=f"mesh_{level:.1f}.obj", mesh_dir=mesh_dir, visualize=visualize)
    return cfgs
   
def generate_single_beam(dim, level, mesh_dir, height=0.0, visualize=False):
    width = 0.3
    thickness = 0.1
    box_dims = [width, 0.9*dim[1], thickness]

    transformations = []
    pitch = np.pi/15.0
    t = trimesh.transformations.rotation_matrix(pitch, [1, 0, 0])
    t[:3, -1] = np.array([0.0, 0.0, height])
    transformations.append(t)
    
    return (BoxMeshPartsCfg(
            name="beam",
            dim=dim,
            transformations=tuple(transformations),            
            box_dims=tuple(box_dims),            
            rotations=(15, 0.0, 0.0),            
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
        ),
    )
    
def concatenate_configs(list_of_configs : list) -> list: 
    cfgs_res = []    
    for cfgs_ in list_of_configs: 
        # each cfgs_ is a tuple
        cfgs = list(cfgs_)
        if len(cfgs) > 2:            
            cfgs.pop(0)
            cfgs.pop(-1)
        cfgs_res = cfgs_res + cfgs
    return tuple(cfgs_res)

def generate_complex_terrain(dim, level, mesh_dir, visualize=False):
    height = 1.5
    floor_cfgs = (
        PlatformMeshPartsCfg(
            name="floor",
            dim=dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    dim_w_longer_y = np.array(dim) + np.array([0.0,0.5,0.0])
    ascend_stair_cfgs = generate_stairs(dim_w_longer_y, height/2.0, stair_type='straight_up', mesh_dir=mesh_dir)
    descend_stair_cfgs = generate_stairs(dim_w_longer_y, height/2.0, stair_type='straight_down', mesh_dir=mesh_dir)
    inclined_surface_cfgs = generate_inclined_surfaces(dim_w_longer_y, level, mesh_dir)
    overhang_cfgs = generate_overhanging_boxes(dim, level, mesh_dir, height)
    beam_cfgs = generate_single_beam(dim, level, mesh_dir, height+0.2)

    cfgs_list1 = [floor_cfgs, inclined_surface_cfgs, ascend_stair_cfgs, overhang_cfgs, 
                 descend_stair_cfgs, floor_cfgs]    
    cfgs_list2 = [floor_cfgs, inclined_surface_cfgs, ascend_stair_cfgs, beam_cfgs, 
                 descend_stair_cfgs, floor_cfgs]    
    
    cfgs_res = concatenate_configs(cfgs_list1)
    
    mesh_dir = os.path.join(mesh_dir, inspect.currentframe().f_code.co_name)
    generate_tiles(cfgs_res, mesh_name="mesh_complex_terrain.obj", mesh_dir=mesh_dir, height_offset=1.0, visualize=visualize)
    

    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create some primitive patches")
    parser.add_argument(
        "--visualize", type=bool, default=False, help="Visualizing the primitives"
    )
    args = parser.parse_args()


    dim = (2.0, 3.0, 3.0)
    level = 1.0
    mesh_dir = "results/primitive_separated"    
        
    # generate_inclined_surfaces(dim, level, mesh_dir, args.visualize)
    # generate_stairs(dim, level, mesh_dir, args.visualize)
    # generate_overhanging_boxes(dim, level, mesh_dir, args.visualize)
    generate_complex_terrain(dim, level, mesh_dir, args.visualize)