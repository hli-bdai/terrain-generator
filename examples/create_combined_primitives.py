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
from export_heightmap import mesh_to_heightmap
import yaml
import itertools

  
def generate_mesh(cfgs, height_offset=0.0, visualize=False) -> trimesh.Trimesh : 
    """ Generate mesh files
        params [in] : cfgs:  tile configurations
                      height_offset : height offset you would like to shift the generated mesh (meter)
                      visualize: option to visualize the generated mesh
        return : generated mesh (Trimesh)
    """ 
    
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
    # Why do this? Align the coordinate frame with the depth image frame (x-axis: screen bottom to up. y-axis: screen left to right)
    result_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2.0, [0, 0, 1]))
    min_bound = result_mesh.bounds[0]
    max_bound = result_mesh.bounds[1]
    center_x = min_bound[0]+(max_bound[0]-min_bound[0])/2.0
    center_offset = -np.array([center_x, 0.0, min_bound[2]-height_offset])
    result_mesh.apply_translation(center_offset)

    if visualize:
        visualize_mesh(result_mesh) 
    return result_mesh

def create_stairs_cfgs(dim, height, stair_type='straignt_up', level=0.0):    
    # * types = {straight_up, straight_down, pyramid_up, pyramid_down, up_down, down_up, plus_up, plus_down}
    # * sets the number of stairs by size of the array
    num_stairs = 7        

    noise = level * height/num_stairs * 0.9
    cfgs = create_stairs(
        MeshPartsCfg(dim=dim), height_diff=height, num_stairs=num_stairs, type=stair_type, noise=noise)
    return cfgs

def create_inclined_surfaces_cfgs(dim, height=0.0):
    width = 0.3                     # width of each block
    side_distance = 0.3             # distane between two blocks in the side direction
    side_std = 0.0
    height_std = 0.0
    incline_angle = np.pi/6.0       # angle of inclined surfaces
    cfgs = create_inclined_surfaces(MeshPartsCfg(dim=dim), width=width, height=height, side_distance=side_distance, 
                         side_std=side_std, height_std=height_std, angle=incline_angle)    
    return cfgs

def create_overhanging_boxes_cfgs(dim, height=0.0):
    cfgs = create_two_overhanging_boxes(MeshPartsCfg(dim=dim), height_offset=height, side_std=0.0, height_std=0.0, aligned=True)    
    return cfgs
   
def create_single_beam_cfgs(dim, height=0.0):
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

def generate_parkour_demo_terrain(dim, mesh_dir, mesh_type="stl", visualize=False):
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
    ascend_stair_cfgs = create_stairs_cfgs(dim_w_longer_y, height, stair_type='straight_up')
    descend_stair_cfgs = create_stairs_cfgs(dim_w_longer_y, height, stair_type='straight_down')
    inclined_surface_cfgs = create_inclined_surfaces_cfgs(dim_w_longer_y, height=0.0)
    overhang_cfgs = create_overhanging_boxes_cfgs(dim, height)
    beam_cfgs = create_single_beam_cfgs(dim, height+0.2)

    single_beam_cfgs_list = [floor_cfgs, inclined_surface_cfgs, ascend_stair_cfgs, overhang_cfgs, 
                 descend_stair_cfgs, floor_cfgs]    
    double_beam_cfgs_list = [floor_cfgs, inclined_surface_cfgs, ascend_stair_cfgs, beam_cfgs, 
                 descend_stair_cfgs, floor_cfgs]    
    
    result_cfgs = concatenate_configs(single_beam_cfgs_list)
    result_mesh = generate_mesh(result_cfgs, height_offset=1.0, visualize=visualize)
    
    # Save mesh to file
    mesh_dir = os.path.join(mesh_dir+"parkour_demo_terrain/")
    mesh_name="terrain_mesh"

    os.makedirs(mesh_dir, exist_ok=True)
    print("saving mesh to ", mesh_name)
    result_mesh.export(os.path.join(mesh_dir, mesh_name+"."+mesh_type))

def generate_task_list(dim, num_tiles = 1, level=0.0) -> dict:
    floor_cfgs = (
        PlatformMeshPartsCfg(
            name="floor",
            dim=dim,            
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
        ),
    )
    stair_height = 1.5
    float_floor_cfgs = (
        PlatformMeshPartsCfg(
            name="floor",
            dim=dim,
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
            height_offset=stair_height,
        ),
    )
    ascend_stair_cfgs = create_stairs_cfgs(dim, height=stair_height, stair_type='straight_up', level=level)
    descend_stair_cfgs = create_stairs_cfgs(dim, height=stair_height, stair_type='straight_down', level=level)
    inclined_surface_cfgs = create_inclined_surfaces_cfgs(dim)

    stair_unit = [floor_cfgs, ascend_stair_cfgs, descend_stair_cfgs, floor_cfgs]
    inclined_surface_unit = [floor_cfgs, inclined_surface_cfgs, floor_cfgs]
    tasks = {
        "plane": [[floor_cfgs] for _ in range(num_tiles)],
        "stair_ascent_descent": [stair_unit for _ in range(int (np.ceil(num_tiles/len(stair_unit))))],
        "inclined_surface": [inclined_surface_unit for _ in range(int (np.ceil(num_tiles/len(inclined_surface_unit))))]
    }
    return tasks

def create_a_task(task_name="", 
                  task_cfg_list=[], 
                  height_offset=0.0, 
                  mesh_dir = "results/primitives_combined/", 
                  mesh_type = "stl",
                  level_id = 0,
                  visualize=False):
    task_cfgs = concatenate_configs(task_cfg_list)
    result_mesh = generate_mesh(task_cfgs, height_offset=height_offset, visualize=visualize)

    mesh_dir = os.path.join(mesh_dir+task_name+"/"+"level{}".format(level_id)+"/")
    mesh_name="terrain_mesh"
    mesh_path = os.path.join(mesh_dir, mesh_name+"."+mesh_type)

    # Save mesh to either obj or stl file
    os.makedirs(mesh_dir, exist_ok=True)
    print("saving mesh to ", mesh_name)
    result_mesh.export(mesh_path)

    # Export height map in the format of depth image to the same folder
    height_map_name = "height_map.png"
    height_map_path = os.path.join(mesh_dir, height_map_name)
    resolution = 0.04
    height_scale = mesh_to_heightmap(mesh_file=mesh_path, output_file=height_map_path, resolution=resolution, check_revserse=False, use_nonzero_ground=True, visualize=visualize)
    height_scale = float(height_scale)

    # Save heightmap data info
    height_map_info_name = "height_map_info.yml"
    height_map_info_path = os.path.join(mesh_dir, height_map_info_name)
    height_map_info = {'/**':{
    'ros__parameters':{
    'terrain_info':{
        'resolution': resolution,
        'height_scale': height_scale
        }}
    }}
    with open(height_map_info_path, 'w') as yaml_file:
        yaml.dump(height_map_info, yaml_file, default_flow_style=False)
    
    # Get terrain length in y direction (x direction in ocs2)
    terrain_len = 0.0
    for cfg in task_cfg_list:
        terrain_len = terrain_len + cfg[0].dim[1]

    # Save command info
    distance = 20.0
    max_velocity = 1.5
    init_y_offset = -(terrain_len/2.0 - 0.5)
    for velocity_level in range(1,11):
        forward_velocity = velocity_level/10.0 * max_velocity
        command_name = "command_{}".format(velocity_level)
        command_dir = mesh_dir+command_name
        os.makedirs(command_dir, exist_ok=True)
        command_path = os.path.join(command_dir, "command_info.yml")        
        command = height_map_info = {'/**':{
        'ros__parameters':{
        'command_info':{
            'forward_velocity': forward_velocity,
            'gait_duration': 1.0,
            'mpc_frequency': 100.0,
            'mrt_frequency': 400.0,
            'distance': distance,
            'init_x_offset': init_y_offset,
            'adaptReferenceToTerrain': True
            }}
        }}
        with open(command_path, 'w') as yaml_file:
            yaml.dump(command, yaml_file, default_flow_style=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create some primitive patches")
    parser.add_argument(
        "--single_task", type=bool, default=False, help="True to generate a single task. False to generate all tasks in predefined task list"
    )
    parser.add_argument(
        "--task", type=str, default="stair_ascent", help="Type of the single-task terrain. options: plane, stair_ascent, stair_descent, stair_ascent_descent, inclined_surface"
    )
    parser.add_argument(
        "--mesh_dir", type=str, default="/home/hli/Data/perceptive_locomotion_tasks/", help="directory to save mesh"
    )
    parser.add_argument(
        "--visualize", type=bool, default=False, help="Visualizing the primitives"
    )
    args = parser.parse_args()

    dim = (2.0, 3.0, 3.0)
     
    for level in np.arange(0.0, 0.2, 0.1):
        task_list = generate_task_list(dim=dim, num_tiles=8, level=level)

        for task_name, task_cfgs_list in task_list.items():
            level_id = int (level * 10)
            task_cfgs_list_2 = []
            for task_cfg in task_cfgs_list:
                task_cfgs_list_2 += task_cfg
            create_a_task(task_name=task_name, task_cfg_list=task_cfgs_list_2, height_offset=0.5, mesh_dir=args.mesh_dir, visualize=args.visualize,level_id=level_id)

    # generate_parkour_demo_terrain(dim, args.mesh_dir, visualize=args.visualize)