import numpy as np
from dataclasses import dataclass
import os
import yaml
import argparse
from typing import Callable

from configs.parkour_cfg import *
from terrain_generator.utils import visualize_mesh
from export_heightmap import mesh_to_heightmap
from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import create_mesh_tile

def concatenate_configs(list_of_configs : list) -> list: 
    cfgs_res = []    
    for cfgs_ in list_of_configs: 
        # each cfgs_ is a tuple
        cfgs = list(cfgs_)        
        cfgs_res = cfgs_res + cfgs
    return tuple(cfgs_res)

def repeat_patterns(mesh_cfg_func, n, tile_dim, terrain_level):
    cfgs_list = []
    for _ in range(n):
        cfgs_list.append(mesh_cfg_func(tile_dim, terrain_level))
    combined_cfgs = concatenate_configs(cfgs_list)
    terrain_length = 0.0
    for cfg in combined_cfgs:
        terrain_length + cfg.dim[1]
    return combined_cfgs, terrain_length

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


@dataclass
class ParkourTask:    
    mesh_cfgs_funcs: Callable[[tuple, float], None]
    terrain_name: str = ""    
    terrain_length: float = 10.0    
    tile_dim : tuple = (2.0, 2.0)
    terrain_var_levels : tuple = ()
    velocity_levels : tuple = ()


def generate_task_for_one_terrain(task : ParkourTask, task_root_dir:str):
    n_repeats = int (np.ceil(task.terrain_length/task.tile_dim[1]))

    # Loop through terrain level
    for level in task.terrain_var_levels:
        mesh_parttern_cfg, terrain_length = repeat_patterns(task.mesh_cfgs_funcs, n_repeats, task.tile_dim, level)
        mesh = generate_mesh(mesh_parttern_cfg)

        level_id = int (level * 10)
        mesh_dir = os.path.join(task_root_dir+task.terrain_name +"/"+"level{}".format(level_id)+"/")
        mesh_name="terrain_mesh"
        mesh_path = os.path.join(mesh_dir, mesh_name+"."+ "stl")

        # Save mesh to either obj or stl file
        os.makedirs(mesh_dir, exist_ok=True)
        print("saving mesh to ", mesh_name)
        mesh.export(mesh_path)

        # Export height map in the format of depth image to the same folder
        height_map_name = "height_map.png"
        height_map_path = os.path.join(mesh_dir, height_map_name)
        resolution = 0.04
        height_scale = mesh_to_heightmap(mesh_file=mesh_path, output_file=height_map_path, resolution=resolution, check_revserse=False, use_nonzero_ground=True, visualize=False)
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



def create_task_lists_for_multiple_terrains():
    tile_dim = (2.0, 3.0, 3.0)
    terrain_length = 10.0
    terrain_levels = np.arange(0.1, 1.1, 0.1)
    velocity_levels = np.arange(0.2, 1.2, 0.2)    
    tasks = [
        ParkourTask(
            mesh_cfgs_funcs = create_stepping_stones_cfgs,
            terrain_name = "stepping_stones",
            terrain_length = terrain_length,            
            tile_dim = tile_dim,
            terrain_var_levels = terrain_levels,
            velocity_levels = velocity_levels
        )
    ]
    return tasks

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create some primitive patches")   
    parser.add_argument(
        "--mesh_dir", type=str, default="/home/hli/Data/perceptive_locomotion_tasks/", help="directory to save mesh"
    )
    parser.add_argument(
        "--visualize", type=bool, default=False, help="Visualizing the primitives"
    )
    args = parser.parse_args()

     
    tasks = create_task_lists_for_multiple_terrains()
    
    for task in tasks:
        generate_task_for_one_terrain(task, args.mesh_dir)

