import numpy as np
from stl import mesh
from pywavefront import Wavefront
import trimesh
import argparse

def convert_obj_to_stl(obj_file, stl_file):    
    mesh = trimesh.load(obj_file, file_type='obj')
    
    # Save as STL file
    mesh.export(stl_file, file_type='stl')

# Example usage
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Converting an obj file to an stl file")
    parser.add_argument(
        "--obj_path", type=str, default="results/primitive_separated/generate_complex_terrain/mesh_complex_terrain.obj", help="Path to the obj file"
    )
    parser.add_argument(
        "--stl_path", type=str, default="results/primitive_separated/generate_complex_terrain/mesh_complex_terrain.stl", help="Path to the exported stl file"
    )
    args = parser.parse_args()
    convert_obj_to_stl(args.obj_path, args.stl_path)
