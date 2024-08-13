import trimesh
import pandas as pd
import numpy as np
import os
from terrain_generator.utils.mesh_utils import get_height_array_of_mesh2
from multiprocessing import Pool
import argparse
import shutil
import json
import yaml
import math

import numpy as np
import trimesh
from PIL import Image

def compute_bounds(vertices):
    """Compute the min and max bounds of the vertices."""
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    return min_bound, max_bound

def heightmap_to_mesh(image_path, scale=1.0, resolution=0.04):
    """Convert a height map image to a 3D mesh and save it as an OBJ file."""
    # Load the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    heightmap = np.array(image, dtype=np.float32)
    
    # Normalize the heightmap values to [0, 1]
    heightmap = heightmap / 255.0
    
    # Optionally apply a scaling factor to adjust height values
    heightmap *= scale

    # Get image dimensions
    height, width = heightmap.shape

    # Create the grid of vertices
    x, y = np.meshgrid(np.arange(width)* resolution, np.arange(height)* resolution) 
    vertices = np.stack([x, y, heightmap], axis=-1)

    # Flatten the vertices for trimesh
    vertices = vertices.reshape(-1, 3)

    # Create the faces (two triangles per quad)
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            # Get the index of the four corners of the cell
            v0 = i * width + j
            v1 = v0 + 1
            v2 = v0 + width
            v3 = v2 + 1
            
            # Create two triangles for the cell
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    faces = np.array(faces, dtype=np.int32)

    # Create a mesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)    

    mesh.show()

def mesh_to_heightmap(mesh_file, output_file, resolution=0.04, check_revserse=False):
    # Load the mesh
    mesh = trimesh.load(mesh_file)
    mesh.show()
    
    # Ensure the mesh is 3D
    if mesh.vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must be 3D (x, y, z coordinates).")

    # Compute the bounds of the mesh
    vertices = mesh.vertices
    min_bound, max_bound = compute_bounds(vertices)
    size = max_bound - min_bound
    print("max bound = {}".format(max_bound))
    print("min bound = {}".format(min_bound))
    
    size_x = math.ceil(size[0]/resolution)
    size_y = math.ceil(size[1]/resolution)
   
    # Create a grid to sample the height map
    grid_x, grid_y = np.mgrid[0:size_x, 0:size_y]
    
    # Normalize the grid to the size of the mesh
    grid_x = grid_x * resolution  + min_bound[0]
    grid_y = grid_y * resolution  + min_bound[1]
    
    # Create an array to hold the heights
    heights = np.full((size_x, size_y), min_bound[2])  # Initialize with min Z

    # Perform ray tracing for each grid point
    for i in range(size_x):
        for j in range(size_y):
            ray_origin = np.array([grid_x[i, j], grid_y[i, j], max_bound[2]])
            ray_direction = np.array([0, 0, -1])  # Ray pointing downward

            # Perform ray tracing to find intersections
            locations, _, _ = mesh.ray.intersects_location(ray_origins=[ray_origin], ray_directions=[ray_direction])
            
            if len(locations) > 0:
                # Get the highest intersection point (max Z value)
                heights[i, j] = np.max([loc[2] for loc in locations])

    # Normalize heights to be between 0 and 255
    height_scale = max_bound[2] - min_bound[2]
    heights = (heights - min_bound[2]) / height_scale * 255
    heights = heights.astype(np.uint8)

    heights = np.flip(heights, axis=1)
    heights = np.flip(heights, axis=0)
    print("Height scale = {}".format(height_scale))

    # Create and save the height map as a PNG file
    height_map = Image.fromarray(heights)
    height_map.save(output_file)

    # heightmap_to_mesh(output_file, max_bound[2]-min_bound[2])
    # DF = pd.DataFrame(heights)
    # print("Converted to DataFrame")

    # # save the dataframe as a csv file and hdf5 file
    # DF.to_csv(output_file, header=None, index=None)
    # print("Saved CSV")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Create mesh from configuration")
  parser.add_argument(
      "--mesh_path", type=str, default="results/primitive_separated/generate_complex_terrain/mesh_complex_terrain.stl", help="Directory to retrieve the mesh files from"
  )
  parser.add_argument(
      "--export_path", type=str, default="results/primitive_separated/generate_complex_terrain/terrains/complex_terrain.png", help="Directory to save the generated heightmap files. Should be an external dir, or empty folder as a subfolder."
  )
  parser.add_argument(
      "--resolution", type=float, default=0.04, help="Resolution of the mesh to be exported (in meters)"
  )
  parser.add_argument(
      "--check_reverse", type=bool, default=False, help="Convert the generated heightmap back to mesh file and visualize it. True or false"
  )

  args = parser.parse_args()

  dimensions = (6.0,30.0,2.0)
  xyzoffsets = [0.0, 0.0, 0.0]
  resolution = args.resolution

  mesh_to_heightmap(args.mesh_path, args.export_path, resolution, args.check_reverse)

