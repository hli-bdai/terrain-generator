import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import argparse
import os


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Create mesh from configuration")
  parser.add_argument(
      "--mesh_path", type=str, default="results/generated_terrain", help="Directory to retrieve the mesh files from"
  )
  parser.add_argument(
      "--export_dir", type=str, default="results/generated_terrain/terrains", help="Directory to save the generated heightmap files. Should be an external dir, or empty folder as a subfolder."
  )
  # Usage
  args = parser.parse_args()
  csv_filepath = args.mesh_path
  image_filepath = os.path.join(args.export_dir, 'depth_image.png')
  
  # Load the CSV file to inspect its contents
  heightmap_data = pd.read_csv(csv_filepath)
  
  # Convert the heightmap data to a numpy array for plotting
  heightmap_array = heightmap_data.to_numpy()
  print(heightmap_array.shape)
  
  # Generate the depth image using a colormap suitable for representing depth
  plt.figure(figsize=(20, 20))
  plt.imshow(heightmap_array, cmap='viridis')
  plt.colorbar(label='Height')
  plt.title('Depth Image of Complex Terrain')
  plt.axis('off')
  plt.show()
  
  # Save the depth image as a PNG file
#   plt.savefig(image_filepath)
  # imageio.imwrite(image_filepath, heightmap_array)