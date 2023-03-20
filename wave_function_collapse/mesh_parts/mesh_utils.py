import os
import numpy as np
from pyglet.window.key import P
import trimesh
from typing import Callable, Any, Optional
from dataclasses import asdict, is_dataclass
import open3d as o3d
import copy


ENGINE = "blender"
CACHE_DIR = "mesh_cache"
# ENGINE = "scad"


def merge_meshes(meshes, minimal_triangles=False, engine="blender"):
    if minimal_triangles:
        meshes = trimesh.boolean.union(meshes, engine=engine)
    else:
        meshes = trimesh.util.concatenate(meshes)
    return meshes


def flip_mesh(mesh, direction):
    """Flip a mesh in a given direction."""
    new_mesh = mesh.copy()
    if direction == "x":
        # Create the transformation matrix for inverting the mesh in the x-axis
        transform = trimesh.transformations.scale_matrix(-1, [0, 0, 0], [1, 0, 0])
    elif direction == "y":
        transform = trimesh.transformations.scale_matrix(-1, [0, 0, 0], [0, 1, 0])
    else:
        raise ValueError(f"Direction {direction} is not defined.")
    # Apply the transformation to the mesh
    new_mesh.apply_transform(transform)
    return new_mesh


def rotate_mesh(mesh, deg):
    """Rotate a mesh in a given degree."""
    new_mesh = mesh.copy()
    if deg == 90:
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    elif deg == 180:
        transform = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    elif deg == 270:
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
    else:
        raise ValueError(f"Rotation degree {deg} is not defined.")
    new_mesh.apply_transform(transform)
    return new_mesh


def get_height_array_of_mesh(mesh, dim, num_points, offset=0.01):
    # intersects_location requires origins to be the same shape as vectors
    x = np.linspace(-dim[0] / 2.0 + offset, dim[0] / 2.0 - offset, num_points)
    y = np.linspace(dim[1] / 2.0 - offset, -dim[1] / 2.0 + offset, num_points)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    origins = np.stack([xv, yv, np.ones_like(xv) * dim[2] * 2], axis=-1)
    vectors = np.stack([np.zeros_like(xv), np.zeros_like(yv), -np.ones_like(xv)], axis=-1)
    # # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(origins, vectors, multiple_hits=False)
    array = np.zeros((num_points * num_points))
    if len(points) > 0:
        array[index_ray] = points[:, 2] + dim[2] / 2.0
        array = np.round(array, 1)
    array = array.reshape(num_points, num_points)
    return array


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y_min = -num_cols * horizontal_scale / 2.0
    y_max = num_cols * horizontal_scale / 2.0

    x_min = -num_rows * horizontal_scale / 2.0
    x_max = num_rows * horizontal_scale / 2.0

    y = np.linspace(y_min, y_max, num_cols)
    x = np.linspace(x_min, x_max, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return mesh


def merge_two_height_meshes(mesh1, mesh2):
    """
    Merge two heightfield meshes into one.
    Parameters:
        mesh1 (trimesh.Trimesh): first heightfield mesh
        mesh2 (trimesh.Trimesh): second heightfield mesh
    Returns:
        merged_mesh (trimesh.Trimesh): merged heightfield mesh
    """
    # merge the two meshes
    merged_mesh = trimesh.util.concatenate(mesh1, mesh2)

    # connect edges
    vertices = merged_mesh.vertices
    faces = merged_mesh.faces

    n_vertices = vertices.shape[0] // 2
    n = int(np.sqrt(n_vertices))
    indices = np.arange(n_vertices).reshape(n, n)

    new_faces = []
    edges = [indices[0, :], indices[-1, :], indices[:, 0], indices[:, -1]]
    for edge_1 in edges:
        ind0 = edge_1[:-1]
        ind1 = edge_1[1:]
        ind2 = ind0 + n_vertices
        ind3 = ind1 + n_vertices
        f1 = np.zeros((n - 1, 3), dtype=np.uint32)
        f1[:, 0] = ind0
        f1[:, 1] = ind1
        f1[:, 2] = ind2
        new_faces.append(f1)
        f2 = np.zeros((n - 1, 3), dtype=np.uint32)
        f2[:, 0] = ind2
        f2[:, 1] = ind3
        f2[:, 2] = ind1
        new_faces.append(f2)
    new_faces = np.vstack(new_faces)
    merged_mesh = trimesh.Trimesh(vertices=vertices, faces=np.vstack([faces, new_faces]))
    return merged_mesh


def cfg_to_hash(cfg, exclude_keys=["weight", "load_from_cache"]):
    """MD5 hash of a config."""
    import hashlib
    import json

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # if isinstance(obj, tuple):
            #     return str(list(obj))
            # if isinstance(obj, dict):
            #     return self.default(obj)
            return json.JSONEncoder.default(self, obj)

    def tuple_to_str(d):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = tuple_to_str(v)
            if isinstance(k, tuple):
                new_d[str(k)] = v
            else:
                new_d[k] = v
        return new_d

    if isinstance(cfg, dict):
        cfg_dict = copy.deepcopy(cfg)
    elif is_dataclass(cfg):
        cfg_dict = asdict(cfg)
    else:
        raise ValueError("cfg must be a dict or dataclass.")
    for key in exclude_keys:
        cfg_dict.pop(key, None)
    cfg_dict = tuple_to_str(cfg_dict)
    encoded = json.dumps(cfg_dict, sort_keys=True, cls=NpEncoder).encode()
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    # encoded = json.dumps(cfg, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_cached_mesh_gen(
    mesh_gen_fn: Callable[[Any], trimesh.Trimesh],
    cfg,
    verbose=False,
    use_cache=True,
) -> Callable[[], trimesh.Trimesh]:
    """Generate a mesh if there's no cache. If there's cache, load from cache."""
    code = cfg_to_hash(cfg)
    os.makedirs(CACHE_DIR, exist_ok=True)
    if hasattr(cfg, "name"):
        name = cfg.name
    else:
        name = ""

    def mesh_gen() -> trimesh.Trimesh:
        if os.path.exists(os.path.join(CACHE_DIR, code + ".stl")) and use_cache:
            if verbose:
                print(f"Loading mesh {name} from cache {code}.stl ...")
            mesh = trimesh.load_mesh(os.path.join(CACHE_DIR, code + ".stl"))
        else:
            # if verbose:
            print(f"Not loading {name} from cache, creating {code}.stl ...")
            mesh = mesh_gen_fn(cfg)
            mesh.export(os.path.join(CACHE_DIR, code + ".stl"))
        return mesh

    return mesh_gen


def visualize_mesh(mesh, save_path=None):
    """Visualize a mesh."""
    # o3d_mesh = o3d.geometry.TriangleMesh()
    if isinstance(mesh, trimesh.Trimesh):
        o3d_mesh = mesh.as_open3d
        # mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        # mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        o3d_mesh = mesh
    # Visualize meshes one by one with Open3D
    o3d_mesh.compute_vertex_normals()
    R = o3d.geometry.get_rotation_matrix_from_xyz([-1.0, 0.0, 0.2])
    print("R ", R)
    o3d_mesh.rotate(R, center=[0, 0, 0])
    o3d.visualization.draw_geometries([o3d_mesh])
    vis = o3d.visualization.Visualizer()
    vis.capture_screen_image(save_path)
