#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import trimesh
import numpy as np
from math import floor


from ..mesh_parts.mesh_parts_cfg import (
    MeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
    WallMeshPartsCfg,
    CapsuleMeshPartsCfg,
    BoxMeshPartsCfg,
)


def create_floor(cfg: MeshPartsCfg, **kwargs):
    dims = [cfg.dim[0], cfg.dim[1], cfg.floor_thickness]
    pose = np.eye(4)
    pose[:3, -1] = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness / 2.0 + cfg.height_offset]
    floor = trimesh.creation.box(dims, pose)
    return floor


def create_step(cfg: MeshPartsCfg, height_diff=0.2, **kwargs):
    height_diff = height_diff + cfg.floor_thickness
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="step",
            dim=cfg.dim,
            array=np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
            * height_diff,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[1, 1], [1, 1]]) * height_diff,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

def create_ramp(cfg: MeshPartsCfg, height_diff=0.2, **kwargs):
    height_diff = height_diff + cfg.floor_thickness
    dim = (30, 30)
    step_height = height_diff / (dim[1] - 1)
    h1 = 0
    array = np.zeros(dim)
    for s in range(dim[0]):
        array[:, s] = h1
        h1 = h1 + step_height
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        HeightMapMeshPartsCfg(
            name="ramp",
            dim=cfg.dim,
            height_map=array,
            rotations=(),
            flips=(),
            slope_threshold=1.5,
            target_num_faces=1000,
            simplify=True,
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[1, 1], [1, 1]]) * height_diff,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs


def create_middle_step(cfg: MeshPartsCfg, height_diff=0.2, n=11, **kwargs):
    height_diff = height_diff + cfg.floor_thickness
    array = np.zeros((n, n))
    middle_n = n // 2
    array[middle_n, :] = height_diff
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]) * height_diff,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="step",
            dim=cfg.dim,
            array=array,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]) * height_diff,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs


def create_gaps(cfg: MeshPartsCfg, gap_length=0.2, height_diff=0.0, **kwargs):
    dims = []
    transformations = []

    # First platform_
    dims.append([cfg.dim[0], (cfg.dim[1] - gap_length) / 2.0, cfg.floor_thickness])
    t = np.eye(4)
    t[:3, -1] = np.array(
        [
            0,
            -cfg.dim[1] / 2.0 + (cfg.dim[1] - gap_length) / 4.0,
            cfg.floor_thickness / 2.0,
        ]
    )
    transformations.append(t)

    # Second platform_
    dims.append([cfg.dim[0], (cfg.dim[1] - gap_length) / 2.0, cfg.floor_thickness])
    t2 = np.eye(4)
    t2[:3, -1] = np.array(
        [
            0,
            -cfg.dim[1] / 2.0 + (3.0 * cfg.dim[1] + gap_length) / 4.0,
            cfg.floor_thickness / 2.0,
        ]
    )
    transformations.append(t2)

    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        BoxMeshPartsCfg(
            name="middle",
            dim=cfg.dim,
            box_dims=tuple(dims),
            transformations=tuple(transformations),
            # rotations=(90, 180, 270),
            # flips=("x", "y"),
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]) + height_diff,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs


def create_narrow(cfg: MeshPartsCfg, width=0.5, side_std=0.0, height_std=0.0, n=10, rotation=(180, ), **kwargs):
    dims = []
    transformations = []

    step_length = cfg.dim[1] / n

    for i in range(floor(n/2)):
        # First platform_
        if i == floor(n/2) - 1:
            dims.append([width, step_length + width/2, cfg.floor_thickness])
            y = -cfg.dim[1] / 2.0 + (i + 0.5) * step_length + width / 4
        else:
            dims.append([width, step_length, cfg.floor_thickness])
            y = -cfg.dim[1] / 2.0 + (i + 0.5) * step_length
        t = np.eye(4)
        x = np.random.normal(0, side_std)
        z = cfg.floor_thickness / 2.0 + np.random.normal(0, height_std)
        t[:3, -1] = np.array([x, y, z])
        transformations.append(t)

    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        BoxMeshPartsCfg(
            name="middle",
            dim=cfg.dim,
            box_dims=tuple(dims),
            transformations=tuple(transformations),
            rotations=rotation,
            # flips=("x", "y"),
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

# angle has unit: rad
def create_inclined_surfaces(cfg: MeshPartsCfg, width=0.5, height=0.0, side_distance=0.0, side_std=0.0, height_std=0.0, n=8, angle=0.0, **kwargs):
    dims = []
    transformations = []
    thickness = 0.05    # thickness of the inclined surfaces
    step_length = cfg.dim[1] / n

    for i in range(n):
        # First platform_
        if i == floor(n/2) - 1:
            dims.append([width, step_length + width/2, thickness])
            y = -cfg.dim[1] / 2.0 + (i + 0.5) * step_length + width / 4
        else:
            dims.append([width, step_length, thickness])
            y = -cfg.dim[1] / 2.0 + (i + 0.5) * step_length
        x = np.random.normal(side_distance*(-1)**(i%2), side_std)
        z = cfg.floor_thickness / 2.0 + np.random.normal(0, height_std) + height
        t = trimesh.transformations.rotation_matrix(angle*(-1)**(i%2+1), [0, 1, 0])
        t[:3, -1] = np.array([x, y, z])
        transformations.append(t)

    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
        ),
        BoxMeshPartsCfg(
            name="surface",
            dim=cfg.dim,
            box_dims=tuple(dims),
            transformations=tuple(transformations),            
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs


def create_stepping(cfg: MeshPartsCfg, width=0.5, side_std=0.0, height_std=0.0, ratio=0.5, n=10, **kwargs):
    dims = []
    transformations = []

    step_length = cfg.dim[0] / n * ratio

    for i in range(n):
        # First platform_
        dims.append([step_length, width, cfg.floor_thickness])
        t = np.eye(4)
        y = np.random.normal(0, side_std)
        x = -cfg.dim[0] / 2.0 + (i + 0.5) * cfg.dim[0] / n
        z = cfg.floor_thickness / 2.0 + np.random.normal(0, height_std)
        t[:3, -1] = np.array([x, y, z])
        transformations.append(t)

    cfgs = (
        BoxMeshPartsCfg(
            name="middle",
            dim=cfg.dim,
            box_dims=tuple(dims),
            transformations=tuple(transformations),          
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
        ),
    )
    return cfgs


def create_box_grid(cfg: MeshPartsCfg, height_diff=0.5, height_std=0.2, n=8, **kwargs):
    height_diff = height_diff + cfg.floor_thickness
    array = np.zeros((n, n))
    for i in range(n):
        if i % 2 == 0:
            array[i, :] = array[i, :] + np.linspace( 0, height_diff, n)
    array[:,range(0, n, 2)] = 0.0
    array = array.T
    array += np.random.normal(0, height_std, size=array.shape)
    # array[5, :] = height_diff
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]) * height_diff,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="middle",
            dim=cfg.dim,
            array=array,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
            minimal_triangles=False,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]) + height_diff,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs


def create_floating_box_grid(
    cfg: MeshPartsCfg, height_diff=0.5, height_std=0.2, n=8, height_gap_mean=1.0, height_gap_std=0.8, **kwargs
):
    cfgs = create_box_grid(cfg, height_diff, height_std, n, **kwargs)
    cfgs = list(cfgs)
    array = cfgs[1].array

    z_dim_array = np.ones_like(array)*1.0
    floating_array = array + np.random.normal(height_gap_mean, height_gap_std, size=array.shape) + z_dim_array
    # print("floating_array", floating_array)
    array[5, :] = height_diff
    cfgs.append(
        PlatformMeshPartsCfg(
            name="floating",
            dim=cfg.dim,
            array=floating_array,
            z_dim_array=z_dim_array,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
            use_z_dim_array=True,
        ),
    )
    return tuple(cfgs)


def create_random_tunnel(
    cfg: MeshPartsCfg, height_diff=0.5, height_std=0.2, n=8, height_gap_mean=1.0, height_gap_std=0.8, wall_n=3, **kwargs
):
    cfgs = create_box_grid(cfg, height_diff, height_std, n, **kwargs)
    cfgs = list(cfgs)
    array = cfgs[1].array

    z_dim_array = np.ones_like(array)
    floating_array = array + np.random.normal(height_gap_mean, height_gap_std, size=array.shape) + z_dim_array
    # print("floating_array", floating_array)

    z_dim_array[:, :wall_n] = 1.0 + floating_array[:, :wall_n]
    z_dim_array[:, -wall_n:] = 1.0 + floating_array[:, -wall_n:]
    # array[5, :] = height_diff
    cfgs.append(
        PlatformMeshPartsCfg(
            name="middle_floating",
            dim=cfg.dim,
            array=floating_array,
            z_dim_array=z_dim_array,
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
            use_z_dim_array=True,
        ),
    )
    return tuple(cfgs)


def create_random_boxes(cfg: MeshPartsCfg, width=0.5, side_std=0.0, height_std=0.0, n=8, **kwargs):
    dims = []
    transformations = []

    step_length = cfg.dim[1] / n

    for i in range(n):
        # First platform_
        dims.append([width, step_length, cfg.floor_thickness])
        t = np.eye(4)
        x = np.random.normal(0, side_std)
        y = -cfg.dim[1] / 2.0 + (i + 0.5) * step_length
        z = cfg.floor_thickness / 2.0 + np.random.normal(0, height_std)
        t[:3, -1] = np.array([x, y, z])
        transformations.append(t)

    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        BoxMeshPartsCfg(
            name="middle",
            dim=cfg.dim,
            box_dims=tuple(dims),
            transformations=tuple(transformations),
            # rotations=(90, 180, 270),
            # flips=("x", "y"),
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs


def create_overhanging_boxes(cfg: MeshPartsCfg, width=0.5, side_std=0.0, height_std=0.0, n=10, **kwargs):
    dims = []
    transformations = []

    step_length = cfg.dim[1] / n

    for i in range(n):
        # First platform_
        dims.append([width, step_length, cfg.floor_thickness])
        t = np.eye(4)
        x = np.random.normal(0, side_std)
        y = -cfg.dim[1] / 2.0 + (i + 0.5) * step_length
        z = cfg.floor_thickness / 2.0 + np.random.normal(0, height_std)
        t[:3, -1] = np.array([x, y, z])
        transformations.append(t)

    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        BoxMeshPartsCfg(
            name="middle",
            dim=cfg.dim,
            box_dims=tuple(dims),
            transformations=tuple(transformations),
            # rotations=(90, 180, 270),
            # flips=("x", "y"),
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

def create_two_overhanging_boxes(cfg: MeshPartsCfg, side_std=0.0, height_offset=0.0, height_std=0.0, aligned=False, add_floor=False, **kwargs):
    box_dims = []
    transformations = []

    n = 2
    width = 0.2
    distance = 0.5
    step_length = 0.4*cfg.dim[1]
    height_diff = 0.3
    pitch = 0.0
    if aligned:
        distance = 0.0
        height_diff = 0.0        

    z_nominal_array = np.array([height_diff/2.0, -height_diff/2.0]) + height_offset
    x_nominal_array = np.array([-distance/2.0, distance/2.0])
    y_nominal_array = np.array([-step_length/2.0, step_length/2.0])        

    for i in range(n):                
        box_dims.append([width, step_length, cfg.floor_thickness*1.5])
        t = trimesh.transformations.rotation_matrix(pitch, [1, 0, 0])
        x = x_nominal_array[i] + np.random.normal(0, side_std)
        y = y_nominal_array[i] 
        z = z_nominal_array[i] + np.random.normal(0, height_std)
        t[:3, -1] = np.array([x, y, z])
        transformations.append(t)

    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        BoxMeshPartsCfg(
            name="overhang_boxes",
            dim=cfg.dim,
            box_dims=tuple(box_dims),
            transformations=tuple(transformations),
            # rotations=(90, 180, 270),
            # flips=("x", "y"),
            weight=0.1,
            minimal_triangles=False,
            add_floor=add_floor,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs


# def create_middle_step(cfg: MeshPartsCfg, height_diff=0.2, **kwargs):
#     height_diff = height_diff + cfg.floor_thickness
#     array = np.ones((11, 11)) * 0.1
#     array[5, :] = height_diff
#     cfgs = (
#         PlatformMeshPartsCfg(
#             name="start",
#             dim=cfg.dim,
#             array=np.array([[0, 0], [0, 0]]) * height_diff,
#             rotations=(90, 180, 270),
#             flips=(),
#             weight=0.1,
#         ),
#         PlatformMeshPartsCfg(
#             name="step",
#             dim=cfg.dim,
#             array=array,
#             rotations=(90, 180, 270),
#             flips=(),
#             weight=0.1,
#         ),
#         PlatformMeshPartsCfg(
#             name="goal",
#             dim=cfg.dim,
#             array=np.array([[0, 0], [0, 0]]) * height_diff,
#             rotations=(90, 180, 270),
#             flips=(),
#             weight=0.1,
#         ),
#     )
#     return cfgs


def create_tunnel(cfg: MeshPartsCfg, height_diff=0.2, **kwargs):
    height_diff = height_diff + cfg.floor_thickness
    array = np.zeros((11, 11))
    array[5, :] = height_diff
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]) * height_diff,
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="step",
            dim=cfg.dim,
            array=array,
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=np.array([[0, 0], [0, 0]]) * height_diff,
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs


def create_stairs(
        cfg: MeshPartsCfg, height_diff=1.0, num_stairs: int=1,
        type: str='straight_up', noise: float=0.0, flip: bool = False):
    # * create an array for the stair feature based on the desired number of stairs
    stairs_array = np.ones((num_stairs, num_stairs))
    start_array = np.array([[0]])
    goal_array = np.array([[0]])

    if type == 'straight_up':
        goal_array = np.array([[1]])
        for i in range(num_stairs):
            stairs_array[i, :] = (num_stairs-i)/(num_stairs+1)
    elif type == 'straight_down':
        start_array = np.array([[1]])
        for i in range(num_stairs):
            stairs_array[i, :] = (i+1)/(num_stairs+1)
    elif type == 'pyramid_up':
        for i in range(int(np.ceil((num_stairs-1)/2))):
            height = (i+1)/(num_stairs+1)
            stairs_array[i:num_stairs-i, i:num_stairs-i] = height
    elif type == 'pyramid_down':
        start_array = np.array([[1]])
        goal_array = np.array([[1]])
        for i in range(int(np.ceil((num_stairs-1)/2))):
            height = (num_stairs-i)/(num_stairs+1)
            stairs_array[i:num_stairs-i, i:num_stairs-i] = height
    elif type == 'up_down':
        for i in range(int(np.ceil((num_stairs)/2))):
            height = (i+1)/(num_stairs+1)
            stairs_array[i:num_stairs-i, :] = height
    elif type == 'down_up':
        start_array = np.array([[1]])
        goal_array = np.array([[1]])
        for i in range(int(np.ceil((num_stairs)/2))):
            height = (num_stairs-i)/(num_stairs+1)
            stairs_array[i:num_stairs-i, :] = height
    elif type == 'plus_up':
        for i in range(int(np.ceil((num_stairs)/2))):
            height = (i)/(num_stairs+1)
            stairs_array[i:num_stairs-i, :] = height
            stairs_array[:, i:num_stairs-i] = height
    elif type == 'plus_down':
        start_array = np.array([[1]])
        goal_array = np.array([[1]])
        for i in range(int(np.ceil((num_stairs)/2))):
            height = (num_stairs-i+1)/(num_stairs+1)
            stairs_array[i:num_stairs-i, :] = height
            stairs_array[:, i:num_stairs-i] = height
    else:
        print(f'error: type {type} not found')
        exit()

    if noise != 0.0:
        noise_array = noise*(2*np.random.rand(num_stairs, num_stairs)-1)
        stairs_array += noise_array

    if flip:
        stairs_array = stairs_array.transpose()
        stairs_array = np.flip(stairs_array, 1)
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=cfg.dim,
            array=((start_array*height_diff)+cfg.floor_thickness),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name=type,
            dim=cfg.dim,
            array=((stairs_array*height_diff)+cfg.floor_thickness),
            # rotations=(90, 180, 270),
            flips=(0,1,0),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=cfg.dim,
            array=((goal_array*height_diff)+cfg.floor_thickness),
            # rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

if __name__ == "__main__":
    # cfg = FloorPattern()
    # cfgs = generate_floating_capsules("capsule", [2, 2, 2], n=10, max_l=1.0, min_l=0.5, min_r=0.05, max_r=0.2, max_n_per_tile=10, weight=1.0, seed=1234)
    # cfgs = generate_random_boxes(
    #     "boxes", [2, 2, 2], n=10, max_h=0.5, min_h=0.1, min_w=0.10, max_w=0.5, max_n_per_tile=15, weight=1.0, seed=1234
    # )
    # cfgs = create_step(cfg=PlatformMeshPartsCfg(dim=(3, 3, 3)), height_diff=0.2)
    # cfgs = create_middle_step(cfg=PlatformMeshPartsCfg(dim=(3, 3, 3)), height_diff=1.0)
    cfgs = create_gaps(cfg=PlatformMeshPartsCfg(dim=(3, 3, 3)), gap_length=0.3, height_diff=0.2)
    print("cfg", cfgs)
    # print("cfg", cfgs)

    from mesh_parts.create_tiles import create_mesh_tile

    visualize_keywords = ["all"]
    # for mesh_part in cfg.mesh_parts:
    tiles = {}
    for cfg in cfgs:
        # print("cfg ", cfg)
        cfg.load_from_cache = False
        mesh_tile = create_mesh_tile(cfg)
        # print("mesh part ", mesh_tile)
        for keyword in visualize_keywords:
            print("keyword ", keyword)
            # print(mesh_part.edges)
            if keyword in mesh_tile.name or keyword == "all":
                # mesh_tile = create_mesh_tile(mesh_part)
                print(mesh_tile.name, mesh_tile.edges)
                mesh_tile.get_mesh().show()
                break

        # if "ramp" in mesh_part.name:
        #     print(mesh_part)
        #     mesh_tile = create_mesh_tile(mesh_part)
        #     mesh_tile.get_mesh().show()
