import numpy as np
import trimesh


from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import MeshPartsCfg

from terrain_generator.trimesh_tiles.primitive_course.steps import *



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

def create_central_stepping_cfgs(dim, level):
    width = 0.5
    side_std = 0.2 * level
    height_std = 0.05 * level
    n = 6
    ratio = 0.5 + (1.0 - level) * 0.3
    cfgs = create_stepping(
        MeshPartsCfg(dim=dim), width=width, side_std=side_std, height_std=height_std, n=n, ratio=ratio
    )
    return cfgs

def create_stepping_stones_cfgs(dim, level):
    n = 8
    height_diff = level * 0.6
    height_diff = height_diff 
    array_up = np.zeros((n, n))
    array_down = np.zeros((n, n))
    for i in range(n):
        if i % 2 == 0:
            array_up[i, :] = array_up[i, :] + np.linspace(0, height_diff, n)
            array_down[i, :] = array_down[i, :] + np.linspace(height_diff, 0,  n)
    array_up[:,range(1, n, 2)] = 0.0
    array_down[:,range(1, n, 2)] = 0.0
    array_up = array_up.T
    array_down = array_down.T
    start_and_goal_dim = (dim[0], 0.3, dim[2])
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=start_and_goal_dim,
            array=np.array([[0, 0], [0, 0]]) ,            
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="middle_up",
            dim=dim,
            array=array_up,            
            flips=(),
            weight=0.1,
            minimal_triangles=False,
        ),
        PlatformMeshPartsCfg(
            name="middle_down",
            dim=dim,
            array=array_down,            
            flips=(),
            weight=0.1,
            minimal_triangles=False,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=start_and_goal_dim,
            array=np.array([[0, 0], [0, 0]]) ,            
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

def create_gaps_cfgs(dim ,level, gap_depth=0.3, gap_width=0.3):
    gap_depth = gap_depth + level * 0.2
    gap_width = gap_width + level * 0.2

    start_and_goal_dim = (dim[0], 0.5, dim[2])
    n = int (dim[0]/gap_width)
    middle_n = n//2
    gap_array = np.ones(n,n) * gap_depth
    gap_array[middle_n - 1, :] = 0
    gap_array[middle_n + 1, :] = 0 
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=start_and_goal_dim,
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="gap",
            dim=start_and_goal_dim,
            array=gap_array,            
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="end",
            dim=start_and_goal_dim,
            array=np.array([[1, 1], [1, 1]]) * gap_depth,            
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

def create_middle_steps_cfgs(dim, level, height = 0.4, n = 8):
    height_std = level * 0.1
    n = 8
    array = np.zeros((n, n))
    middle_n = n // 2
    array[middle_n - 1, :] = height + np.random.normal(0, height_std)
    array[middle_n + 1, :] = height + np.random.normal(0, height_std)
    cfgs = (
        PlatformMeshPartsCfg(
            name="step",
            dim=cfg.dim,
            array=array,            
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

def create_ramp_cfgs(dim, level, height_diff = 0.3):
    height_std = level * 0.3  
    height_noise = np.random.normal(0, height_std) 
    height_diff = height_diff + cfg.floor_thickness + height_noise 
    step_height = height_diff / (dim[1] - 1)
    h1 = 0
    array_ramp_up = np.zeros(dim)
    array_ramp_down = np.zeros(dim)
    # Relatively narrow starting and ending platform
    start_and_goal_dim = (dim[0], 0.3, dim[2])
    for s in range(int (dim[0])):
        array_ramp_up[:, s] = h1
        array_ramp_down[:, int (dim[0]) - s] = h1
        h1 = h1 + step_height
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=start_and_goal_dim,
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
        ),
        HeightMapMeshPartsCfg(
            name="ramp_up",
            dim=cfg.dim,
            height_map=array_ramp_up,
            rotations=(),
            flips=(),
            slope_threshold=1.5,
            target_num_faces=1000,
            simplify=True,
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=start_and_goal_dim,
            array=np.array([[1, 1], [1, 1]]) * height_diff,            
            flips=(),
            weight=0.1,
        ),
        HeightMapMeshPartsCfg(
            name="ramp_down",
            dim=cfg.dim,
            height_map=array_ramp_down,
            rotations=(),
            flips=(),
            slope_threshold=1.5,
            target_num_faces=1000,
            simplify=True,
            weight=0.1,
        ),
    )
    return cfgs