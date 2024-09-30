import numpy as np
import trimesh


from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import MeshPartsCfg

from terrain_generator.trimesh_tiles.primitive_course.steps import *


def create_flat_terrain_cfgs(dim, level=0.0):
    cfgs = (
        PlatformMeshPartsCfg(
            name="flat",
            dim=dim,
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

def create_up_down_stairs_cfgs(dim, level=0.0):   
    """
        dim order (x,y,z)  
        mesh coordinate (x,y,z)
        array order [y,x]
    """ 
    # * types = {straight_up, straight_down, pyramid_up, pyramid_down, up_down, down_up, plus_up, plus_down}
    # * sets the number of stairs by size of the array
    max_height =1.5
    height = level * max_height

    num_stairs = 12        
    stair_dim = (dim[0],dim[1],dim[2])
    cfgs = create_stairs(
        MeshPartsCfg(dim=stair_dim), height_diff=height, num_stairs=num_stairs, type='up_down', flip=True)
    stair_cfg = cfgs[1]

    start_and_end_dim = [dim[0]/4, dim[1], dim[2]]
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=start_and_end_dim,
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
        ),
        stair_cfg,
        PlatformMeshPartsCfg(
            name="goal",
            dim=start_and_end_dim,
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

def create_inclined_surfaces_cfgs(dim, level=0):
    width = 0.3                     # block width in side direction
    side_distance = 0.2             # distane between two blocks in the side direction (measured from center)
    height=0.0   
    step_length = 0.3               # block length in forward direction
    total_length = dim[0]
    max_angle = np.pi/6.0           # angle of inclined surfaces
    angle = level * max_angle
    
    n_block = int (floor(total_length/step_length))
    
    # Create surface blocks
    box_dims = []
    transformations = []
    thickness = 0.06    # thickness of the inclined surfaces

    for i in range(n_block):
        # First platform_
        if i == floor(n_block/2) - 1:
            box_dims.append([step_length + width/2, width, thickness])
            x = -total_length / 2.0 + (i + 0.5) * step_length + width / 4
        else:
            box_dims.append([step_length+ width/2, width, thickness])
            x = -total_length / 2.0 + (i + 0.5) * step_length
        y = side_distance*(-1)**(i%2)
        z = thickness / 2.0 + height
        t = trimesh.transformations.rotation_matrix(-angle*(-1)**(i%2+1), [1, 0, 0])
        t[:3, -1] = np.array([x, y, z])
        transformations.append(t)

    # Create terrain config
    start_and_end_dim = [0.3, dim[1], dim[2]]
    surface_dim = [total_length, dim[1], dim[2]]
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=start_and_end_dim,
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
        ),
        BoxMeshPartsCfg(
            name="surface",
            dim=tuple(surface_dim),
            box_dims=tuple(box_dims),
            transformations=tuple(transformations),            
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
        ),
        PlatformMeshPartsCfg(
            name="goal",
            dim=start_and_end_dim,
            array=np.array([[0, 0], [0, 0]]),            
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

   
def create_single_beam_cfgs(dim, level):
    width = 0.25
    thickness = 0.2
    box_dims = [[0.8*dim[0], width, thickness]]

    height=level * 0.4
    transformations = []
    pitch = 0.0
    t = trimesh.transformations.rotation_matrix(pitch, [1, 0, 0])
    t[:3, -1] = np.array([0.0, 0.0, height])
    transformations.append(t)
    
    cfgs = (
        BoxMeshPartsCfg(
            name="beam",
            dim=dim,
            transformations=tuple(transformations),            
            box_dims=box_dims,            
            rotations=(0.0, 0.0, 0.0),            
            weight=0.1,
            minimal_triangles=False,
            add_floor=True,
        ),
    )
    return cfgs

def create_passage_cfgs(dim, level):
    """
        dim order (x,y,z)  
        mesh coordinate (x,y,z)
        array order [x,y]
    """
    width = 0.5
    side_std = 0.2 * level
    height_std = 0.05 * level
    n = 6
    ratio = 0.5 + (1.0 - level) * 0.3
    
    step_length = dim[0] / n * ratio
    dims = []
    transformations = []
    floor_thickness = 0.1
    for i in range(n):
        # First platform_
        dims.append([step_length, width, floor_thickness])
        t = np.eye(4)
        y = np.random.normal(0, side_std)
        x = -dim[0] / 2.0 + (i + 0.5) * dim[0] / n
        z = floor_thickness / 2.0 + np.random.normal(0, height_std)
        t[:3, -1] = np.array([x, y, z])
        transformations.append(t)

    start_and_goal_dim = (0.3, dim[1], dim[2])
    cfgs = (
        PlatformMeshPartsCfg(
            name="start",
            dim=start_and_goal_dim,
            array=np.array([[0, 0], [0, 0]]) ,            
            flips=(),
            weight=0.1,
        ),
        BoxMeshPartsCfg(
            name="middle",
            dim=dim,
            box_dims=tuple(dims),
            transformations=tuple(transformations),          
            weight=0.1,
            minimal_triangles=False,
            add_floor=False,
        ),
        PlatformMeshPartsCfg(
            name="end",
            dim=start_and_goal_dim,
            array=np.array([[0, 0], [0, 0]]) ,            
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs
    

def create_stepping_stones_cfgs(dim, level):
    """
        dim order (x,y,z)  
        mesh coordinate (x,y,z)
        array order [x,y]
    """
    n = 11
    height_diff = level * 1.0
    height_diff = height_diff 
    array_up = np.zeros((n, n))
    array_down = np.zeros((n, n))
    for i in range(n):
        if i % 2 == 0:            
            array_up[:, i] = array_up[:, i] + np.linspace(height_diff/n, height_diff, n)            
            array_down[:, i] = array_down[:, i] + np.linspace(height_diff, height_diff/n, n)
    array_up[range(0, n, 2),:] = 0.0    
    array_down[range(0, n, 2), :] = 0.0    
    array_up = array_up.T    
    array_down = array_down.T    
    start_and_goal_dim = (0.3, dim[1], dim[2])
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

def create_gaps_cfgs(dim ,level, gap_depth=0.4, gap_width=0.8):
    """
        dim order (x,y,z)  
        mesh coordinate (x,y,z)
        array order [x,y]
    """
    gap_depth = gap_depth + level * 0.2
    gap_width = gap_width + level * 0.5
    n = 20
    
    n_per_gap = int (gap_width/(dim[0]/n))    
    gap_array = np.ones((n,n)) * gap_depth
    middle_n = n//2    
    gap_array[middle_n - n_per_gap//2 : middle_n + n_per_gap//2, :] = 0
    
    cfgs = (
        PlatformMeshPartsCfg(
            name="gap",
            dim=dim,
            array=gap_array.transpose(),            
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs

def create_middle_steps_cfgs(dim, level, height = 0.1, n = 9):
    """
        dim order (x,y,z)  
        mesh coordinate (x,y,z)
        array order [x,y]
    """
    height_offset = level * 0.4
    array = np.zeros((n, n))
    middle_n = n // 2
    array[middle_n - 2, :] = height + height_offset
    array[middle_n + 2, :] = height + height_offset
    cfgs = (
        PlatformMeshPartsCfg(
            name="step",
            dim=dim,
            array=array.transpose(),            
            flips=(),
            weight=0.1            
        ),
    )
    return cfgs

def create_ramp_cfgs(dim, level, height_diff = 0.3):   
    """
        dim order (x,y,z)  
        mesh coordinate (x,y,z)
        array order [x,y]
    """   
    n = 20
    dim_square = (dim[0], dim[0], dim[2])
    height_diff = height_diff + level * 0.3  + 0.1
    step_height = height_diff / (n - 1)    
    array_ramp_up = np.zeros((n,n))
    array_ramp_down = np.zeros((n,n))
    # Relatively narrow starting and ending platform
    start_and_goal_dim = (0.4, dim[1], dim[2])
    h1 = 0.0
    for s in range(n):
        array_ramp_up[s, :] = h1
        array_ramp_down[n - s - 1, :] = np.max((h1, 0.0))
        h1 = h1 + step_height
    cfgs = (        
        HeightMapMeshPartsCfg(
            name="ramp_up",
            dim=dim_square,
            height_map=array_ramp_up,
            rotations=(),
            flips=(),
            slope_threshold=1.5,
            target_num_faces=1000,
            simplify=True,
            weight=0.1,
            add_floor=False            
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
            dim=dim_square,
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

def calculate_length_of_cfg(cfgs, axis = 0):
    l = 0.0
    for cfg in cfgs:
        l = l + cfg.dim[axis]
    return l