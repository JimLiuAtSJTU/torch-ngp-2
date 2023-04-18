import json
import time
import warnings

import numpy as np
import pdb
import torch

from .ray_utils import get_rays, get_ray_directions, get_ndc_rays


BOX_OFFSETS = torch.tensor(
    [[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],

                               device='cuda')



def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

def hash_spatial(coords, log2_hashmap_size):
    '''
    coords: expected to use 3D spatial coordinates.
    log2T:  logarithm of T w.r.t 2
    '''

    assert coords.shape[-1]==3, "expected to use spacial hash function for 3D input"



    primes = [1, 5427713651, 9942695869] # use different 10 digits prime number


    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

def hash_fuse_spatial_T(coords, log2_hashmap_size):
    '''
    coords: expected to use 3D spatial coordinates.
    log2T:  logarithm of T w.r.t 2
    '''

    assert coords.shape[-1]==2, "expected to use spacial fuse hash function for 2D input"



    primes = [6388031233, 7252871531, 4763631751] # use different 10 digits prime number


    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result





def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5*W/np.tan(0.5 * camera_angle_x)

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []

    for frame in camera_transforms["frames"]:
        c2w = torch.FloatTensor(frame["transform_matrix"])
        rays_o, rays_d = get_rays(directions, c2w)
        
        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([1.0,1.0,1.0]), torch.tensor(max_bound)+torch.tensor([1.0,1.0,1.0]))


def get_bbox3d_for_llff(poses, hwf, near=0.0, far=1.0):
    H, W, focal = hwf
    H, W = int(H), int(W)
    
    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []
    poses = torch.FloatTensor(poses)
    for pose in poses:
        rays_o, rays_d = get_rays(directions, pose)
        rays_o, rays_d = get_ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([0.1,0.1,0.0001]), torch.tensor(max_bound)+torch.tensor([0.1,0.1,0.0001]))



'''
function used for getting spatial xyz voxel indx
'''
def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3.
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = torch.tensor(bounding_box,device=xyz.device)

    keep_mask = xyz==torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        warnings.warn("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0],device=grid_size.device)*grid_size


    #bottom_left_idx.unsqueeze(1) B x 1 x d
    #BOX_OFFSETS 1 x (2^d) x d
    # d =3
    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask


def get_3D_block_hashed_coordinates(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis

    return: (0,1) hashed coordinates
    '''
    box_min, box_max = torch.tensor(bounding_box, device=xyz.device)

    keep_mask = xyz == torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        warnings.warn("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution

    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
    #voxel_min_vertex = bottom_left_idx * grid_size + box_min

    hashed_voxel_indices = hash_spatial(bottom_left_idx, log2_hashmap_size) # 0 ~ hashmap_length
    hashed_voxel_indices=hashed_voxel_indices.unsqueeze(-1)
    #hashmap_length=1<<log2_hashmap_size -1 # if log2 size >0 ,definitely >0

    return hashed_voxel_indices,keep_mask

'''
function for fused time hash key-value mapping.
'''
def get_spatial_fused_time_indices(xyz:torch.Tensor, bounding_box, resolution_xyz, log2_hashmap_size_xyz:int,
                                   t:torch.Tensor,
                                   resolution_T:torch.Tensor,
                                   log2_hashmap_size_t:int,
                                   time_bound
                                   ):
    '''
    xyz: 3D coordinates of samples. B x 3
    t: B x 1
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''

    batch_size=xyz.shape[0]
    assert batch_size==t.shape[0]

    xyz_hashed_coordinates,keep_mask=get_3D_block_hashed_coordinates(xyz, bounding_box, resolution_xyz, log2_hashmap_size_xyz)


    t_min, t_max = torch.tensor(time_bound, device=xyz.device)

    t_slice_size = (t_max - t_min) / resolution_T

    bottom_left_idx = torch.floor((t - t_min) / t_slice_size).int()
    t_min_stamp = bottom_left_idx * t_slice_size + t_min

    t_max_stamp = t_min_stamp + torch.tensor([1.0],device=t_slice_size.device)*t_slice_size

    time_OFFSETS = torch.tensor([[0],[1]],device=t.device).unsqueeze(0)


    #bottom_left_idx.unsqueeze(1) B x 1 x d
    #BOX_OFFSETS 1 x 2^d x d
    # d = 1



    time_indices = bottom_left_idx.unsqueeze(1) + time_OFFSETS


    # xyz_hashed_coordinates B x 1  : duplicate to B x 2 x 1
    # time_indices: B X 2 X 1
    xyz_hashed_coordinates=xyz_hashed_coordinates.unsqueeze(-1).repeat([1,2,1])
    gamma_t_coordinates=torch.cat([xyz_hashed_coordinates,time_indices],dim=-1)

    hashed_T_indices=hash_fuse_spatial_T(gamma_t_coordinates, log2_hashmap_size_t)

    #hashed_T_indices: B x 2


    return t_min_stamp, t_max_stamp, hashed_T_indices, keep_mask


if __name__=="__main__":
    with open("data/nerf_synthetic/chair/transforms_train.json", "r") as f:
        camera_transforms = json.load(f)
    
    bounding_box = get_bbox3d_for_blenderobj(camera_transforms, 800, 800)
