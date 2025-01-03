import argparse
from pathlib import Path
from typing import cast

import h5py
import open3d as o3d
import torch
from einops import rearrange
from torch import Tensor
from pytorch3d.ops import cubify
from utils import unflatten_mlp_params
from pycarus.geometry.pcd import  voxelize_pcd
from pycarus.geometry.mesh import get_o3d_mesh_from_tensors
from pycarus.learning.models.siren import SIREN
import numpy as np

parser = argparse.ArgumentParser(description='This script is designed to generate voxels from the weights of an MLP')

parser.add_argument('--input_directory', type = str, required=True, help= 'The path to h5 file containing mlp weights required to generate voxel')
parser.add_argument('--threshold', type = float, default = 0.5, help = 'The script would consider the probability higher than this threshold as an occupied voxel(default: 0.5)')

args = parser.parse_args()

path_h5 = Path(args.input_directory)
threshold = args.threshold
vox_res = 64

directory_vox_gt = path_h5.parent / 'ground_truth'
directory_vox_pred = path_h5.parent / 'prediction'

output_dir = [directory_vox_gt, directory_vox_pred]

for directory in output_dir:
    if not directory.exists():
        directory.mkdir(parents=True)
        print(f"Directory '{directory}' created.")

def output_path(path_dir):

    path = path_h5
    output_filename = f"{path.stem}.ply"
    output_file_path = path_dir / output_filename
    output_file_string = str(output_file_path)

    return output_file_string


path_vox_pred = output_path(directory_vox_pred)
path_vox_gt = output_path(directory_vox_gt)


with h5py.File(path_h5, "r") as f:
    pcd_vox_gt = torch.from_numpy(np.array(f.get("pcd")))
    params = torch.from_numpy(np.array(f.get("params")))

mlp = SIREN(3, 512, 4, 1)
mlp.load_state_dict(unflatten_mlp_params(params, mlp.state_dict()))

vgrid, centroids = voxelize_pcd(pcd_vox_gt, 64, -1, 1)
vgrid_gt = vgrid.unsqueeze(0)
vgrid_gt_cubified = cubify(vgrid_gt, 0.5, align ="center")
gt_v = cast(Tensor, vgrid_gt_cubified.verts_packed())
gt_t = cast(Tensor, vgrid_gt_cubified.faces_packed())
vgrid_gt_o3d = get_o3d_mesh_from_tensors(gt_v,gt_t)
o3d.io.write_triangle_mesh(path_vox_gt, vgrid_gt_o3d)

with torch.no_grad():

    centr = centroids.unsqueeze(0)
    centr = rearrange(centr, "b r1 r2 r3 d -> b (r1 r2 r3) d")
    
    vgrid_pred = torch.sigmoid(mlp(centr)[0])
    vgrid_pred = vgrid_pred.squeeze(-1)
    
    vgrid_pred= rearrange(vgrid_pred, "b (r1 r2 r3) -> b r1 r2 r3", r1=vox_res, r2=vox_res)


vgrid_pred_cubified = cubify(vgrid_pred, 0.5, align="center")

pred_v = cast(Tensor, vgrid_pred_cubified.verts_packed())
pred_t = cast(Tensor, vgrid_pred_cubified.faces_packed())
vgrid_pred_o3d = get_o3d_mesh_from_tensors(pred_v, pred_t).translate((2,0,0))
o3d.io.write_triangle_mesh(path_vox_pred, vgrid_pred_o3d)