import argparse
from pathlib import Path
from typing import List

import h5py
import open3d as o3d
import torch
from torch import Tensor
from utils import unflatten_mlp_params
from pycarus.geometry.pcd import get_o3d_pcd_from_tensor, sample_pcds_from_udfs
from pycarus.learning.models.siren import SIREN
import numpy as np

parser = argparse.ArgumentParser(description='This script is designed to generate point clouds from the weights of an MLP')

parser.add_argument('--input_directory', type = str, required=True, help= 'The path to h5 file containing mlp weights required to generate pointcloud')
parser.add_argument('--number_of_points', type = int, default = 8192, help = 'The number of preferred points within the point cloud (default: 8192)')
parser.add_argument('--refinement_steps', type = int, default = 2, help= 'The number of preferred refinement steps (default: 2)')

args = parser.parse_args()

path_h5 = Path(args.input_directory)
num_points = args.number_of_points
ref_step = args.refinement_steps

directory_pcd_gt = path_h5.parent / 'ground_truth'
directory_pcd_pred = path_h5.parent / 'prediction'

output_dir = [directory_pcd_gt, directory_pcd_pred]

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

path_pcd_pred = output_path(directory_pcd_pred)
path_pcd_gt = output_path(directory_pcd_gt)


with h5py.File(path_h5, "r") as f:
    pcd_gt = torch.from_numpy(np.array(f.get("pcd")))
    params = torch.from_numpy(np.array(f.get("params")))

mlp = SIREN(3, 512, 4, 1)
mlp.load_state_dict(unflatten_mlp_params(params, mlp.state_dict()))

def udfs_func(coords: Tensor, indices: List[int]) -> Tensor:
    pred = torch.sigmoid(mlp(coords)[0])
    pred = 1 - pred
    pred *= 0.1
    return pred

# save ground-truth point cloud

pcd_gt_o3d = get_o3d_pcd_from_tensor(pcd_gt)
o3d.io.write_point_cloud(path_pcd_gt, pcd_gt_o3d)

# save predicted point cloud

pred_pcd = sample_pcds_from_udfs(udfs_func, 1, 4096, (-1, 1), 0.05, 0.02, num_points, ref_step, use_cuda = False)[0]
pcd_pred_o3d = get_o3d_pcd_from_tensor(pred_pcd)
o3d.io.write_point_cloud(path_pcd_pred, pcd_pred_o3d)
