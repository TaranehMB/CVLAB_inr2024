import argparse
import h5py
import open3d as o3d
import torch

from pathlib import Path
from torch import Tensor
from utils import unflatten_mlp_params
from pycarus.geometry.mesh import get_o3d_mesh_from_tensors, marching_cubes
from pycarus.learning.models.siren import SIREN
import numpy as np

parser = argparse.ArgumentParser(description='This script is designed to generate meshgrid from the weights of an MLP')

parser.add_argument('--input_directory', type=str, required=True, help= 'The path to h5 file containing mlp weights required to generate the meshgrid')
parser.add_argument('--resolution', type = int, default= 128, help= 'The preferred resolution of the output meshgrid (default: 128)')

args= parser.parse_args()

path_h5 = Path(args.input_directory)

directory_mesh_gt = path_h5.parent / 'ground_truth'
directory_mesh_pred = path_h5.parent / 'prediction'

output_dir = [directory_mesh_gt, directory_mesh_pred]

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

path_mesh_pred = output_path(directory_mesh_pred)
path_mesh_gt = output_path(directory_mesh_gt)


with h5py.File(path_h5, "r") as f:
    gt_v = torch.from_numpy(np.array(f.get("vertices"), dtype=np.float32))
    gt_num_v = torch.from_numpy(np.array(f.get("num_vertices"), dtype=np.int32))
    gt_t = torch.from_numpy(np.array(f.get("triangles"), dtype=np.float32))
    gt_num_t = torch.from_numpy(np.array(f.get("num_triangles"), dtype=np.int32))
    params = torch.from_numpy(np.array(f.get("params")))



mlp = SIREN(3, 512, 4, 1)
mlp.load_state_dict(unflatten_mlp_params(params, mlp.state_dict()))

def levelset_func(c: Tensor) -> Tensor:
    pred = mlp(c)[0].squeeze(-1)
    pred = torch.sigmoid(pred)     
    pred *= 0.2                    
    pred -= 0.1                    
    return pred

res = args.resolution

# save ground-truth point cloud
mesh_gt_o3d = get_o3d_mesh_from_tensors(gt_v[:gt_num_v], gt_t[:gt_num_t])
o3d.io.write_triangle_mesh(path_mesh_gt, mesh_gt_o3d)

pred_v, pred_t = marching_cubes(levelset_func, (-1, 1), res, use_cuda=False)
mesh_pred_o3d = get_o3d_mesh_from_tensors(pred_v, pred_t).translate((2, 0, 0))
o3d.io.write_triangle_mesh(path_mesh_pred, mesh_pred_o3d)