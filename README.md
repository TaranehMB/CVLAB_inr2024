# Visualizing INRs of Shapes

This repository contains codes for visualizing implicit nerual representations (INRs) of 3D shapes, specifically point clouds, meshes and voxels. 

In the following sections, I will provide a step-by-step guide for utilizing this repository effectiely. 

---

The code contained in this repository has been tested on ubuntu 22.04 with python 3.8.6.

## Setup

The setup necessary for running the codes within this repository, follows the same procedure found in [CVlab official **inr2vec** repository](https://github.com/CVLAB-Unibo/inr2vec).

First, create a virtual environment for installing `pycarus` library. This procedure can be directly done via python's API, following the procedure below:
```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip setuptools
$ pip install pycarus
```

Or alternatively, in case you want to need your virtual environment operate a specific version of python compatible with `pycarus`, you can make the virtual environment via conda (by either having conda or miniconda installed) as follows:
```
$ conda create -n env-name python=3.8.6
$ conda activate env-name
$ pip install pycarus
```
Then, by trying to import `pycarus`, you would get the command for installing torch:
```
$ import pycarus
...
ModuleNotFoundError: PyTorch is not installed. Install it by running: source /XXX/.venv/lib/python3.8/site-packages/pycarus/install_torch.sh
```
The script following the Module error, downloads and installs the wheels for torch, torchvision, pytorch3d and torch-geometric. In case there is an error regarding the installation of pytorch3d, it can be installed manually. 

## Visualizing the INRs
The main branch of the repository, contains three seperate codes, each for visualizing a different implicit neural representation of a 3D shape, along with a `utils.py`
script which the visualization codes import to operate effectively. 
The three folders present in the repository, each contain a specific amount of INRs trained on a different 3D shape. I will now get into the details of executing the codes provided on either the available dataset or an alternative dataset with the same structure. 

### Visualizing INRs of meshgrids from manifold40 dataset
