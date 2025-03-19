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

Or alternatively, in case you want to have your virtual environment operate a specific version of python compatible with 'pycarus', you can make the virtual environment via conda (by either having conda or miniconda installed) as follows:
```
$ conda create -n env-name python=3.8.6
$ conda activate env-name
$ pip install pycarus
```
