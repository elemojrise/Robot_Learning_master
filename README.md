# Robot_Learning_master
Master repository for robot learning with Petter Rasmussen and Ole Jørgen Rise


## How to train on Idun

### Get custom made .sif file
If you already have a custom made .sif file jump straight to

The mujoco_robo.def file is designed for running code on the the cluster gpu-s with mujoco as physics engine

If not on Idun download singularity on your local machine described in this link: https://sylabs.io/guides/3.0/user-guide/installation.html

Then create a .sif file from the mujoco_robo.def with singularity either remote on Idun or on your local machine.

remote: singularity build -remote file_name.sif mujoco_robo.def
local: singularity build file_name.sif mujoco_robo.def

This might take a while but you should now have your own .sif file

Some erros when creating .sif file

In the %files section remember to change to your local path to the file you want to implement


### Run custom environment on Idun

1) Log in on Idun with: ssh username@idun-login1.hpc.ntnu.no
2) Start using the gpu-s with: srun --nodes=1 --partition=GPUQ --gres=gpu:8 --time=02:00:00 --pty bash
3) Use your custom .sif file and make a singularity container with: singularity shell --nv file_name.sif
4) Run: pip install -r requrements.txt
5) You are now able to train your robots with mujoco environment good luck


