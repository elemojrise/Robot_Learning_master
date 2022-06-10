# Robot_Learning_master
This repository is used in master thesis for Petter Rasmussen and Ole Jørgen Rise.

The overarching goal of this framework is to make a robot manipulator able to automatically grasp objects with the use of vision-based deep reinforcement learning.

## Installation
The framework has been tested to run with Ubuntu20.04, python3.8. mujoco-py 2.1.2.14 robosuite 1.3.2 and stable_baselines3 1.5.0

## Train on Idun

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
2) Start using the gpu-s with: srun --nodes=1 --cpus-per-task=1 --partition=GPUQ --gres=gpu:A100m40:1 --time=02:00:00 --pty bash
3) Use your custom .sif file and make a singularity container with: singularity shell --nv file_name.sif
4) Run: pip install -r requirements.txt
5) You are now able to train your robots with mujoco environment good luck

## Train and run an RL agent
It is made possible to train an RL agent to perform the robot grasp task with the identical robot at the MANULAB NTNU. The framework has been integrated with the PPO algorithm from Stable_Baselines3. Different controller and training setting can be specified in a **.yaml** file located **config_files**. 

To train an agent run the following code:

```
python3 rl_new.py
```
wether to train a new agent or continue training an alerady existing model is specified in the **.yaml** file.
