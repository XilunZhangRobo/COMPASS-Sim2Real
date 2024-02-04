import argparse
import json
import os
import time

import gym
import numpy as np
import imageio
import copy
from tqdm import tqdm
import robosuite as suite
from robosuite.wrappers import GymWrapper
import robosuite.utils.transform_utils as trans
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from robosuite import load_controller_config
from stable_baselines3.common.vec_env.subproc_vec_env import  SubprocVecEnv, _flatten_obs
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
import cv2
import argparse 
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict
import pickle

from env_pusher import PusherSingleAction

from env_drop import DropSingleAction
from Pose_run import PoseChanger

""" 
=== Env Params and Clip Range ===
"""
CLIP_RANGE = {
            "dynamic@plate1_g0@friction_sliding": [0.01, 0.15],
            "dynamic@plate2_g0@friction_sliding": [0.01, 0.15],
            "dynamic@knob_g0@damping_ratio": [-50, -3],
            "dynamic@plate1_g0@damping_ratio": [-50, -3],
            "dynamic@plate2_g0@damping_ratio": [-50, -3],
            "dynamic@x_right_wall_g0@damping_ratio": [-50, -3],
            "dynamic@x_left_wall_g0@damping_ratio": [-50, -3],
            "dynamic@y_front_wall_g0@damping_ratio": [-50, -3],
            "dynamic@y_back_wall_g0@damping_ratio": [-50, -3],
            "dynamic@block_g0@damping_ratio": [-50, -3],
            "init@knob@velocity_delay": [0.01, 1.0],
            
            "dynamic@env@density": [0.1, 1.5],
            "dynamic@env@viscosity": [1.5e-5, 2e-4], 
            
            # Inertia for plate 1  
            "dynamic@plate1_main@inertia_ixx": [1e-07,4e-05],
            "dynamic@plate1_main@inertia_iyy": [1e-07,4e-05],
            "dynamic@plate1_main@inertia_izz": [1e-07,4e-05],
            
            # Inertia for plate 2 
            "dynamic@plate2_main@inertia_ixx": [1e-07,4e-05],
            "dynamic@plate2_main@inertia_iyy": [1e-07,4e-05],
            "dynamic@plate2_main@inertia_izz": [1e-07,4e-05],

            # Inertia for the knob
            "dynamic@knob_main@inertia_ixx": [1e-07,4e-05],
            "dynamic@knob_main@inertia_iyy": [1e-07,4e-05],
            "dynamic@knob_main@inertia_izz": [1e-07,4e-05],
            
            # Inertia for step_stone 
            "dynamic@step_stone_main@inertia_ixx": [1e-07, 1e-02],
            "dynamic@step_stone_main@inertia_iyy": [1e-07, 1e-02],
            "dynamic@step_stone_main@inertia_izz": [1e-07, 2e-02],
            
            # for drop env 
            "dynamic@ball_g0@damping_ratio": [-50, -5],
            "dynamic@step_stone_g0@damping_ratio": [-50, -3],
            "dynamic@step_stone_2_g0@damping_ratio": [-60, -3],
            # "dynamic@ball_g0@damping_ratio": [-12, -8],
            # "dynamic@step_stone_g0@damping_ratio": [-15, -11],
            # "dynamic@step_stone_2_g0@damping_ratio": [-21, -16],
            "dynamic@ball_main@mass": [0.079, 0.085],
            "dynamic@step_stone_main@mass": [0.05, 0.15],
            "dynamic@step_stone_2_main@mass": [0.4, 0.7],
            }

full_interested_context_pose = [   
            "init@knob@velocity_delay",       
            "dynamic@env@density",
            "dynamic@env@viscosity",
            "lighting@light@pos_x",
            "lighting@light@pos_y",
            "lighting@light@pos_z",
            "lighting@light@ambient_r",
            "lighting@light@ambient_g",
            "lighting@light@ambient_b",
            "lighting@light@active", 
            
            ## Table Context
            "dynamic@table@inertia_ixx",
            "dynamic@table@inertia_iyy",
            "dynamic@table@inertia_izz",
            "dynamic@table@mass",
            "texture@table_visual@geom_r",
            "texture@table_visual@geom_g",
            "texture@table_visual@geom_b",
            "texture@table_visual@material_reflectance",
            "texture@table_visual@material_shininess",
            "texture@table_visual@material_specular",
            
            ## Cube Context 
            "dynamic@cube_main@inertia_ixx",
            "dynamic@cube_main@inertia_iyy",
            "dynamic@cube_main@inertia_izz",
            "dynamic@cube_main@mass",
            "dynamic@cube_g0@friction_sliding",
            "dynamic@cube_g0@friction_torsional",
            "dynamic@cube_g0@friction_rolling",
            "dynamic@cube_g0@damping_ratio",
            "texture@cube_g0_vis@geom_r",
            "texture@cube_g0_vis@geom_g",
            "texture@cube_g0_vis@geom_b",
            
            ## Gripper_hand 
            "texture@gripper0_hand_visual@geom_r",
            "texture@gripper0_hand_visual@geom_g",
            "texture@gripper0_hand_visual@geom_b",
            "dynamic@gripper0_hand_collision@damping_ratio",
            "dynamic@gripper0_hand_collision@friction_sliding",
            "dynamic@gripper0_hand_collision@friction_torsional",
            "dynamic@gripper0_hand_collision@friction_rolling",
            
            ## gripper0_left_outer_knuckle
            "dynamic@gripper0_left_outer_knuckle@inertia_ixx",
            "dynamic@gripper0_left_outer_knuckle@inertia_iyy",
            "dynamic@gripper0_left_outer_knuckle@inertia_izz",
            "dynamic@gripper0_left_outer_knuckle@mass",
            "dynamic@gripper0_left_outer_knuckle_collision@friction_sliding",
            "dynamic@gripper0_left_outer_knuckle_collision@friction_torsional",
            "dynamic@gripper0_left_outer_knuckle_collision@friction_rolling",
            "dynamic@gripper0_left_outer_knuckle_collision@damping_ratio",
            "dynamic@gripper0_left_outer_knuckle_visual@geom_r",
            "dynamic@gripper0_left_outer_knuckle_visual@geom_g",
            "dynamic@gripper0_left_outer_knuckle_visual@geom_b",
            
            ## gripper0_left_inner_knuckle
            "dynamic@gripper0_left_inner_knuckle@inertia_ixx",
            "dynamic@gripper0_left_inner_knuckle@inertia_iyy",
            "dynamic@gripper0_left_inner_knuckle@inertia_izz",
            "dynamic@gripper0_left_inner_knuckle@mass",
            "dynamic@gripper0_left_inner_knuckle_collision@friction_sliding",
            "dynamic@gripper0_left_inner_knuckle_collision@friction_torsional",
            "dynamic@gripper0_left_inner_knuckle_collision@friction_rolling",
            "dynamic@gripper0_left_inner_knuckle_collision@damping_ratio",
            "dynamic@gripper0_left_inner_knuckle_visual@geom_r",
            "dynamic@gripper0_left_inner_knuckle_visual@geom_g",
            "dynamic@gripper0_left_inner_knuckle_visual@geom_b",
            
            ## gripper0_left_outer_finger 
            "dynamic@gripper0_left_outer_finger_collision@friction_sliding",
            "dynamic@gripper0_left_outer_finger_collision@friction_torsional",
            "dynamic@gripper0_left_outer_finger_collision@friction_rolling",
            "dynamic@gripper0_left_outer_finger_collision@damping_ratio",
            "dynamic@gripper0_left_outer_finger_visual@geom_r",
            "dynamic@gripper0_left_outer_finger_visual@geom_g",
            "dynamic@gripper0_left_outer_finger_visual@geom_b",
            
            ## gripper0_left_inner_finger 
            "dynamic@gripper0_left_inner_finger@inertia_ixx",
            "dynamic@gripper0_left_inner_finger@inertia_iyy",
            "dynamic@gripper0_left_inner_finger@inertia_izz",
            "dynamic@gripper0_left_inner_finger@mass",
            "texture@gripper0_left_inner_finger_visual@geom_r",
            "texture@gripper0_left_inner_finger_visual@geom_g",
            "texture@gripper0_left_inner_finger_visual@geom_b",
            "dynamic@gripper0_left_inner_finger_collision@dampling_ratio",
            "dynamic@gripper0_left_inner_finger_collision@friction_sliding",
            "dynamic@gripper0_left_inner_finger_collision@friction_torsional",
            "dynamic@gripper0_left_inner_finger_collision@friction_rolling",
            
            ## gripper0_right_outer_knuckle
            "dynamic@gripper0_right_outer_knuckle@inertia_ixx",
            "dynamic@gripper0_right_outer_knuckle@inertia_iyy",
            "dynamic@gripper0_right_outer_knuckle@inertia_izz",
            "dynamic@gripper0_right_outer_knuckle@mass",
            "dynamic@gripper0_right_outer_knuckle_collision@friction_sliding",
            "dynamic@gripper0_right_outer_knuckle_collision@friction_torsional",
            "dynamic@gripper0_right_outer_knuckle_collision@friction_rolling",
            "dynamic@gripper0_right_outer_knuckle_collision@damping_ratio",
            "dynamic@gripper0_right_outer_knuckle_visual@geom_r",
            "dynamic@gripper0_right_outer_knuckle_visual@geom_g",
            "dynamic@gripper0_right_outer_knuckle_visual@geom_b",
            
            ## gripper0_right_inner_knuckle
            "dynamic@gripper0_right_inner_knuckle@inertia_ixx",
            "dynamic@gripper0_right_inner_knuckle@inertia_iyy",
            "dynamic@gripper0_right_inner_knuckle@inertia_izz",
            "dynamic@gripper0_right_inner_knuckle@mass",
            "dynamic@gripper0_right_inner_knuckle_collision@friction_sliding",
            "dynamic@gripper0_right_inner_knuckle_collision@friction_torsional",
            "dynamic@gripper0_right_inner_knuckle_collision@friction_rolling",
            "dynamic@gripper0_right_inner_knuckle_collision@damping_ratio",
            "dynamic@gripper0_right_inner_knuckle_visual@geom_r",
            "dynamic@gripper0_right_inner_knuckle_visual@geom_g",
            "dynamic@gripper0_right_inner_knuckle_visual@geom_b",
            
            
            ## gripper0_right_outer_finger 
            "dynamic@gripper0_right_outer_finger_collision@friction_sliding",
            "dynamic@gripper0_right_outer_finger_collision@friction_torsional",
            "dynamic@gripper0_right_outer_finger_collision@friction_rolling",
            "dynamic@gripper0_right_outer_finger_collision@damping_ratio",
            "dynamic@gripper0_right_outer_finger_visual@geom_r",
            "dynamic@gripper0_right_outer_finger_visual@geom_g",
            "dynamic@gripper0_right_outer_finger_visual@geom_b",
            
            ## gripper0_right_inner_finger 
            "dynamic@gripper0_right_inner_finger@inertia_ixx",
            "dynamic@gripper0_right_inner_finger@inertia_iyy",
            "dynamic@gripper0_right_inner_finger@inertia_izz",
            "dynamic@gripper0_right_inner_finger@mass",
            "texture@gripper0_right_inner_finger_visual@geom_r",
            "texture@gripper0_right_inner_finger_visual@geom_g",
            "texture@gripper0_right_inner_finger_visual@geom_b",
            "dynamic@gripper0_right_inner_finger_collision@dampling_ratio",
            "dynamic@gripper0_right_inner_finger_collision@friction_sliding",
            "dynamic@gripper0_right_inner_finger_collision@friction_torsional",
            "dynamic@gripper0_right_inner_finger_collision@friction_rolling",
            
            
            ]


full_interested_context_drop = [
            "dynamic@env@density",
            "dynamic@env@viscosity",
            "lighting@light@pos_x",
            "lighting@light@pos_y",
            "lighting@light@pos_z",
            "lighting@light@ambient_r",
            "lighting@light@ambient_g",
            "lighting@light@ambient_b",
            "lighting@light@active",
            
            
            ## Ball related contexts
            "dynamic@ball_g0@friction_sliding",
            "dynamic@ball_g0@friction_torsional",
            "dynamic@ball_g0@friction_rolling",
            "dynamic@ball_g0@damping_ratio",
            "dynamic@ball_main@inertia_ixx",
            "dynamic@ball_main@inertia_iyy",
            "dynamic@ball_main@inertia_izz",
            "dynamic@ball_main@mass",
            
            ## Step Stone related contexts
            "dynamic@step_stone_g0@friction_sliding",
            "dynamic@step_stone_g0@friction_torsional",
            "dynamic@step_stone_g0@friction_rolling",
            "dynamic@step_stone_g0@damping_ratio",
            "dynamic@step_stone_main@inertia_ixx",
            "dynamic@step_stone_main@inertia_iyy",
            "dynamic@step_stone_main@inertia_izz",
            "dynamic@step_stone_main@mass",
            
            ## Step Stone2 related contexts
            "dynamic@step_stone_2_g0@friction_sliding",
            "dynamic@step_stone_2_g0@friction_torsional",
            "dynamic@step_stone_2_g0@friction_rolling",
            "dynamic@step_stone_2_g0@damping_ratio",
            "dynamic@step_stone_2_main@inertia_ixx",
            "dynamic@step_stone_2_main@inertia_iyy",
            "dynamic@step_stone_2_main@inertia_izz",
            "dynamic@step_stone_2_main@mass",
            
            ## Tabel Context
            "dynamic@table@inertia_ixx",
            "dynamic@table@inertia_iyy",
            "dynamic@table@inertia_izz",
            "dynamic@table@mass",
            "dynamic@table_collision@damping_ratio",
            
            ## Robot Context
            "dynamic@robot0_base@inertia_ixx",
            "dynamic@robot0_base@inertia_iyy",
            "dynamic@robot0_base@inertia_izz",
            "dynamic@robot0_base@mass",
            "dynamic@robot0_shoulder_link@inertia_ixx",
            "dynamic@robot0_shoulder_link@inertia_iyy",
            "dynamic@robot0_shoulder_link@inertia_izz",
            "dynamic@robot0_shoulder_link@mass",
            "dynamic@robot0_HalfArm1_Link@inertia_ixx",
            "dynamic@robot0_HalfArm1_Link@inertia_iyy",
            "dynamic@robot0_HalfArm1_Link@inertia_izz",
            "dynamic@robot0_HalfArm1_Link@mass",
            "dynamic@robot0_HalfArm2_Link@inertia_ixx",
            "dynamic@robot0_HalfArm2_Link@inertia_iyy",
            "dynamic@robot0_HalfArm2_Link@inertia_izz",
            "dynamic@robot0_HalfArm2_Link@mass",
            "dynamic@robot0_forearm_link@inertia_ixx",
            "dynamic@robot0_forearm_link@inertia_iyy",
            "dynamic@robot0_forearm_link@inertia_izz",
            "dynamic@robot0_forearm_link@mass",
            
            ## Texture Context
            "texture@floor@geom_r", 
            "texture@floor@geom_g",
            "texture@floor@geom_b",
            "texture@floor@material_reflectance",
            "texture@floor@material_shininess",
            "texture@floor@material_specular",
            "texture@table_visual@geom_r",
            "texture@table_visual@geom_g",
            "texture@table_visual@geom_b",
            "texture@table_visual@material_reflectance",
            "texture@table_visual@material_shininess",
            "texture@table_visual@material_specular",
            "texture@ball_g0_vis@geom_r",
            "texture@ball_g0_vis@geom_g",
            "texture@ball_g0_vis@geom_b",
            "texture@ball_g0_vis@material_reflectance",
            "texture@ball_g0_vis@material_shininess",
            "texture@ball_g0_vis@material_specular",
            "texture@step_stone_g0_visual@geom_r",
            "texture@step_stone_g0_visual@geom_g",
            "texture@step_stone_g0_visual@geom_b",
            "texture@step_stone_2_g0_visual@geom_r",
            "texture@step_stone_2_g0_visual@geom_g",
            "texture@step_stone_2_g0_visual@geom_b",
        ]

full_interested_context_pusher = [
        "init@knob@velocity_delay",
        "dynamic@env@density",
        "dynamic@env@viscosity",
        "dynamic@plate1_g0@friction_sliding",
        "dynamic@plate2_g0@friction_sliding",
        "dynamic@plate1_g0@friction_torsional",
        "dynamic@plate2_g0@friction_torsional",
        "dynamic@plate1_g0@friction_rolling",
        "dynamic@plate2_g0@friction_rolling",
        "lighting@light@pos_x",
        "lighting@light@pos_y",
        "lighting@light@pos_z",
        "lighting@light@ambient_r",
        "lighting@light@ambient_g",
        "lighting@light@ambient_b",
        # "dynamic@plate1_main@mass",
        # "dynamic@plate2_main@mass",
        # "dynamic@knob_main@mass",
        "dynamic@knob_g0@damping_ratio",
        "dynamic@plate1_g0@damping_ratio",
        "dynamic@plate2_g0@damping_ratio",
        
        # Inertia for plate 1  
        "dynamic@plate1_main@inertia_ixx",
        "dynamic@plate1_main@inertia_iyy",
        "dynamic@plate1_main@inertia_izz",
        
        # Inertia for plate 2 
        "dynamic@plate2_main@inertia_ixx",
        "dynamic@plate2_main@inertia_iyy",
        "dynamic@plate2_main@inertia_izz",
        
        # Inertia for the knob 
        "dynamic@knob_main@inertia_ixx",
        "dynamic@knob_main@inertia_iyy",
        "dynamic@knob_main@inertia_izz",
        
        # === [May 16] Camera Shift ===
        "init@camera@shift_x",
        "init@camera@shift_y",
        
        # == After env modification, the following context is addedd ==
        # Wall Inertia Context
        "dynamic@x_right_wall_main@inertia_ixx",
        "dynamic@x_right_wall_main@inertia_iyy",
        "dynamic@x_right_wall_main@inertia_izz",
        "dynamic@x_left_wall_main@inertia_ixx",
        "dynamic@x_left_wall_main@inertia_iyy",
        "dynamic@x_left_wall_main@inertia_izz",
        "dynamic@y_front_wall_main@inertia_ixx",
        "dynamic@y_front_wall_main@inertia_iyy",
        "dynamic@y_front_wall_main@inertia_izz",
        "dynamic@y_back_wall_main@inertia_ixx",
        "dynamic@y_back_wall_main@inertia_iyy",
        "dynamic@y_back_wall_main@inertia_izz",
        
        ## Wall Sliding Friction Context 
        "dynamic@x_right_wall_g0@friction_sliding",
        "dynamic@x_left_wall_g0@friction_sliding",
        "dynamic@y_front_wall_g0@friction_sliding",
        "dynamic@y_back_wall_g0@friction_sliding",
        
        ## Wall torsional friction context
        "dynamic@x_right_wall_g0@friction_torsional",
        "dynamic@x_left_wall_g0@friction_torsional",
        "dynamic@y_front_wall_g0@friction_torsional",
        "dynamic@y_back_wall_g0@friction_torsional",
        
        ## Wall rolling friction context
        "dynamic@x_right_wall_g0@friction_rolling",
        "dynamic@x_left_wall_g0@friction_rolling",
        "dynamic@y_front_wall_g0@friction_rolling",
        "dynamic@y_back_wall_g0@friction_rolling",
        
        ## Wall damping context 
        "dynamic@x_right_wall_g0@damping_ratio",
        "dynamic@x_left_wall_g0@damping_ratio",
        "dynamic@y_front_wall_g0@damping_ratio",
        "dynamic@y_back_wall_g0@damping_ratio",
        
        
        # ## Obstacle context 
        # "dynamic@obstacle_main@mass",
        "dynamic@block_main@inertia_ixx",
        "dynamic@block_main@inertia_iyy",
        "dynamic@block_main@inertia_izz",
        "dynamic@block_g0@friction_sliding",
        "dynamic@block_g0@friction_torsional",
        "dynamic@block_g0@friction_rolling",
        "dynamic@block_g0@damping_ratio",
    ]


""" 
=== Prime Public Methods ===
"""

def init_env(args, n_envs=1):
    # Parallel environments
    def make_env(seed):
        def _make_env():
            if args.exp_name == "pusher":
                env = PusherSingleAction() 
            elif args.exp_name == "drop":
                env = DropSingleAction() 
            else:
                env = PoseChanger()
            env.seed(seed)
            env.action_space.seed(seed)
            return env
        return _make_env

    if n_envs < 0:
        return make_env(0)()
    elif n_envs == 1:
        # return DummyVecEnv([make_env(i) for i in range(n_envs)])
        return DummyVecEnv([make_env(args.seed) for i in range(n_envs)])
    else:
        # return CustomSubprocVecEnv([make_env(i) for i in range(n_envs)])
        return CustomSubprocVecEnv([make_env(args.seed) for i in range(n_envs)])


def init_e_dr_sim(args, env):
    if isinstance(env, (SubprocVecEnv, DummyVecEnv)):
        # Handle parallelized envs
        full_context = env.env_method("get_context", indices=0)[0]
    else:
        full_context = env.get_context()
    
    if args.exp_name == "pusher":
        return init_e_dr_sim_pusher(full_context)
    elif args.exp_name == "drop":
        return init_e_dr_sim_drop(full_context)
    elif args.exp_name == "pose":
        return init_e_dr_sim_pose(full_context)
    else:
        raise NotImplementedError
    
    
def init_interested_context(args):
    if args.exp_name == "pusher":
        return full_interested_context_pusher
    elif args.exp_name == "drop":
        return full_interested_context_drop
    
    elif args.exp_name == "pose":
        return full_interested_context_pose
    else:
        raise NotImplementedError

def init_e_dr_real(args, e_dr_sim):
    # Make a deep copy of the simulation context!!
    e_dr_real = copy.deepcopy(e_dr_sim)
    
    if args.exp_name == "pusher":
        return init_e_dr_real_pusher(e_dr_real)
    elif args.exp_name == "drop":
        return init_e_dr_real_drop(e_dr_real)
    elif args.exp_name == "pose":
        return init_e_dr_real_pose(e_dr_real)
    else:
        raise NotImplementedError


def log_performance_on_current_e_dr(args, k, single_env, causual_DR_writer, e_dr_sim, e_dr_real, command_act_list, real_traj_act_load):
    traj_real_all = []
    traj_sim_all = []
    delta_tau_Baseline = []
    act_real_traj = []
    for j in range(args.real_rollout):
        # Rollout 1 trajectory based on the real environment + agent
        if args.load_real_dataset:
            index = j
            traj_real, act_real = [real_traj_act_load['traj_real'][index]], [[real_traj_act_load['act_real'][index]]]
        else:
            traj_real, act_real = rollout(single_env, args, e_dr_list=[e_dr_real], real=False, return_act=True, command_act=command_act_list[j])

        # Rollout 1 trajectory with e_dr_sim + agent
        traj_sim = rollout(single_env, args, e_dr_list=[e_dr_sim], command_act=act_real[0][0])
        
        traj_real_all.append(traj_real)
        traj_sim_all.append(traj_sim)
        act_real_traj.append(act_real)
        # Check early stopping condition
        delta_tau_Baseline.append(compute_delta_tau(k, args, causual_DR_writer, traj_real, traj_sim_e_dr=[traj_sim])) 

    
    # Save the trajectory
    save_data(k, args, {'traj_sim_all':traj_sim_all, 'traj_real_all': traj_real_all}, 'traj')
    # plot the trajectory toward the tensorboard
    plot_traj(k, args, causual_DR_writer, traj_real_all, traj_sim_all, task=args.exp_name)
    # Add the statistics of delta_tau to the tensorboard
    delta_tau_Baseline = np.array(delta_tau_Baseline)
    causual_DR_writer.add_scalar('delta/delta_tau_mean[0]', delta_tau_Baseline.mean(axis=0)[0, 0], k)
    causual_DR_writer.add_scalar('delta/delta_tau_max[0]', delta_tau_Baseline.max(axis=0)[0, 0], k)
    causual_DR_writer.add_scalar('delta/delta_tau_min[0]', delta_tau_Baseline.min(axis=0)[0, 0], k)
    if args.exp_name == "pusher" or args.exp_name == "pose":
        causual_DR_writer.add_scalar('delta/delta_tau_mean[1]', delta_tau_Baseline.mean(axis=0)[0, 1], k)
        causual_DR_writer.add_scalar('delta/delta_tau_max[1]', delta_tau_Baseline.max(axis=0)[0, 1], k)
        causual_DR_writer.add_scalar('delta/delta_tau_min[1]', delta_tau_Baseline.min(axis=0)[0, 1], k)
        
        
    return traj_real_all, act_real_traj


""" 
=== Secondary Public Methods ===
"""

def rollout(env, args, e_dr_list=[{}], real=False, return_act=False, command_act=None):
    print("Rolling out One or Multiple Trajectory on Sim Robot")
    traj = [[] for _ in e_dr_list]
    bar = tqdm(range(len(e_dr_list)))
    
    # Num of parallel execution env
    n_envs = max(1, args.n_envs)
    n_envs = min(n_envs, len(e_dr_list))
    
    act = []
    for batch_idx in range(0, len(e_dr_list), n_envs):

        batch_context = e_dr_list[batch_idx: batch_idx + n_envs]
        for i in range(n_envs):
            # set context of each environment
            if isinstance(env, (SubprocVecEnv, DummyVecEnv)):
                env.env_method("set_context", batch_context[i], indices=i)
            else:
                env.set_context(batch_context[i])
                
            # set random seed of each env
            if isinstance(env, (SubprocVecEnv, DummyVecEnv)):
                env.env_method("seed", 0, indices=i)
            else:
                # Using the same seed for all envs -- seed=0
                env.seed(0)

        # set random seed of each environment
        env.reset()
        
        env.num_envs = n_envs
        
        if command_act is not None: 
            if args.load_real_dataset and args.exp_name == "drop":
                offset = 0.8
                actions = [command_act + offset for _ in range(n_envs)]
            else:
                actions = [command_act for _ in range(n_envs)]
            
        else:
            actions = [env.action_space.sample() for _ in range(n_envs)]
            
        for action in actions:
            # assert actoin is not all zeros
            assert (action != np.zeros(4)).any(), "Action is all zeros!"
            
        act.append(actions)
        _, _, _, infos = env.step(actions)
        for i in range(n_envs):
            if batch_idx + i >=  len(e_dr_list):
                print("Data collection is done!")
                break
            else:
                #TODO: change info traj 
                if args.exp_name == "pusher":
                    traj[batch_idx + i] = [infos[i]["plate1_pos"], infos[i]["plate2_pos"]]
                elif args.exp_name == "drop":
                    traj[batch_idx + i] = [infos[i]["pos"]]
                elif args.exp_name == "pose":
                    a = infos[i]["pos"]
                    b = [np.asarray(trans.quat2euler(infos[i]["quat"][0]))]
                    traj[batch_idx + i] = [a, b]
                else:
                    raise NotImplementedError
        bar.update(n_envs)


    if return_act:
        return traj, act
    else:
        return traj
 

def compute_delta_tau(iter, args, writer, traj_real=[], traj_sim_e_dr=[[]]):
    """
    Input:
        traj_real.shape = (1, dim_y, 100, 3)
        traj_sim_e_dr.shape = (1, n_sim, dim_y, 100, 3)
    
    Intermediate Variable:
        delta_tau_raw.shape = (n_sim, dim_y, 100, 3)
        
    Output:
        delta_tau.shape = (n_sim, dim_y)

        """
    """ Compute the delta tau between the real and simulated trajectories. """
    """ Summary:
    """
    # SimTrajectories.shape = (n_sim, 1, dim_y, 100, 3)
    if len(traj_sim_e_dr) == 1:
        SimTrajectories = np.array(traj_sim_e_dr[0])
    else:
        SimTrajectories = np.array(traj_sim_e_dr)
    n_sim = SimTrajectories.shape[0]

    Baseline = np.array(traj_real[0])
    Baseline = Baseline.reshape(np.concatenate(((1, ), Baseline.shape)))
    delta_tau_raw = SimTrajectories - Baseline
    
    if args.exp_name == "pusher" or args.exp_name == "pose":
        delta_tau_norm = np.linalg.norm(delta_tau_raw, axis=-1)
    else: 
        delta_tau_norm = (np.linalg.norm(delta_tau_raw[:,:,6:-3], axis=-1))
        
    if args.exp_name == "drop" and args.use_weighted_sum:
        delta_tau = np.zeros((n_sim, delta_tau_norm.shape[1]))
        for sim in range(n_sim):
                for i, number in enumerate(delta_tau_norm[sim][0]):
                    delta_tau[sim][0] += ( number ** 2 * (args.gamma ** i) ) 
    else: 
        delta_tau = np.sum(delta_tau_norm, axis=-1)

    if len(traj_sim_e_dr) == 1:
        if args.exp_name == "pusher":
            figure, axs = plt.subplots(1, 1, figsize=(10, 5))
            axs.plot(Baseline[0, 0, :, 0], Baseline[0, 0, :, 1], '-o', c='r', label=f"Real plate1")
            axs.plot(Baseline[0, 1, :, 0], Baseline[0, 1, :, 1], '-^', c='r', label=f"Real plate2")
            
            axs.plot(SimTrajectories[0, 0, :, 0], SimTrajectories[0, 0, :, 1], '-o', c='b', label=f"Sim plate1", alpha=0.5)
            axs.plot(SimTrajectories[0, 1, :, 0], SimTrajectories[0, 1, :, 1], '-^', c='b', label=f"Sim plate2", alpha=0.5)
            plot_boundary_pusher(axs)
        elif args.exp_name == "pose":
            
            figure, axs = plt.subplots(1, 1, figsize=(10, 5))
            axs.plot(Baseline[0, 0, :, 0], Baseline[0, 0, :, 1], '-o', c='r', label=f"Real Pos")
            axs.plot(Baseline[0, 1, :, 0], Baseline[0, 1, :, 1], '-^', c='r', label=f"Real Quat")
            
            axs.plot(SimTrajectories[0, 0, :, 0], SimTrajectories[0, 0, :, 1], '-o', c='b', label=f"Sim Pos", alpha=0.5)
            axs.plot(SimTrajectories[0, 1, :, 0], SimTrajectories[0, 1, :, 1], '-^', c='b', label=f"Sim Quat", alpha=0.5)
            
        else: 
            figure, axs = plt.subplots(1, 1, figsize=(10, 5))
            axs.plot(Baseline[0, 0, :, 0], Baseline[0, 0, :, 1], '-o', c='r', label=f"Real Ball Pos")
            axs.plot(SimTrajectories[0, 0, :, 0], SimTrajectories[0, 0, :, 1], '-o', c='b', label=f"Sim Ball Pos", alpha=0.5)
            plot_boundary_drop(axs)
        
        axs.axis('equal')
        axs.legend()
        plt.close()
    else:
        figure, axs = plt.subplots(1, 1, figsize=(5, 5))
        if args.exp_name == "pusher":
            
            for i in range(min(n_sim, 50)):
                axs.plot(SimTrajectories[i, 0, :, 0], SimTrajectories[i, 0, :, 1], '-o', c='b', alpha=0.05)
                axs.plot(SimTrajectories[i, 1, :, 0], SimTrajectories[i, 1, :, 1], '-^', c='g', alpha=0.05)
            axs.axis('equal')
            plot_boundary_pusher(axs)
        elif args.exp_name == "pose":
            
            for i in range(min(n_sim, 50)):
                axs.plot(SimTrajectories[i, 0, :, 0], SimTrajectories[i, 0, :, 1], '-o', c='b', alpha=0.05)
                axs.plot(SimTrajectories[i, 1, :, 0], SimTrajectories[i, 1, :, 1], '-^', c='g', alpha=0.05)
            axs.axis('equal')
            
        else: 
            for i in range(min(n_sim, 50)):
                axs.plot(SimTrajectories[i, 0, :, 0], SimTrajectories[i, 0, :, 2], '-o', c='b', alpha=0.05)
            plot_boundary_drop(axs)
        writer.add_figure("Trajectory/DR", figure, iter)
        plt.close()
    
    return delta_tau



""" 
=== Private Methods for agent training ===
"""

def init_e_dr_sim_pusher(full_context):
    full_context["dynamic@knob_g0@damping_ratio"] = -10
    full_context["dynamic@plate1_g0@damping_ratio"] = -10
    full_context["dynamic@plate2_g0@damping_ratio"] = -10 # -6
    full_context["dynamic@x_right_wall_g0@damping_ratio"] = -10
    full_context["dynamic@x_left_wall_g0@damping_ratio"] = -10
    full_context["dynamic@y_front_wall_g0@damping_ratio"] = -10
    full_context["dynamic@y_back_wall_g0@damping_ratio"] = -10
    full_context["dynamic@block_g0@damping_ratio"] = -10
    
    full_context["dynamic@plate1_g0@friction_sliding"] = 0.05 # 0.03
    full_context["dynamic@plate2_g0@friction_sliding"] = 0.05 # 0.03
    
    full_context["init@knob@velocity_delay"] = 0.75
    
    return full_context

def init_e_dr_sim_drop(full_context):
    full_context['dynamic@ball_g0@damping_ratio'] = -5  # -4.5
    full_context['dynamic@step_stone_g0@damping_ratio'] = -10  # -25
    full_context['dynamic@step_stone_2_g0@damping_ratio'] = -10 # -20
    ## TO ensure the stability of simulator, we need to modify table solref manually
    full_context['dynamic@table_collision@time_constant'] = -10000
    full_context['dynamic@table_collision@damping_ratio'] = -10
    
    return full_context

def init_e_dr_sim_pose(full_context):
    full_context['dynamic@cube_main@mass'] = 0.08 
    full_context['dynamic@cube_g0@damping_ratio'] = -5
     
    return full_context
    

def init_e_dr_real_pusher(e_dr_real):
    e_dr_real["dynamic@plate1_g0@friction_sliding"] -= 0.01
    e_dr_real["dynamic@plate2_g0@friction_sliding"] -= 0.02
    
    e_dr_real["init@knob@velocity_delay"] += 0.1

    e_dr_real["dynamic@knob_g0@damping_ratio"] = -6
    e_dr_real["dynamic@plate1_g0@damping_ratio"] = -6
    e_dr_real["dynamic@plate2_g0@damping_ratio"] = -6 # -6
    e_dr_real["dynamic@x_right_wall_g0@damping_ratio"] = -6
    e_dr_real["dynamic@x_left_wall_g0@damping_ratio"] = -6
    e_dr_real["dynamic@y_front_wall_g0@damping_ratio"] = -6
    e_dr_real["dynamic@y_back_wall_g0@damping_ratio"] = -6
    e_dr_real["dynamic@block_g0@damping_ratio"] = -6
    
    e_dr_real["init@camera@shift_x"] += 0.03
    e_dr_real["init@camera@shift_y"] -= 0.02
    
    return e_dr_real

def init_e_dr_real_drop(e_dr_real):
    # Maintain
    e_dr_real["dynamic@step_stone_main@mass"] = 0.6
    e_dr_real["dynamic@ball_main@mass"] = 0.08

    # V2
    e_dr_real['dynamic@ball_g0@damping_ratio'] = -7     # -5
    e_dr_real['dynamic@step_stone_g0@damping_ratio'] = -20  # -10
    e_dr_real["dynamic@step_stone_2_g0@damping_ratio"] = -15 # -10
    ## TO ensure the stability of simulator, we need to modify table solref manually
    e_dr_real['dynamic@table_collision@time_constant'] = -10000
    e_dr_real['dynamic@table_collision@damping_ratio'] = -10
    
    
    # # V1
    # e_dr_real['dynamic@ball_g0@damping_ratio'] = -5
    # e_dr_real['dynamic@step_stone_g0@damping_ratio'] = -5
    # e_dr_real["dynamic@step_stone_main@mass"] = 0.6
    # e_dr_real["dynamic@ball_main@mass"] = 0.08
    # e_dr_real["dynamic@step_stone_2_g0@damping_ratio"] = -7       
    
    return e_dr_real


def init_e_dr_real_pose(e_dr_real):
    
    ## TO ensure the stability of simulator, we need to modify table solref manually
    e_dr_real['dynamic@table_collision@time_constant'] = -10000
    e_dr_real['dynamic@table_collision@damping_ratio'] = -10
    
    return e_dr_real
    

def init_env_for_agent_training(args, render, e_dr=None):
    assert e_dr is not None, "e_dr is None!"
    seed = args.seed
    n_envs = args.n_envs
    
    def make_env():
        def _make_env():
            if args.exp_name == "pusher":
                env = PusherSingleAction(render=render)
            elif args.exp_name == "drop":
                env = DropSingleAction(render=render)
            else:
                env = PoseChanger(render=render)
            env.set_context(e_dr)
            env.seed(seed)
            env.action_space.seed(seed)
            return NormalizeActionSpaceWrapper(env)
        
        if n_envs == -1:
            return _make_env()
        else:
            return CustomMonitor(_make_env())

    if n_envs == -1:
        return make_env()
    if n_envs == 1:
        return DummyVecEnv([make_env for _ in range(n_envs)])
    else:
        return SubprocVecEnv([make_env for _ in range(n_envs)])

class CustomMonitor(Monitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_successes = []

    def step(self, action):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observation, reward, done, info = super().step(action)

        if done:
            info["episode"]["s"] = info.get('success')
            info["episode"]["dist_to_goal"] = info.get("dist_to_goal")
            info["episode"]["dist_to_goal_plate1"] = info.get("dist_to_goal_plate1") 
            info["episode"]["action"] = action

        return observation, reward, done, info

class NormalizeActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Store both the high and low arrays in their original forms
        self.action_space_low = self.action_space.low
        self.action_space_high = self.action_space.high

        # We normalize action space to a range [-1, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.action_space.shape, dtype=np.float32)

    def action(self, action):
        # convert action from [-1,1] to original range
        action = self.denormalize_action(action)
        return action

    def reverse_action(self, action):
        # convert action from original range to [-1,1]
        action = self.normalize_action(action)
        return action

    def normalize_action(self, action):
        action = 2 * ((action - self.action_space_low) / (self.action_space_high - self.action_space_low)) - 1
        return action

    def denormalize_action(self, action):
        action = (action + 1) / 2 * (self.action_space_high - self.action_space_low) + self.action_space_low
        return action




""" 
==== Logger / Ploting Methods ====
"""
   
def plot_traj(k, args, writer, traj_real_all, traj_sim_all, task = "pusher"):
    num_traj = len(traj_real_all)
    figure, axs = plt.subplots(num_traj, 1, figsize=(10, num_traj*5))
    if task == "pusher":
        for i in range(num_traj):
            # print(traj_real_all[i], traj_sim_all[i])
            Baseline = np.array(traj_real_all[i])
            SimTrajectories = np.array(traj_sim_all[i])

            axs[i].plot(Baseline[0, 0, :, 0], Baseline[0, 0, :, 1], '-o', c='r', label="Real plate1")
            axs[i].plot(Baseline[0, 1, :, 0], Baseline[0, 1, :, 1], '-^', c='r', label="Real plate2")

            axs[i].plot(SimTrajectories[0, 0, :, 0], SimTrajectories[0, 0, :, 1], '-o', c='b', label="Sim plate1", alpha=0.5)
            axs[i].plot(SimTrajectories[0, 1, :, 0], SimTrajectories[0, 1, :, 1], '-^', c='b', label="Sim plate2", alpha=0.5)
            plot_boundary_pusher(axs[i])

            axs[i].axis('equal')
            axs[i].legend()
    else: # For Drop Env
        for i in range(num_traj):
            Baseline = np.array(traj_real_all[i])
            SimTrajectories = np.array(traj_sim_all[i])
            axs[i].plot(Baseline[0, 0, :, 0], Baseline[0, 0, :, 2], '-o', c='r', label="Real ball Traj")
            axs[i].plot(SimTrajectories[0, 0, :, 0], SimTrajectories[0, 0, :, 2], '-o', c='b', label="Sim ball Traj", alpha=0.5)
            plot_boundary_drop(axs[i])
            axs[i].axis('equal')
            axs[i].legend()
    
    writer.add_figure("Trajectory/Sim-to-Real-all", figure, k)
    plt.close()

def plot_boundary_pusher(axs):
    axs.plot([0.2, 0.2], [0.09, -0.09], '-', c='k', alpha=1)
    axs.plot([0.45, 0.45], [0.24, -0.24], '-', c='k',alpha=1)
    axs.plot([-0.45, 0.45], [0.24, 0.24], '-', c='k',alpha=1)
    axs.plot([-0.45, 0.45], [-0.24, -0.24], '-', c='k',alpha=1)

def plot_boundary_drop(axs):
    axs.plot([-0.035, 0.095], [0.96, 0.96], '^', c='g', alpha=1) # Goal line
    axs.plot([-0.3485, -0.2515], [0.857, 0.833], '-', c='k',alpha=1) # Plate 1
    axs.plot([0.21881, 0.48119], [0.95728, 1.10272], '-', c='k',alpha=1) # Plate 2 


def save_data(iter, args, data, name):
    with open(os.path.join(args.logdir_causual_DR, 'data', 'iter_{}_{}.pkl'.format(iter, name)), 'wb') as f:
        pickle.dump(data, f)

def log_param(iter, writer, env_params, interested_context):
    for c in interested_context:
        writer.add_scalar('env_params/' + c, env_params[c], iter)
        

""" 
==== Supporting Class ====
"""

class CustomSubprocVecEnv(SubprocVecEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
