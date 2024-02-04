

import time
import os
import time

import gym
import numpy as np
import copy
from tqdm import tqdm
import robosuite as suite
from robosuite.wrappers import GymWrapper
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


from utils.common_Helper import *

from utils.common_Helper import save_data, CLIP_RANGE
from utils.causal_sim2real import CausalSim2Real


""" 
==== Public Methods -- Causuality Guided DR ====
"""

def causality_guided_DR(iter, env_params, interested_context, args):
    
    n_context = args.n_context
    
    env_param = env_params.copy()
    
    e_dr_list = [copy.deepcopy(env_param) for _ in range(n_context)]
    
    # Initialize the context perturbation matrix
    context_purturb_matrix_dict = OrderedDict()
    
    for key in interested_context:
        context_purturb_matrix_dict[key] = None
    
    """ === [new] High Efficiency === """
    for key in interested_context:
        val = env_param[key]
        if "lighting@light@ambient" in key or "lighting@light@pos" in key:
            context_purturb_matrix_dict[key] = val + (args.dr_anneal**iter)*np.random.uniform(-1.5, 1.5, n_context)
        elif "init@knob@velocity_delay" in key:
            context_purturb_matrix_dict[key] = val + (args.dr_anneal**iter)*np.random.uniform(-0.1, 0.1, n_context)
        elif "dynamic@knob_g0@damping_ratio" in key or "dynamic@plate1_g0@damping_ratio" in key or "dynamic@plate2_g0@damping_ratio" in key or "dynamic@ball_g0@damping_ratio" in key or "dynamic@step_stone_g0@damping_ratio" in key:
            context_purturb_matrix_dict[key] = val + (args.dr_anneal**iter)*np.random.uniform(-10., 10., n_context)
        elif "dynamic@x_right_wall_g0@damping_ratio" in key or "dynamic@x_left_wall_g0@damping_ratio" in key or "dynamic@y_front_wall_g0@damping_ratio" in key or "dynamic@y_back_wall_g0@damping_ratio" in key or "dynamic@block_g0@damping_ratio" in key or "dynamic@step_stone_2_g0@damping_ratio" in key:
            context_purturb_matrix_dict[key] = val + (args.dr_anneal**iter)*np.random.uniform(-30., 30., n_context)
        elif "dynamic@cube_g0@friction_sliding" in key or "dynamic@gripper0_hand_collision@friction_sliding" in key:  
            context_purturb_matrix_dict[key] = val + (args.dr_anneal**iter)*np.random.uniform(-0.4, 0.4, n_context)
            # print ("key value ", context_purturb_matrix_dict[key])
        elif "friction_sliding" in key or "dynamic@cube_g0@friction_torsional" in key or "dynamic@cube_g0@friction_rolling" in key:            
            context_purturb_matrix_dict[key] = val + (args.dr_anneal**iter)*np.random.uniform(-0.02, 0.02, n_context)
        # === [May 16] Camera Shift ===
        elif "init@camera@shift_x" in key or "init@camera@shift_y" in key:
            context_purturb_matrix_dict[key] = val + np.random.uniform(-0.03, 0.03, n_context)
        elif "dynamic@gripper0_hand_collision@friction_torsional" in key:
            context_purturb_matrix_dict[key] = val + np.random.uniform(-0.005, 0.005, n_context)
        elif "dynamic@cube_g0_damping_ratio" in key:
            context_purturb_matrix_dict[key] = val + (args.dr_anneal**iter)*np.random.uniform(-10., 10., n_context)
        elif "dynamic@gripper0_left_inner_knuckle_collision@damping_ratio" in key:
            context_purturb_matrix_dict[key] = val + (args.dr_anneal**iter)*np.random.uniform(-10., 10., n_context)
        else:
            context_purturb_matrix_dict[key] = np.random.uniform(0.5 * val, 1.5 * val, n_context)
        
        # clip
        if key in CLIP_RANGE.keys():
            context_purturb_matrix_dict[key] = np.clip(context_purturb_matrix_dict[key], CLIP_RANGE[key][0], CLIP_RANGE[key][1])
        
    """ Convert the context_purturb_matrix_dict to a list of dictionary """
    for i in range(n_context):
        for key, val in env_param.items():
            if key in interested_context:
                e_dr_list[i][key]= context_purturb_matrix_dict[key][i].copy()
    
    return e_dr_list, context_purturb_matrix_dict


""" 
==== Public Methods -- Causuality Model ====
"""


def init_causal_model(iter, args, in_dim, out_dim, action_space_len):
    # set seed one more time
    set_random_seed(args.seed, using_cuda=True)
    
    """ Initialize the causal model. """
    print("sparsity_weight: ", args.sparsity_weight)
    print('input_dim', in_dim + action_space_len)
    if args.mlp:
        model = MLP(in_dim+action_space_len, out_dim, hidden_dim=256, num_hidden=2, out_activation=None, dropout=0.0)
    else:
        ## input_dim=in_dim+action_space_len
        # action_space_len = 4
        # out_dim = 2
        model = CausalSim2Real(input_dim=in_dim+action_space_len, output_dim=out_dim, 
                            emb_dim=32, hidden_dim=256, causal_dim=32, 
                            num_hidden=2, sparse_weight=args.sparsity_weight, 
                            sparse_norm=args.sparse_norm, use_full=args.use_full, action_size = action_space_len
                            )
    return model

def train_causal_model(device, iter, args, writer, causal_model, context_purturb_matrix_dict, delta_tau_e_dr, e_dr_list, act_dr, act_real):
    """_summary_

    Args:
        args (_type_): _description_
        causal_model (_type_): _description_
        context_purturb_matrix_dict ( {[]} ): _description_
        delta_tau_e_dr ( [{}] ): _description_
    """
    
    causal_model.normalizer = Normalizer()
    n_context = args.n_context
    
    """ Pre-process the data (input of network) """
    data = e_dr_list.copy()
    
    # print(data.shape, act_dr.shape, act_real.shape)
    data = np.hstack((data, act_real))
    # print('data.shape', data.shape)
    
    data = causal_model.normalizer.normalize(data)
    # Flaten the high(mixed) dimensional context to a 1D vector
    context_keys_flat = []
    for key in context_purturb_matrix_dict.keys():
        length = context_purturb_matrix_dict[key].size // n_context
        context_keys_flat += [f"{key}" for i in range(length)]
    
    
    """ Pre-process the label (output of network) """
    # [Note] Normalize the difference between real and simulated trajectories in compute_delta_tau()
    label = delta_tau_e_dr 
    print("prev_label.shape", label.shape)
    print("prev_data.shape", data.shape)
    
    # Remove the outlier of label (delta_tau_e_dr) in an ascending order (small --> great)
    sorted_indices = np.argsort(label, axis=0).flatten()
    # find the outlier index of top (last) ~5% percentile from the sorted_indices
    outlier_index = sorted_indices[int(-1 * len(sorted_indices)*args.remove_outlier_percent):]    
    # remove the item from both data and label based on the outlier_index
    label = np.delete(label, outlier_index, axis=0)
    data = np.delete(data, outlier_index, axis=0)
    print("[Without Outlier] label.shape", label.shape)
    print("[Without Outlier] data.shape", data.shape)
    
    # normaliza the label to 0~1 range in each dimension
    label = (label - np.min(label)) / (np.max(label) - np.min(label))
    
    """ To torch tensor """
    data = torch.from_numpy(data[:, :]).float()
    label = torch.from_numpy(label[:]).float()
    
    figure, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(data[:, 0], label[:, 0], 'o', alpha=0.1)
    axs[0].title.set_text('data[:, 0], label[:, 0]')
    if args.exp_name == "pusher" or args.exp_name == "pose":
        axs[1].plot(data[:, 1], label[:, 1], 'o', alpha=0.1)
        axs[1].title.set_text('data[:, 1], label[:, 1]')
    writer.add_figure('Causal model/Data_vis', figure, iter)
    plt.close()
    
    
    """ To dataset and dataloader """
    dataset = TensorDataset(data, label)
    
    # split data into train and test
    n = data.shape[0]
    train_size = int(0.8 * n)
    test_size = n - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False) # 128
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    """ Train the model """
    causal_model.to(device)
    
    
    # Adam is the key to train the model!!!
    optimizer = torch.optim.Adam(causal_model.parameters(), lr=1e-3)
    
    epochs = args.epoch # 100
    
    train_loss_list = []
    test_loss_list = []

    train_mse_loss_list = []
    test_mse_loss_list = []

    train_sparsity_loss_list = []
    test_sparsity_loss_list = []

    for _ in tqdm(range(epochs)):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_mse_loss, train_sparsity_loss = _train_causal(causal_model, train_dataloader, optimizer, device)
        test_loss, test_mse_loss, test_sparsity_loss = _test_causal(causal_model, test_dataloader, device, print_pred=False)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_mse_loss_list.append(train_mse_loss)
        test_mse_loss_list.append(test_mse_loss)
        train_sparsity_loss_list.append(train_sparsity_loss)
        test_sparsity_loss_list.append(test_sparsity_loss)

    print("train_causal_model is Done!")
    
    # plot subplots
    figure, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(train_mse_loss_list)
    axs[0].plot(test_mse_loss_list)
    axs[0].legend(['train', 'test'])
    axs[0].title.set_text('MSE Loss')
    axs[1].plot(train_sparsity_loss_list)
    axs[1].plot(test_sparsity_loss_list)
    axs[1].legend(['train', 'test'])
    axs[1].title.set_text('Sparsity Loss')
    plt.show()
    writer.add_figure('Causal model/training', figure, iter)
    plt.close()

    yticklabels = [str(i) + ':' + context_keys_flat[i] for i in range(len(context_keys_flat))]
    mask_prob, mask = causal_model.get_mask(0.5)
    if args.exp_name == "pusher":
        plt.figure(figsize=(20, 20))
        ax = sns.heatmap(mask_prob.cpu().detach().numpy(), square=True, fmt=".2f", annot=True, yticklabels=yticklabels, xticklabels=['plate1', 'plate2'])
    elif args.exp_name == "pose":
        plt.figure(figsize=(40, 40))
        ax = sns.heatmap(mask_prob.cpu().detach().numpy(), square=True, fmt=".2f", annot=True, yticklabels=yticklabels, xticklabels=['Pos', 'Roll', 'Pitch', 'Yaw'])
    else:
        plt.figure(figsize=(20, 20))
        ax = sns.heatmap(mask_prob.cpu().detach().numpy(), square=True, fmt=".2f", annot=True, yticklabels=yticklabels, xticklabels=['Ball Pos'])
    plt.show()
    
    writer.add_figure('Causal model/mask_prob', ax.figure, iter)
    save_data(iter, args, {'mask_prob':mask_prob, 'yticklabels':yticklabels}, "mask_prob")
    
    return causal_model 


def optimize_env_params(device, iter, writer, args, causal_model, env_params, context_purturb_matrix_dict, act_real_traj):
    """_summary_
    
    Args:
        args (_type_): _description_
        causal_model (_type_): _description_
        env_params (dict, optional): _description_. Defaults to {}.
        context_purturb_matrix_dict (dict, optional): _description_. Defaults to {[]}.
    """
     
    print("Optimization is called!")
    env_params = env_params.copy()
    
    mask_prob_graph, mask_metric = causal_model.get_mask(0.5)
    if args.use_full:
        mask_prob_graph = np.ones_like(mask_prob_graph.cpu().detach().numpy())
    
    ## Define Action space length 
    if args.exp_name == 'pusher':
        action_len = 4
    elif args.exp_name == 'pose':
        action_len = 120
    elif args.exp_name == 'drop':
        action_len = 1 
    else: 
        raise NotImplementedError

    
    loss = nn.MSELoss()
    causal_model.eval()
    causal_model.requires_grad_(False)
    
    interested_context_value = []
    interested_context = []
    for key in context_purturb_matrix_dict.keys():
        interested_context.append(key)
        interested_context_value.append(env_params[key])
    
    X = np.copy(interested_context_value).reshape((1, -1))
    X = np.array(X)
    X = np.concatenate([X for _ in range(len(act_real_traj))], axis=0)
    
    act_real = np.array(act_real_traj)
    print('act_real.shape', act_real.shape)
    act_real = act_real.reshape((X.shape[0], -1))
    print('act_real.shape', act_real.shape)
    
    X = np.hstack((X, act_real.copy())).astype(np.float64)
    X = causal_model.normalizer.normalize(X)
    X_copy = np.copy(X)
    X = torch.tensor(X, requires_grad=True, dtype=torch.float).to(device)
    
    X.retain_grad()
    
    optimizer = torch.optim.Adam([X], lr=1e-3)
    
    x_hist = []
    l_hist = []

    delta_loss = 0.0
    last_loss = None

    for i in tqdm(range(args.optimize_steps)):
        pred_y = causal_model(X, threshold=0.3)
        l = torch.mean(pred_y[:, :])
        
        l.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        # X.data[:, :-action_len] = torch.FloatTensor(X_copy[:, :-action_len]).to(device)
        X.data[:, -action_len:] = torch.cuda.FloatTensor(X_copy[:, -action_len:])
        
        # set the rest dimension to the average across the samples of X.data
        mean_X = X.data[:, :-action_len].cpu().detach().numpy().mean(axis=0)
        # X.data[:, :-action_len] = torch.FloatTensor(np.repeat(mean_X.reshape((1, -1)), X.shape[0], axis=0)).to(device)
        X.data[:, :-action_len] = torch.cuda.FloatTensor(np.repeat(mean_X.reshape((1, -1)), X.shape[0], axis=0))
        
        # store x in x_hist
        last_loss = np.copy(l.item())
        l_hist.append(l.item())
        x_hist.append(X[-1].cpu().detach().numpy()) # any index should be the same env param
        
        
        norm_diff = np.linalg.norm(X_copy[-1, :-action_len] - X[-1, :-action_len].cpu().detach().numpy())

    writer.add_scalar('Causal model/norm_diff', norm_diff, iter)
    
    l_hist = np.array(l_hist)
    x_hist = np.array(x_hist)
    
    figure, axs = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(len(interested_context)):
        axs[0].plot(x_hist[:, i], label=interested_context[i])
    axs[0].legend(bbox_to_anchor=(0, 1.02), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    axs[0].title.set_text('Env params')
    axs[1].plot(l_hist[:])
    axs[1].title.set_text('Loss')
    plt.show()
    
    writer.add_figure('Causal model/optimizing', figure, iter)
    
    # End of the for loop
    new_interested_context = []
    last_item = x_hist[-1]
    last_item = causal_model.normalizer.denormalize(last_item)
    for i, key in enumerate(context_purturb_matrix_dict.keys()):
        env_params[key] = last_item[i]
        
        if key in CLIP_RANGE.keys():
            env_params[key] = np.clip(env_params[key], CLIP_RANGE[key][0], CLIP_RANGE[key][1])
        
        if mask_prob_graph[i].max() > 0.3:
            new_interested_context.append(key)

    return env_params, new_interested_context



""" 
==== Private Methods ====
"""


def _train_causal(causal_model, dataloader, optimizer, device):
    # define a training function
    size = len(dataloader.dataset)
    epoch_loss = 0.
    epoch_mse_loss = 0.
    epoch_sparsity_loss = 0.
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = causal_model(X)
    
        assert(pred.shape == y.shape)
        
        loss, info = causal_model.loss_function(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_mse_loss += info['mse']
        epoch_sparsity_loss += info['sparsity']
        

    epoch_loss /= len(dataloader)
    epoch_mse_loss /= len(dataloader)
    epoch_sparsity_loss /= len(dataloader)

    return epoch_loss, epoch_mse_loss, epoch_sparsity_loss

def _test_causal (causal_model, dataloader, device, print_pred=False):
    # define a testing function
    num_batches = len(dataloader)
    epoch_loss = 0.
    epoch_mse_loss = 0.
    epoch_sparsity_loss = 0.
    with torch.no_grad():
        for X, y in dataloader:
            # move data to GPU
            X, y = X.to(device), y.to(device)
            pred = causal_model(X, threshold=0.5)
            loss, info = causal_model.loss_function(pred, y)

            epoch_loss += loss.item()
            epoch_mse_loss += info['mse']
            epoch_sparsity_loss += info['sparsity']
    
    if print_pred:
        print(torch.round(torch.abs(pred - y)/(y), decimals=2))

    epoch_loss /= len(dataloader)
    epoch_mse_loss /= len(dataloader)
    epoch_sparsity_loss /= len(dataloader)

    return epoch_loss, epoch_mse_loss, epoch_sparsity_loss



def _save_tensor_with_timestamp(tensor, name):
    # Save the tensor to the specified file_path with a timestamp in the filename
    timestamp = int(time.time())
    file_name = "./debugRandomSeed/"+f"{name}_{timestamp}.pt"
    torch.save(tensor, file_name)


""" 
==== Customized Class ====
"""

class Normalizer(): 
    def __init__(self) -> None:
        self.min = None 
        self.main = None
    
    def normalize(self, data):
        if self.min is None:
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
            self.max = np.where(np.abs(self.max - self.min) <= 1e-8, 100, self.max)
            
        return 2* ((data - self.min) / (self.max - self.min)) - 1 
    
    def denormalize(self, data):
        return ((data + 1) * (self.max - self.min) / 2) + self.min
    

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_hidden=2, out_activation=None, dropout=0.0):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = nn.ReLU()
        self.out_activation = out_activation
        self.dropout = nn.Dropout(dropout)
        
        self.mask_prob = torch.ones((self.input_dim, self.output_dim))
        self.mask = torch.ones((self.input_dim, self.output_dim))

        self.fc_list = nn.ModuleList()
        self.fc_list.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden):
            self.fc_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, threshold=None):
        assert x.shape[-1] == self.input_dim
        for i in range(len(self.fc_list)):
            x = self.activation(self.fc_list[i](x))
            x = self.dropout(x)

        # no activation for the last layer
        if self.out_activation is not None:
            return self.out_activation(self.output_fc(x))
        return self.output_fc(x)

    def loss_function(self, pred, label):
        info = {}
        mse_loss = F.mse_loss(pred, label)
        info['mse'] = mse_loss.item()
        info['sparsity'] = 0.
        
        return mse_loss, info
    
    def get_mask(self, threshold=0.5):
        return self.mask_prob, self.mask 

    