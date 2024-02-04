import argparse
import json
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO, SAC

from stable_baselines3.common.utils import set_random_seed
import pickle
import time

import numpy as np
import copy


from utils.common_Helper import *

from utils.causual_DR_additional_Helper import causality_guided_DR, init_causal_model, optimize_env_params, train_causal_model, _save_tensor_with_timestamp

from utils.train_agent_additional_Helper import init_training_callback, train_agent_and_get_cmd_act, train_agent_only




def main(args):
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device(args.device)
    
    """ === [Init] prepare the essential objects for the main pipeline === """
    #### create tensorboard causual_DR_writer, train_agent_writer
    causual_DR_writer = SummaryWriter(args.logdir_causual_DR)
    train_agent_writer = init_training_callback(args)
    
    #### Init the single_env to get context (e_dr)
    # Will init multi_env once finish the training of agent to lower the memory usage
    single_env= init_env(args, n_envs=1)
    # Reset the environment
    single_env.reset()
    
    #### Figure out the interested context
    # A Dictiondary of Full Context (400+) and their values
    e_dr_sim = init_e_dr_sim(args, single_env)
    # A List of the Interested Context Keys Only (~64)
    interested_context = init_interested_context(args)
    copy_of_interested_context = copy.deepcopy(interested_context)
    # Log the current value of the interested context in the simulation environment
    log_param(0, causual_DR_writer, e_dr_sim, copy_of_interested_context)
    
    # Create a fake ground truth of the real environment
    e_dr_real = init_e_dr_real(args, e_dr_sim)
    
    ############# 
    # Parameters Checkpoint pass 
    print('Environment Parameters: ')
    for c in interested_context:
        print("real:",c, e_dr_real[c])
        print("sim: ",c, e_dr_sim[c])
    #############
    
    # set_random_seed(args.seed)
    set_random_seed(args.seed, using_cuda=True)
    
    #### Load from real experiments OR not
    if args.load_real_dataset:
        #### Load from real experiments
        real_traj_act_load = dict(np.load(args.load_real_dataset))
        args.real_rollout = real_traj_act_load['traj_real'].shape[0]
        print('Loading real traj and action from: ', args.load_real_dataset)
    else:
        real_traj_act_load = None
    
    """ === [multi_env] Init the multi_env for high efficient rollouts === """
    multi_env = init_env(args, n_envs=args.n_envs)
    multi_env.reset()
    
    for k in range(args.iter):
        print('------------------------------')
        print('iter ', k, '| logdir ', args.logdir)
        save_data(k, args, {'e_dr_sim': e_dr_sim}, 'e_dr_sim')
        
        """ === [Training Agent and Generate command actions] === """
        if k % args.agent_retrain_freq == 0:
            # Will use either Dummy agent or SAC agent to pick command actions.
            # For SAC agent, use predict() to generate action from its policy
            # For Dummy agent [Skip Training], use fake predict() to random sample from entire action sapce
            command_act_list = train_agent_and_get_cmd_act(args, single_env, e_dr_sim, train_agent_writer, save_postfix=f"{k}-thOptimizedContext")
            
        
        """ === [Logging] push those trajectories (image) to the tensorboard logdir ===
        In Sim2Sim:
            Generate trajectories using current e_dr_sim and e_dr_real under same command action
        In Sim2Real:
            Generated trajectories using current e_dr_sim and dataset from real experiments
        """
        traj_real_all, act_real_traj = log_performance_on_current_e_dr(args, k, single_env, causual_DR_writer, e_dr_sim, e_dr_real, command_act_list, real_traj_act_load)
        
        
        """ === [COMPASS-1] Causality Guided DR === """
        delta_tau_e_dr_all = []
        act_dr_all = []
        act_real_all = []

        traj_sim_e_dr_all = []
        context_purturb_matrix_dict_all = []
        for i, traj_real in enumerate(traj_real_all):
            e_dr_list, context_purturb_matrix_dict = causality_guided_DR(k, e_dr_sim, interested_context, args)
            
            traj_sim_e_dr, act_dr = rollout(multi_env, args, e_dr_list, return_act=True, command_act=act_real_traj[i][0][0])
            delta_tau_e_dr = compute_delta_tau(k, args, causual_DR_writer, traj_real, traj_sim_e_dr)
            traj_sim_e_dr_all.append(traj_sim_e_dr)
            delta_tau_e_dr_all.append(delta_tau_e_dr)
            act_dr_all.append(act_dr)
            context_purturb_matrix_dict_all.append(copy.deepcopy(context_purturb_matrix_dict))
            act_real_all.append(np.concatenate([act_real_traj[i][0][0] for _ in range(args.n_context)], axis=0))
        
        
        action_space_len = len(act_real_traj[0][0][0])
        delta_tau_e_dr_all = np.concatenate(delta_tau_e_dr_all, axis=0) 
        act_dr_all = np.concatenate(act_dr_all, axis=0).reshape(-1, action_space_len) 
        act_real_all = np.concatenate(act_real_all, axis=0).reshape(-1, action_space_len)

        """ === [COMPASS-2] Causality Model Training === """
        data_all = []
        for matrix_dict in context_purturb_matrix_dict_all: 
            data = []
            for key in matrix_dict.keys():
                data.append(copy.deepcopy(matrix_dict[key].reshape((args.n_context, 1))))
            data = np.hstack(data)
            data_all.append(data)
        data_all = np.concatenate(data_all, axis=0)
        index = range(delta_tau_e_dr_all.shape[0])
        causal_model = init_causal_model(k, args, len(interested_context), delta_tau_e_dr_all.shape[1], action_space_len)
        
        
        # Train causal model; Get interested context from causal graph
        causal_model = train_causal_model(device, k, args, causual_DR_writer, causal_model, context_purturb_matrix_dict, delta_tau_e_dr_all[index], data_all[index], act_dr_all[index], act_real_all[index])
        
        # _save_tensor_with_timestamp(causal_model, name="causal_model")
        
        """ === [COMPASS-3] Env Parameters Optimization === """
        # Using gradient descent to update the value of e_dr_sim
        e_dr_sim, interested_context = optimize_env_params(device, k, causual_DR_writer, args, causal_model, e_dr_sim, context_purturb_matrix_dict, act_real_traj)
        
        log_param(k+1, causual_DR_writer, e_dr_sim, copy_of_interested_context)
        if k==0:
            args.sparsity_weight = args.sparsity_weight * args.sw_discount
            
        # End of the k-th iteration
    
        """ === [COMPASS-0] DEBUG RANDOM SEED === """
        # args.debugRandomSeed = True
        if args.debugRandomSeed:
            print('Early Quit-2')
            quit()


            
            
    """ === [Train Agent-2] Based on the optimized e_dr_sim (best guess of e_dr_real) === """
    if args.use_sac_agent or args.use_ppo_agent:
        print('training agent based on the optimized e_dr_sim (best guess of e_dr_real)')
        # Re-train agent based on the postOptimizedContext (e_dr_sim)
        agent = train_agent_only(args, single_env, e_dr_sim, train_agent_writer, save_postfix="postOptimizedContext")    
            
    
    print(" ======= COMPASS pipeline is finished ======= ")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description ='Causality Guided Domain Randomization + SAC/Dummy Agent')
    
    parser.add_argument('--debugRandomSeed', default=False, action='store_true', help="whether debugRandomSeed -- Early Quit")
    parser.add_argument('--remove_outlier_percent', type=float, default=0.05, help="remove_outlier_percent")
    
    
    #### Log directory
    parser.add_argument('--logdir', type=str, default='./logdir/', help="Log directory")
    parser.add_argument('--exp_name', type=str, default="pusher", help="Optional experiment name")
    parser.add_argument('--logdir_causual_DR', type=str, default=None, help="Log directory of causality guided DR, based on the logdir+exp_name")
    parser.add_argument('--logdir_train_agent', type=str, default=None, help="Log directory of SAC/Dummy agent training, based on the logdir+exp_name")
    
    #### agent training parameters
    parser.add_argument('--use_sac_agent', default=False, action='store_true', help="whether use SAC or Dummy agent for rollouts")
    parser.add_argument('--use_ppo_agent', default=False, action='store_true', help="whether use PPO agent for rollouts")
    parser.add_argument('--load_preTrained', type=str, default=None, help="Load pre-trained agent")
    parser.add_argument('--agent_training_steps', type=int, default=10000, help="training steps")   
    parser.add_argument('--agent_retrain_freq', type=int, default=5, help="the SAC retrain frequency during context iteration steps")   
    
    
    #### domain randomization parameters    
    parser.add_argument('--n_envs', type=int, default=64, help="Number of parallel environments to use for training.")
    parser.add_argument('--n_context', type=int, default=64, help="Number of purturbation on context to use for training.")
    
    parser.add_argument('--real_rollout', type=int, default=10, help="Number of real rollouts.")
    parser.add_argument('--dr_anneal', type=float, default=1.0, help="Domain randomization annealing factor.")

    parser.add_argument('--use_weighted_sum', action='store_true', help="Whether to use weighted sum of delta_tau.")
    parser.add_argument('--gamma', type=float, default=1.03, help="Time dependent factor for calculating weighted sum of delta_tau.")
    
    #### causal model parameters
    parser.add_argument('--epoch', type=int, default=4000, help="Number of epochs to train the causal model.")
    parser.add_argument('--sparsity_weight', type=float, default=0.01, help="Sparsity weight.")
    parser.add_argument('--sparse_norm', type=float, default=1, help="Sparsity norm.")
    parser.add_argument('--sw_discount', type=float, default=0.35, help="Sparsity weight discount factor.")
    parser.add_argument('--use_full', action='store_true', help="Whether to use the full graph.")
    parser.add_argument('--mlp', action='store_true', help="Whether to use MLP.")
    parser.add_argument('--param_opt_lr', type=float, default=0.001, help="Learning rate of the gradient descent to update the environment parameters.")
    
    parser.add_argument('--optimize_steps', type=int, default=2000, help="how many steps compass optimize the environment parameters.")
    
    #### main algorithm parameters
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--iter', type=int, default=10, help="Maximum number of total iterations")
    
    parser.add_argument('--load_real_dataset', type=str, default=None, help="Load real trajectory")
    
    ## For Pusher different initial parameters 
    parser.add_argument('--pusher_init', type=int, default=1, help="which pusher initial parameters to use")

    
    # parse arguments
    args = parser.parse_args()
    

    #### update logdir with time stamp
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if args.load_real_dataset is None:
        folder_name = 'Sim' + args.exp_name + timestr + '_nc_{}_sw_{}_swd_{}_ep_{}_f_{}_mlp_{}_rr_{}_use_sac_agent_{}_gamma_{}_'.format(args.n_context, args.sparsity_weight, args.sw_discount, args.epoch, args.use_full, args.mlp, args.real_rollout, args.use_ppo_agent, args.gamma)
    else:
        args.logdir = './logdir_real/'
        folder_name = 'Real' + args.exp_name + timestr + '_nc_{}_sw_{}_swd_{}_ep_{}_f_{}_mlp_{}_rr_{}_use_sac_agent_{}_gamma_{}_'.format(args.n_context, args.sparsity_weight, args.sw_discount, args.epoch, args.use_full, args.mlp, args.real_rollout, args.use_ppo_agent, args.gamma)
        
    args.logdir = os.path.join(args.logdir, folder_name)
    
    args.logdir_causual_DR = os.path.join(args.logdir, "causal_DR")
    args.logdir_train_agent = os.path.join(args.logdir, "train_agent")
    
    #### create logdir if not exist
    causual_DR_folder_name = args.logdir_causual_DR
    if not os.path.exists(causual_DR_folder_name):
        os.makedirs(causual_DR_folder_name)
        os.makedirs(os.path.join(causual_DR_folder_name, 'data'))
    
    train_agent_folder_name = args.logdir_train_agent
    if not os.path.exists(causual_DR_folder_name):
        os.makedirs(causual_DR_folder_name)
    
    #### save args to logdir as json file
    with open(os.path.join(args.logdir, 'config.json'), 'a+') as f:
        print('Log directory: ', args.logdir)
        json.dump(vars(args), f, indent=4)
    
    
    #### execute main algorithm
    main(args)
    
    
    
    