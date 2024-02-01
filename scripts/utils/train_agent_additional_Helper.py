
from stable_baselines3 import PPO, SAC


from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, set_random_seed
from stable_baselines3.common.monitor import Monitor
import gym
import numpy as np
from tqdm import tqdm
import os

from scripts.utils.common_Helper import *



from scripts.utils.common_Helper import init_env_for_agent_training



""" 
==== Public Methods ====
"""

def init_training_callback(args):
    
    if args.use_sac_agent:
        algorithm = 'sac'
    elif args.use_ppo_agent:
        algorithm = 'ppo'
    else:
        algorithm = 'dummy'
    
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
                            save_freq= int(args.agent_training_steps/10),
                            save_path=args.logdir_train_agent,
                            name_prefix=algorithm,
                            save_replay_buffer=True,
                            save_vecnormalize=True,
                            )
    if args.exp_name == "pusher":
        tensorboard_callback = TensorboardCallback_Pusher()
    else:
        tensorboard_callback = TensorboardCallback_Drop()
    train_callback = CallbackList([checkpoint_callback, tensorboard_callback])
        
    return train_callback



def train_agent_and_get_cmd_act(args, single_env, e_dr_sim, train_agent_writer, save_postfix=None):
    assert save_postfix is not None, "save_postfix can not be none!"
    # Init either a dummy agent or a sac agent (ready for training)
    agent = _init_agent(args, single_env, e_dr_sim)
    
    """ === [Train Agent-1] Based on the initial guess (default) e_dr_sim === """
    if args.use_sac_agent:
        print('Using SAC Agent')
        if (args.load_preTrained is not None):
            print('Loading Pretrained SAC Agent')
            agent = SAC.load(args.load_preTrained)
        else:
            print('Training SAC Agent from scratch')
            agent = _train_agent(args, agent, train_agent_writer, save_postfix=save_postfix)
    elif args.use_ppo_agent:
        print('Using PPO Agent')
        if (args.load_preTrained is not None):
            print('Loading Pretrained PPO Agent')
            agent = PPO.load(args.load_preTrained)
        else:
            print('Training PPO Agent from scratch')
            agent = _train_agent(args, agent, train_agent_writer, save_postfix=save_postfix)
    else:
        print('Using Dummy Agent; Skip training')
    
    command_act_list = []
    obs = single_env.reset()
    
    for _ in range(args.real_rollout):
        # For Dummy Agent, its predict() just randomly samples action from entire action_space
        action, _ = agent.predict(obs, deterministic=False)  
        if args.use_sac_agent or args.use_ppo_agent:      
            action = agent.env.env_method("action", np.array(action).squeeze(), indices=0)
            action = np.array(action).squeeze()
        command_act_list.append(action)
    if args.use_sac_agent or args.use_ppo_agent:      
        # print ("------Closing training env to avoid memory leak------")
        agent.env.close()
    
    return command_act_list
    
        

def train_agent_only(args, single_env, e_dr_sim, train_agent_writer, save_postfix="postOptimizedContext"):
    agent = _init_agent(args, single_env, e_dr_sim)
    agent = _train_agent(args, agent, train_agent_writer, save_postfix=save_postfix)    
    return agent




""" 
==== Private Methods ====
"""


def _init_agent(args, single_env, e_dr_sim):
    multi_env = init_env_for_agent_training(args, render=False, e_dr=e_dr_sim)
    if args.use_sac_agent:
        agent = SAC("MlpPolicy",    # policy type
                multi_env,            # environment
                verbose=1,      # print progressbar
                learning_starts=100,
                gradient_steps=32,
                batch_size=32,
                train_freq=8,
                ent_coef=0.005,
                policy_kwargs=dict(net_arch=[32, 32]),
                tensorboard_log=args.logdir_train_agent
                )
    elif args.use_ppo_agent:
        agent = PPO(
            "MlpPolicy",
            multi_env,
            target_kl=0.03,
            verbose=1,
            batch_size = 64, # 64
            n_steps = 4, # 512
            n_epochs = 5,
            learning_rate = 3e-4,
            tensorboard_log=args.logdir_train_agent,
            policy_kwargs=dict(
                net_arch=[32, 32],
                log_std_init=-1.5,
                ),

        )
    else:
        agent = DummyAgent(args, single_env)
    return agent


def _train_agent(args, agent, train_agent_writer, save_postfix):
    print ("----------------Algorithm training started------------------")
    
    # train the agent
    agent.learn(total_timesteps=args.agent_training_steps, callback=train_agent_writer)
    if args.use_sac_agent:
        agent.save(os.path.join(args.logdir_train_agent, f"sac_{args.exp_name}_{save_postfix}"))
    else: 
        agent.save(os.path.join(args.logdir_train_agent, f"ppo_{args.exp_name}_{save_postfix}"))
    print ("----------------Algorithm training finished------------------")
    return agent


""" 
==== Customized Class ====
"""

class DummyAgent:
    def __init__(self, args, single_env) -> None:
       
        self.action_sapce = single_env.action_space
        self.action_sapce.seed(args.seed)

    def predict(self, obs, deterministic=False):
        return self.action_sapce.sample(), None


class TensorboardCallback_Pusher(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # success_rate = self.training_env.get_success_rate(window_size=100)
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            success_rate = safe_mean([ep_info["s"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/ep_success_rate", success_rate)
            dist_to_goal = safe_mean([ep_info["dist_to_goal"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/ep_mean_distance_to_goal", dist_to_goal)
            dist_to_goal = safe_mean([ep_info["dist_to_goal_plate1"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/ep_mean_distance_to_goal_plate1", dist_to_goal)
            action_1 = safe_mean([ep_info["action"][0] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/ep_mean_x_pos", action_1)
            action_2 = safe_mean([ep_info["action"][1] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/ep_mean_y_pos", action_2)
            action_3 = safe_mean([ep_info["action"][2] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/ep_mean_theta", action_3)
            action_4 = safe_mean([ep_info["action"][3] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/ep_mean_velocity", action_4)
            return True
        
class TensorboardCallback_Drop(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # success_rate = self.training_env.get_success_rate(window_size=100)
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            success_rate = safe_mean([ep_info["s"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/ep_success_rate", success_rate)
            action = safe_mean([ep_info["action"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/drop_height(absolute)", action)
            return True