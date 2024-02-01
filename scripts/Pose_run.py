from robosuite import load_controller_config
from robosuite.environments.manipulation.pose_changer import Pose_changer
import gym
import numpy as np
import time
import cv2
import imageio
import copy
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
import pickle as pkl


class PoseChanger(gym.Env):
    def __init__(self, render=False):
        self.render = render
        
        config = load_controller_config(
                     custom_fpath="../robosuite/controllers/config/osc_position_custom.json")
        
        self.camera_name = "sideview"
        self.env = Pose_changer(
            robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=False,
            has_offscreen_renderer=self.render,
            use_camera_obs=self.render,
            controller_configs=config,
            camera_names = self.camera_name,
            horizon=30,
            control_freq=20,
            initialization_noise=None,
        )
        
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1, 0.9]),
                    high=np.array([1, 1, 1, 1]),
                    dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self._seed = None
        self.context = {}
        
    def step(self, action, writer=None):
        done = False
        pos = []
        front_view = [] 
        success = 0
        quat = []
        obs, reward, done, info = self.env.step(action)
        
        pos.append(obs['cube_pos'])
        quat.append(obs['cube_quat'])
        if self.render:
            front_view.append(info[self.camera_name + "_image"].copy())
            if writer is not None:
                writer.append_data(cv2.rotate(info[self.camera_name + "_image"], cv2.ROTATE_180))
            # self.env.render()
            time.sleep(0.01)
        if self.env._check_success():
            success = 1

        obs = self.env.get_obs()
        
        info = {'reward': reward, 
                'success': success,
                'render_image': front_view,
                'pos': pos,
                'quat': quat,
                'gripper_to_cube_dist': np.linalg.norm(info['gripper_to_cube_pos']),
                'cube_dist_to_goal': info['cube_dist_to_goal'],
                'orientation_error': info['orientation_error']}
        
        
        return obs, reward, done, info

    def reset(self):
        self.env.reset()
        
        use_rk4 = False # Enable RK4 integrator
        
        if use_rk4:
            self.env.sim.model.opt.integrator = 1
            
        return self.env.get_obs()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
        else:
            self._seed = np.random.seed(0)
    
    def get_context(self):
        # print (self.env)
        return self.env.get_context()
    
    def set_context(self, context):
        self.env.set_context(context)
        self.context = context
    
    def print_context(self):
        context = self.get_context()
        for k, v in context.items():
            print(f"{k}: {v}")

    def get_goal(self):
        # get the goal from the env
        return self.env.target_pos
    


# if __name__ == '__main__':
#     main()
