from robosuite import load_controller_config
from robosuite.environments.manipulation.drop import Drop
import gym
import numpy as np
import time
import imageio
import cv2
import pickle as pkl  
from matplotlib import pyplot as plt

class DropSingleAction(gym.Env):
    def __init__(self, render=False):
        self.render = render
        self.release_height = 0.5
        self.scaled_damping_ratio = -10

        config = load_controller_config(
                     custom_fpath="../robosuite/controllers/config/osc_position_custom.json")
        
        # frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'
        self.camera_name = "sideview"
        self.env = Drop(
            robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=False,
            has_offscreen_renderer=self.render,
            use_camera_obs=self.render,
            controller_configs=config,
            release_height = self.release_height,
            scaled_damping_ratio=self.scaled_damping_ratio,
            camera_names = self.camera_name,
        )
        
        self.action_space = gym.spaces.Box(low=np.array([1.5]), # release height
                                    high=np.array([1.75]),
                                    dtype=np.float32)
        

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self._seed = None
        self.context = {}
        
    def step(self, action, writer=None):
        accum_reward = 0
        done = False
        z_pos = []
        x_pos = []
        z_vel = []
        pos = []
        front_view = [] 
        success = 0
        self.set_release_pos(action)
        horizon = 30
        rewards = []
        for _ in range(horizon):
            obs, reward, done, info = self.env.step(np.zeros(4))
            z_pos.append([obs['cube_pos'][2],0,0])
            x_pos.append([obs['cube_pos'][0],0,0])
            z_vel.append([obs['cube_vel'][2],0,0])
            pos.append(obs['cube_pos'])
            if self.render:
                front_view.append(info[self.camera_name + "_image"].copy())
                if writer is not None:
                    writer.append_data(cv2.rotate(info[self.camera_name + "_image"], cv2.ROTATE_180))
                # self.env.render()
                time.sleep(0.01)
            accum_reward += reward
            rewards.append(reward)
            if self.env._check_success():
                success = 1

        obs = self.get_obs()
        done = True

        info = {'accum_reward': max(rewards), 
                'success': success,
                'render_image': front_view,
                'x_pos': x_pos,
                'z_pos': z_pos,
                'z_vel': z_vel,
                'pos': pos}
        
        return obs, max(rewards), done, info

    def reset(self):
        self.env.reset()
        use_rk4 = False # Enable RK4 integrator
        
        if use_rk4:
            self.env.sim.model.opt.integrator = 1
        return self.get_obs()

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

    def get_obs(self):
        # get observation from env (observation should be the goal position)
        obs = self.get_goal()
        assert len(obs.shape) == 1, "observation should be 1D"
        return obs.astype(np.float32)

    def get_goal(self):
        # get the goal from the env
        return self.env.target_pos
    
    def set_release_pos(self, action):
        temp = self.env.sim.data.get_joint_qpos("ball_joint0")
        temp[2] = action
        self.env.sim.data.set_joint_qpos("ball_joint0", temp)


# if __name__ == '__main__':
#     main()

