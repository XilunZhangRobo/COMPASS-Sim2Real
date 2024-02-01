from robosuite import load_controller_config
from robosuite.environments.manipulation.pusher import Pusher

import gym
import numpy as np
import time
import cv2
import copy
import matplotlib.pyplot as plt



USE_BIRDVIEW = True

class PusherSingleAction(gym.Env):
    def __init__(self, render=False):
        self.render = render
        
        config = load_controller_config(
                     custom_fpath="../robosuite/controllers/config/osc_position_custom.json")

        self.camera_name = "birdview"
        self.env = Pusher(
            robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=self.render,
            has_offscreen_renderer=self.render,
            use_camera_obs=self.render,
            controller_configs=config,
            camera_names=self.camera_name,
            control_freq=20,
        )
        # Define discrete action space for the pusher environment
        # Action space is 4D: (x, y location of the pusher, push angle, push velocity)
        
        self.action_space = gym.spaces.Box(low=np.array([-0.16, 0.07, -0.01*np.pi, 0.4]),
                                           high=np.array([-0.14, 0.08, 0.01*np.pi, 0.5]),
                                           dtype=np.float32) 
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.context = {}
        self._seed = None

    def step(self, action, writer=None):
        """
        :param action: 4D action vector (x, y location of the pusher, push angle, push velocity)
        """
        act = copy.deepcopy(action)
        rewards = []
        done = False
        state_keys = ['plate1_pos','plate2_pos']
        states = {x: [] for x in state_keys}
        horizon = 50
        camera_view = []
        tmp = self.env.sim.data.get_joint_qpos("knob_joint0")
        tmp[:2] = [action[0], action[1]]
        self.env.sim.data.set_joint_qpos("knob_joint0", tmp)
        
        
        """ === Modify the meaning of anlge (action[2]) === """
        plate1_pos = self.env.sim.data.get_joint_qpos("plate1_joint0")
        # Calculate the angle between the knob and the plate1 COM, as the reference angle
        y_diff = plate1_pos[1] - action[1]
        x_diff = plate1_pos[0] - action[0]
        reference_angle = np.arctan2(y_diff, x_diff)
        
        # update the action[2] with the reference angle
        act[2] = reference_angle + action[2]
        
        x_init, y_init, theta, v = action
        plate1_pos = self.env.sim.data.get_joint_qpos("plate1_joint0")
        ## Setting where knob needs to stop to match the real kinova pusher action
        move_distance = np.linalg.norm([x_init, y_init]-plate1_pos[:2])-0.045
        push_time = move_distance / (self.env.velocity_delay * v)
                
        for _ in range(horizon):
            self.set_knob_vel(act,push_time)
                        
            obs, reward, done, infos = self.env.step(np.zeros(4))
            if writer is not None:
                writer.append_data(
                                cv2.rotate(infos[self.camera_name+"_image"], 
                                cv2.ROTATE_180)
                            )
                
            if self.render:
                time.sleep(0.01)
            
            # get state of the env
            for key in state_keys:
                states[key].append(obs[key])
            rewards.append(reward)
            if self.render:
                camera_view.append(infos[self.camera_name + "_image"].copy())
        
        # process observations
        observation = self.get_obs()
        done = True
        info = {
            'rewards': rewards,
            **states,
            'success': self.env._check_success(),
            'dist_to_goal': self.env.get_plate_to_goal(),
            'dist_to_goal_plate1': self.env.get_plate1_to_goal(),
            'frontview_image': camera_view,
        }
        
        return observation, rewards[-1], done, info


    def set_knob_vel(self, action, push_time):
        vel = np.zeros(6) # x, y, z, around x, around y, around z
        if self.env.sim.data.time < push_time:
            x_init, y_init, theta, v = action
            velocity = self.env.velocity_delay * v
            vel_x = velocity * np.cos(theta)
            vel_y = velocity * np.sin(theta)
            vel[0] = vel_x
            vel[1] = vel_y
        self.env.sim.data.set_joint_qvel("knob_joint0", vel)
    
    def reset(self):
        self.env.reset()
        return self.get_obs()

    def render(self):
        pass

    def close(self):
        # print("Closing environment")
        self.env.close()

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
        else:
            self._seed = np.random.seed(0)

    def get_obs(self):
        goal = self.env.target_pos
        plate1_pos = self.env.plate1_init_pos
        plate2_pose = self.env.plate2_init_pos
        # Normalize the observation
        goal = goal * 0
        plate1_pos = (plate1_pos - (-0.15, 0, 0.8)) / (0.01, 0.01, 1)
        plate2_pose = (plate2_pose - (-0.075, -0.075, 0.8)) / (0.01, 0.01, 1)
        obs = np.concatenate([goal, plate1_pos, plate2_pose])
        assert len(obs.shape) == 1, "observation should be 1D"
        return obs.astype(np.float32)
    
    def get_context(self):
        return self.env.get_context()

    def set_context(self, context):
        self.env.set_context(context)
        self.context = context

# if __name__ == '__main__':
#     main()
