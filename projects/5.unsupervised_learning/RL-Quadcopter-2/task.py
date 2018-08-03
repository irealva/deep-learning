import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
#     def get_reward_new2(self):
#         #return reward
#         reward = 0.0

#         # Reward positive velocity along z-axis
#         reward += self.sim.v[2]

#         # Reward positions close to target along z-axis
#         reward -= (abs(self.sim.pose[ 2] - self.target_pos[ 2])) / 2.0 

#         # A lower sensativity towards drifting in the xy-plane
#         reward -= (abs(self.sim.pose[:2] - self.target_pos[:2])).sum() / 4.0

        
#         reward -= (abs(self.sim.angular_v[:3])).sum()

#         return reward
        
    def get_reward_new(self):
        reward = 0
          
        # tasks suggested from slack forum!
        # reward staying on the target position coordinate z   
        reward = reward - (0.5 * (abs(self.sim.pose[2] - self.target_pos[2])))
        # reward = reward - ((abs(self.sim.pose[2] - self.target_pos[2])))

        # reward agent for not moving much on x, y axis
        reward = reward - (0.25 * (abs(self.sim.pose[:2] - self.target_pos[:2])).sum())
        # reward = reward - ((abs(self.sim.pose[:2] - self.target_pos[:2])).sum()) 

        # bias for euler angles
        reward = reward - (abs(self.sim.angular_v[:3])).sum()
        
        # Added based on review #1 from udacity reviewer
        # Reward vertical velocity on z-axis to make sure the quad lifts off
        #reward = reward + (4 * self.sim.v[2])
        reward = reward + self.sim.v[2]
        
        # Added based on review #1 from udacity reviewer
        # You can also normalize the rewards between -1 and 1 to better help the neural networks learn the gradient parameters 
        # without high magnitude deviations.
        reward = np.clip(reward, -1, 1)
                
        return reward

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
             
        reward = 0    
        
        # task is suggested from slack forum!
        # bias for euler angles
        euler_bias = 10
        eulers_angle_penalty = abs(self.sim.pose[3:] - self.target_pos).sum() - euler_bias
        
       # reward staying on the target position coordinate z
        z_reward = abs(self.sim.pose[2] - self.target_pos[2])
                      
        # reward agent for not moving much on x, y axis
        other_reward = abs(self.sim.pose[:2] - self.target_pos[:2]).sum()                    
                      
        penalties = (-.0003*(other_reward) - .0009*(z_reward) - .0003*(eulers_angle_penalty))/3
          
        # adding a reward for every extra second of flying
        #reward = 1 + penalties
        
        reward = reward - penalties
               
        # Added based on review #1 from udacity reviewer
        # Reward vertical velocity on z-axis to make sure the quad lifts off
        reward = reward + self.sim.v[2]
        
        # Added based on review #1 from udacity reviewer
        # You can also normalize the rewards between -1 and 1 to better help the neural networks learn the gradient parameters 
        # without high magnitude deviations.
        reward = np.clip(reward, -1, 1)
        
        return reward
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            
            # Added based on review #1 from udacity reviewer
            # Penalize a crash
#             if done and self.sim.time < self.sim.runtime: 
#                 reward = -1
                
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        
        # Randomise the starting pose of the quadrocopter on each reset
        random_pose = np.copy(self.sim.init_pose)
        random_pose[:3] += np.random.normal(0.0, 0.30, 3)
        random_pose[3:] += np.random.normal(0.0, 0.03, 3)
        self.sim.pose = np.copy(random_pose)
        
        return state