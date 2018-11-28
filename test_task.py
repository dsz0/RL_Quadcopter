import numpy as np
from physics_sim import PhysicsSim

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class FlyTask():
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
        #task.sim.pose 飞行器Quadcopter的位置及欧拉角状态，可以体现一边翻跟斗一边飞？？欧拉角的变化应该尽量小稳定
        #task.sim.v   飞行器Quadcopter位置的变化速度  #task.sim.angular_v  飞行器的角速度，也就是飞行器自身变换的速度。
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        #self.action_low = 0  #减少动作范围。但是我对动作范围还是不太熟悉。
        #self.action_high = 900
        self.action_low = 300
        self.action_high = 600
        self.action_size = 4

        # Goal
        if target_pos is None :
            print("Setting default init pose")
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #下面这个奖励 函数是审核员的最新指导，不过这个基本上就是强化了对z轴变化的奖励，基本上是强行指导向上飞，感觉对于更普遍的任务没有什么帮助。
        #reward = 1.- 0.003*(abs(self.sim.pose[:3]-self.target_pos)).sum()
        reward = np.tanh(1.-.03*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        return reward
        
        #reward = 2*self.sim.pose[2] +0.2
        #上面的设计设计其实是对陷入了一个必须反馈值为正的怪圈，其实反馈值为负，对于我们这个设定更加方便。反正速度越大，偏差越大，分数就越低
        #z_diff = abs(self.target_pos[2] - self.sim.pose[2])
        #z_vel = self.sim.v[2]
        #xy_dist = np.sqrt(((self.target_pos[:2] - self.sim.pose[:2])**2).sum())
        #reward = -5.0*z_diff - 2.0*abs(z_vel) - xy_dist
    
    def get_reward_ex(self, old_angular_v, old_v):
        dis_from_target = sigmoid(sum(abs(self.sim.pose[:3] - self.target_pos))/3)
        #dis_from_target = sigmoid(sum(abs(self.sim.pose[:3] - self.target_pos)))
        #惩罚欧拉和速度改变
        euler_change = sigmoid(sum(abs(old_angular_v - self.sim.angular_v)))
        velocity_change = sigmoid(sum(abs(old_v - self.sim.v)))
        reward = 1. - dis_from_target - 0.02* euler_change - 0.02* velocity_change
        return reward
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state