#定义智能体的文件，同时把回放缓冲区和Ornstein-Uhlenbeck噪点相关实现放在了这里。方便调用。
from agents.actor import Actor
from agents.critic import Critic

import numpy as np
import copy

import random
from collections import namedtuple, deque
#使用回放存储器，缓冲器来进行存储，调用经验元组。
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    #随机的取出一批过去的经验（所谓的经验调用）
    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


#具有所需特性的特定噪点流程。根据高斯分布生成随机样本。
#这是为了向动作中添加噪点，促进探索行为。但是因为动作是应用到飞行器上的力和扭矩，所以连续动作不能变化太大。
#项目的设定还是挺好的，但是我真的很不习惯项目中给出的参考代码，对函数参数完全没有说明……可能是故意强迫去看论文？？
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu) #按照论坛提示，进行修正
        #self.state=self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# 下面是智能体的实现 DDPG,使用行动者模型和策略模型。
#  每个模型的两个副本都需要，一个local model 一个target model, Fix Q target. 拆分被更新的参数和生成目标值的参数。

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model 行动者模型
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model 评价者模型
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process 噪音处理流程 ，均值，
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.3
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory 
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Algorithm parameters 衰减，或者说未来折扣率。
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters 最后进行软更新，由参数tau控制。

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward 记录经验和这一步的信息。
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory 如果已经存储了足够的部署
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample() #随机取出部分经验,进行学习
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0] #取出在state中按当前策略执行的动作
        return list(action + self.noise.sample())  # add some noise for exploration 流程中向动作添加一些噪点，促进探索行为。

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        # 首先将经验数据分成多个数组
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models 根据actor_target的模型，通过next_state集合获取动作
        #     Q_targets_next = critic_target(next_state, actor_target(next_state)) #在critic_target上，使用state和action获取Q值。
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    
    #在多个经验训练之后，我们将新学习的权重（本地模型）复制到目标模型中。
    #但是单个批次可能会引入向流程中引入很多bias偏差，所有要进行软更新，tau就是这个软更新的参数。local只有tau控制的内容量更新到target

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

