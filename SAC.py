# 文件名: sac_33bus_pv_only.py (已最终修正)
# 作用: 使用SAC算法训练智能体在33节点光伏调压环境中进行控制

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pandas.core.frame import DataFrame
import pandas as pd

# 导入我们精简后的环境
from env import grid_case

# 设置随机种子以保证结果可复现
seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# --- 经验回放池 (与原项目保持一致) ---
class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 2], dtype=np.float32) 
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

# --- 神经网络初始化 (与原项目保持一致) ---
def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

# --- 策略网络 (Actor) 和价值网络 (Critic) 定义 (与原项目SAC部分保持一致) ---
class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, log_std_min: float = -20, log_std_max: float = 2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden1 = nn.Linear(in_dim, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.mu_layer = init_layer_uniform(nn.Linear(512, out_dim))
        self.log_std_layer = init_layer_uniform(nn.Linear(512, out_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)

class CriticQ(nn.Module):
    def __init__(self, in_dim: int):
        super(CriticQ, self).__init__()
        self.hidden1 = nn.Linear(in_dim, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.out = init_layer_uniform(nn.Linear(512, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)

# --- SAC智能体 (Agent) 定义 ---
class SACAgent:
    def __init__(self, env, memory_size: int, batch_size: int, gamma: float = 0.9, 
                 tau: float = 5e-3, initial_random_steps: int = 1e4):
        obs_dim = env.observation_space.shape[0]
        
        # 修正此处的错误
        self.action_dim = env.action_dim
        
        self.env = env
        self.memory = ReplayBuffer(obs_dim, self.action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.target_entropy = -np.prod((self.action_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = Actor(obs_dim, self.action_dim).to(self.device)
        self.qf_1 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_2 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target1 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target2 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target1.load_state_dict(self.qf_1.state_dict())
        self.qf_target2.load_state_dict(self.qf_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=3e-4)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=3e-4)

        self.total_step = 0
        self.is_test = False
        self.state = self.env.reset()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.total_step < self.initial_random_steps and not self.is_test:
            return np.random.uniform(-1, 1, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action, _, _ = self.actor(state_tensor)
        return action.cpu().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        _, reward, done, violation, _, _, grid_loss, new_observation = self.env.step_model(action)
        self.state = new_observation
        return new_observation, reward, done, {"grid_loss": grid_loss, "violation": violation}

    def update_model(self) -> Tuple[float, float, float]:
        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.FloatTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"]).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        
        with torch.no_grad():
            next_action, log_prob, _ = self.actor(next_state)
            q_target_next1 = self.qf_target1(next_state, next_action)
            q_target_next2 = self.qf_target2(next_state, next_action)
            q_target_next = torch.min(q_target_next1, q_target_next2)
            alpha = self.log_alpha.exp()
            combined_reward = reward[:, 0].reshape(-1, 1) + 50 * reward[:, 1].reshape(-1, 1)
            q_target = combined_reward + self.gamma * (1 - done) * (q_target_next - alpha * log_prob)

        qf1_loss = F.mse_loss(self.qf_1(state, action), q_target)
        self.qf_1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf_1_optimizer.step()

        qf2_loss = F.mse_loss(self.qf_2(state, action), q_target)
        self.qf_2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf_2_optimizer.step()

        qf_loss = qf1_loss + qf2_loss
        
        new_action, log_prob, _ = self.actor(state)
        q_pred1 = self.qf_1(state, new_action)
        q_pred2 = self.qf_2(state, new_action)
        q_pred = torch.min(q_pred1, q_pred2)
        
        actor_loss = (alpha.detach() * log_prob - q_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (-self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._target_soft_update()

        return actor_loss.item(), qf_loss.item(), alpha_loss.item()

    def _target_soft_update(self):
        for t_param, l_param in zip(self.qf_target1.parameters(), self.qf_1.parameters()):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)
        for t_param, l_param in zip(self.qf_target2.parameters(), self.qf_2.parameters()):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)

    def train(self, num_frames: int, plotting_interval: int = 400):
        self.is_test = False
        actor_losses, qf_losses, alpha_losses = [], [], []
        scores, grid_losses, violation_rates = [], [], []
        score, grid_loss_sum, violation_sum = 0, 0, 0

        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(self.state)
            next_state_obs, reward, done, info = self.step(action)
            
            self.memory.store(self.state, action, reward, next_state_obs, done)
            
            score += reward.sum()
            grid_loss_sum += info["grid_loss"]
            violation_sum += info["violation"]

            if len(self.memory) >= self.batch_size and self.total_step > self.initial_random_steps:
                losses = self.update_model()
                actor_losses.append(losses[0])
                qf_losses.append(losses[1])
                alpha_losses.append(losses[2])

            if self.total_step % 96 == 0:
                day = self.total_step // 96
                scores.append(score/96)
                grid_losses.append(grid_loss_sum/96)
                violation_rates.append(violation_sum / 96)
                score, grid_loss_sum, violation_sum = 0, 0, 0
                print(f"Day {day} completed. Avg Reward: {scores[-1]:.2f}")

            if self.total_step % plotting_interval == 0:
                self._plot(self.total_step, scores, grid_losses, violation_rates, actor_losses, qf_losses, alpha_losses)
        
        data_to_save = {"scores_avg": scores, "grid_losses_avg": grid_losses, "violation_rates_avg": violation_rates}
        df = DataFrame(data_to_save)
        df.to_csv('train_sac_33bus_pv_only.csv')
        
        losses_to_save = {"actor_losses": actor_losses, "qf_losses": qf_losses, "alpha_losses": alpha_losses}
        df_loss = DataFrame(losses_to_save)
        df_loss.to_csv('train_loss_sac_33bus_pv_only.csv')

    def _plot(self, frame_idx: int, scores: List[float], grid_losses: List[float], violation_rates: List[float], 
              actor_losses: List[float], qf_losses: List[float], alpha_losses: List[float]):
        clear_output(True)
        plt.figure(figsize=(20, 10))
        plt.subplot(231)
        plt.title(f"Frame {frame_idx}. Avg Score: {np.mean(scores[-10:]):.2f}")
        plt.plot(scores)
        plt.subplot(232)
        plt.title("Avg Grid Loss (Negative)")
        plt.plot(grid_losses)
        plt.subplot(233)
        plt.title("Avg Violation Rate")
        plt.plot(violation_rates)
        plt.subplot(234)
        plt.title("Actor Loss")
        plt.plot(actor_losses)
        plt.subplot(235)
        plt.title("Critic Loss")
        plt.plot(qf_losses)
        plt.subplot(236)
        plt.title("Alpha Loss")
        plt.plot(alpha_losses)
        plt.show()

# --- 主执行函数 ---
if __name__ == "__main__":
    num_frames = 96 * 300
    memory_size = 30000
    batch_size = 128
    initial_random_steps = 96 * 10

    load_pu = np.load('load96.npy')
    gene_pu = np.load('gen96.npy')
    id_iber_33 = [17, 21, 24]

    env = grid_case(load_pu, gene_pu, id_iber_33)
    agent = SACAgent(env, memory_size, batch_size, gamma=0.9, initial_random_steps=initial_random_steps)
    agent.train(num_frames, plotting_interval=4000)