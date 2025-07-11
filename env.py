# 文件名: env_33bus_pv_only.py (已更正)

import numpy as np
import pandapower as pp
import pandapower.converter as pc
import copy

def Relu(x: np.ndarray):
    return np.maximum(0, x)

class grid_case:
    def __init__(self, load_pu, gene_pu, id_iber):
        """初始化33节点光伏调压环境"""
        self.id_iber = id_iber
        self.load_pu = load_pu
        self.gene_pu = gene_pu
        self.action_dim = len(self.id_iber)
        self.step_n = 0
        self.done = False
        
        self.model = pc.from_mpc('case33_bw.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        self.n_bus = len(self.model.bus)

        # 仅创建可控的光伏逆变器 (IBVR) - 已修正函数名
        for i in self.id_iber:
            pp.create_sgen(self.model, bus=i, p_mw=0, q_mvar=0, name='IBVR', scaling=1.0, in_service=True,
                           max_p_mw=4, min_p_mw=0, max_q_mvar=2, min_q_mvar=-2, controllable=True)
        
        pp.runpp(self.model, algorithm='bfsw')
        self.observation_space = copy.deepcopy(np.hstack(
            (np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), np.array(self.model.res_bus.q_mvar),
             np.zeros(self.action_dim))))

        self.init_load_p_mw = copy.deepcopy(self.model.load.p_mw)
        self.init_load_q_mvar = copy.deepcopy(self.model.load.q_mvar)
        self.init_sgen_p_mw = copy.deepcopy(self.model.sgen.p_mw)

        # 加载预生成的数据
        self.load_pu_data = np.load('two33load.npy')
        self.gene_pu_data = np.load('two33gen.npy')

    def action_clip(self, action: np.ndarray) -> np.ndarray:
        """将[-1, 1]的动作映射到设备的实际约束范围"""
        low = self.model.sgen.min_q_mvar.values
        high = self.model.sgen.max_q_mvar.values
        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor
        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)
        return action

    def step_model(self, action: np.ndarray):
        """环境步进函数"""
        self.step_n += 1
        action = self.action_clip(action)
        
        self.model.sgen.q_mvar = action
        self.model.load.p_mw = self.load_pu_data[self.step_n - 1] * self.init_load_p_mw
        self.model.load.q_mvar = self.load_pu_data[self.step_n - 1] * self.init_load_q_mvar
        self.model.sgen.p_mw = self.gene_pu_data[self.step_n - 1]

        pp.runpp(self.model, algorithm='bfsw')
        
        # 计算奖励
        violation_M = Relu(self.model.res_bus.vm_pu - 1.05).sum()
        violation_N = Relu(0.95 - self.model.res_bus.vm_pu).sum()
        grid_loss = -self.model.res_line.pl_mw.sum()
        
        reward_p = grid_loss
        reward_v = -violation_M - violation_N
        reward = np.array((reward_p, reward_v))

        violation = 1 if (violation_M + violation_N > 0) else 0

        # 获取下一个状态
        next_state = np.hstack((np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), 
                                np.array(self.model.res_bus.q_mvar), action))
        
        # 准备下一个时间点的扰动，用于更新观测值
        if self.step_n < len(self.load_pu_data):
            self.model.load.p_mw = self.load_pu_data[self.step_n] * self.init_load_p_mw
            self.model.load.q_mvar = self.load_pu_data[self.step_n] * self.init_load_q_mvar
            self.model.sgen.p_mw = self.gene_pu_data[self.step_n]
            pp.runpp(self.model, algorithm='bfsw')
            new_observation = np.hstack((np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), 
                                         np.array(self.model.res_bus.q_mvar), action))
        else:
            new_observation = next_state # 如果是最后一个时间点，则状态不变
        
        return next_state, reward, self.done, violation, violation_M, violation_N, grid_loss, new_observation

    def reset(self):
        """重置环境"""
        self.done = False
        self.step_n = 0
        return self.observation_space