# 文件名: generate_data_33bus.py (已更正)

import numpy as np
import pandapower as pp # 导入 pandapower
import pandapower.converter as pc
import copy

print("开始生成33节点系统的仿真数据...")

class grid_case:
    def __init__(self, env_name, load_pu, gene_pu, id_iber):
        """初始化环境并生成数据"""
        self.id_iber = id_iber
        self.load_pu = load_pu
        self.gene_pu = gene_pu
        self.env_name = env_name

        if self.env_name == 33:
            self.model = pc.from_mpc('case33_bw.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        else:
            raise ValueError("此脚本仅为33节点系统设计")

        self.n_bus = len(self.model.bus)
        
        # 仅创建光伏逆变器 (IBVR) - 已修正函数名
        for i in self.id_iber:
            pp.create_sgen(self.model, bus=i, p_mw=0, q_mvar=0, name='IBVR', scaling=1.0, in_service=True,
                           max_p_mw=4, min_p_mw=0, max_q_mvar=2, min_q_mvar=-2, controllable=True)

        self.init_load_p_mw = copy.deepcopy(self.model.load.p_mw)

        # 加入随机性并生成数据文件
        self.load_pu = self.load_pu[:, np.newaxis] * np.ones_like(self.init_load_p_mw)[np.newaxis, :]
        self.load_pu[:370*96, :] *= np.random.uniform(low=0.8, high=1.2, size=self.load_pu[:370*96, :].shape)
        
        self.gene_pu = self.gene_pu[:, np.newaxis] * np.ones_like(self.id_iber)[np.newaxis, :]
        self.gene_pu[:370*96, :] *= np.random.uniform(low=0.8, high=1.2, size=self.gene_pu[:370*96, :].shape)
        
        np.save('two' + str(self.n_bus) + 'load.npy', self.load_pu)
        np.save('two' + str(self.n_bus) + 'gen.npy', self.gene_pu)

if __name__ == "__main__":
    try:
        load_pu = np.load('load96.npy')
        gene_pu = np.load('gen96.npy')
        
        # 33节点系统的光伏逆变器位置
        id_iber_33 = [17, 21, 24]
        
        # 初始化环境以触发数据保存
        env = grid_case(33, load_pu, gene_pu, id_iber_33)
        
        print("\n成功生成 'two33load.npy' 和 'two33gen.npy' 文件！")
        print("现在您可以运行主训练脚本了。")

    except FileNotFoundError:
        print("\n错误：找不到原始数据文件 'load96.npy' 或 'gen96.npy'。")
        print("请确保这些文件与脚本在同一目录下。")