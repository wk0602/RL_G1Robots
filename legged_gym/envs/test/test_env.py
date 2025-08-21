
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np

class TestG1Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    
    def _resample_commands(self, env_ids):
        """ 重写命令重采样函数，确保机器人沿着课程学习方向前进
        
        在课程学习地形中：
        - X方向（前进方向）：难度从简单到复杂递增
        - 机器人需要主要沿+X方向移动，体验不同难度的地形
        
        Args:
            env_ids (List[int]): 需要新命令的环境ID列表
        """
        from isaacgym.torch_utils import torch_rand_float
        
        # 主要沿+X方向前进，这是课程学习的难度递增方向
        # 添加一些随机性，让训练更加鲁棒
        self.commands[env_ids, 0] = torch_rand_float(0.8, 1.5, (len(env_ids), 1), device=self.device).squeeze(1)
        
        # 允许轻微的左右调整，帮助机器人适应地形变化
        self.commands[env_ids, 1] = torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device).squeeze(1)
        
        # 允许轻微的转向调整，但主要保持前进方向
        self.commands[env_ids, 2] = torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device).squeeze(1)
        
        # 保持朝向稳定
        self.commands[env_ids, 3] = 0.0
        
        # 确保小的命令不会被设置为零（保持前进动力）
        # 这里我们只对Y方向速度应用最小阈值，确保X方向始终有前进速度
        self.commands[env_ids, 1] *= (torch.abs(self.commands[env_ids, 1]) > 0.05).float()
        
    def _get_phase_info(self):
        """ 获取步态相位信息，用于步态同步 """
        return {
            'phase': self.phase,
            'phase_left': self.phase_left,
            'phase_right': self.phase_right,
            'leg_phase': self.leg_phase
        }
    
    def get_terrain_curriculum_info(self):
        """ 获取当前地形课程学习信息，用于监控训练进度 """
        if hasattr(self, 'terrain_levels'):
            return {
                'current_terrain_levels': self.terrain_levels,
                'max_terrain_level': self.max_terrain_level,
                'terrain_types': self.terrain_types,
                'average_difficulty': torch.mean(self.terrain_levels.float()).item()
            }
        return None
    
    def _get_env_origins(self):
        """ 重写环境原点设置，确保所有机器人都从最简单的地形开始
        
        在课程学习中：
        - 行索引 i = 0 表示最简单的地形
        - 行索引 i = num_rows-1 表示最复杂的地形
        - 我们让所有机器人都从 i = 0 开始，然后沿着 +X 方向前进
        """
        if self.terrain is not None:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            
            # 强制所有机器人从最简单的地形开始（第0行）
            self.terrain_levels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            
            # 在不同的地形类型中分布机器人（不同的列）
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device), 
                (self.num_envs / self.cfg.terrain.num_cols), 
                rounding_mode='floor'
            ).to(torch.long)
            
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            
            # 设置环境原点坐标
            self.env_origins[:, 0] = self.terrain_origins[self.terrain_levels, self.terrain_types, 0]
            self.env_origins[:, 1] = self.terrain_origins[self.terrain_levels, self.terrain_types, 1] 
            self.env_origins[:, 2] = self.terrain_origins[self.terrain_levels, self.terrain_types, 2]
            
            print(f"✅ 所有 {self.num_envs} 个机器人都从最简单的地形开始 (terrain_level=0)")
            print(f"📊 地形类型分布: {torch.bincount(self.terrain_types)}")
        else:
            # 如果没有地形，使用默认的网格布局
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            for i in range(self.num_envs):
                self.env_origins[i, 0] = (i % self.num_envs_per_row) * self.cfg.env.env_spacing
                self.env_origins[i, 1] = (i // self.num_envs_per_row) * self.cfg.env.env_spacing
                self.env_origins[i, 2] = 0.
    