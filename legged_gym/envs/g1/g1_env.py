from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch

class G1Robot(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self._default_box_width_size = 0.5
        self._default_box_length_size = 0.5
        self._default_box_height_size = 0.5

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    
    def _load_box_asset(self):
        width_box_size = self._default_box_width_size
        length_box_size = self._default_box_length_size
        height_box_size = self._default_box_height_size

        self._box_asset = []
        self.asset_density = torch.zeros(self.num_envs).to(self.device)

        for env_id in range(self.num_envs):
            asset_options = gymapi.AssetOptions()
            asset_options.density = 50.0
            self.asset_density[env_id] = asset_options.density

            self._box_asset.append(self.gym.create_box(
                self.sim, length_box_size, width_box_size, height_box_size, asset_options))
        return
    
    def _build_box(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 3.0

        self._width_box_size[env_id] = self._default_box_width_size
        self._length_box_size[env_id] = self._default_box_length_size
        self._height_box_size[env_id] = self._default_box_height_size

        box_handle = self.gym.create_actor(
            env_ptr, self._box_asset[env_id], default_pose, "box", col_group, col_filter, segmentation_id)
        
        self._box_handles.append(box_handle)
        return
    
    def _create_envs(self):
        self._box_handles = []
        sim_device = self.device
        self._width_box_size = torch.zeros(self.num_envs).to(sim_device)
        self._length_box_size = torch.zeros(self.num_envs).to(sim_device)
        self._height_box_size = torch.zeros(self.num_envs).to(sim_device)
        self._load_box_asset()

        super()._create_envs()
        return
    
    def _build_env(self, i, env_handle, robot_asset, start_pose, dof_props_asset):
        super()._build_env(i, env_handle, robot_asset, start_pose, dof_props_asset)
        self._build_box(i, env_handle)
        return

    def _build_box_tensors(self):
        num_actors = self.get_num_actors_per_env()
        
        self._box_states = self.root_states.view(
            self.num_envs, num_actors, self.root_states.shape[-1])[..., 1, :]
        self.box_standing_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self.box_held_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self._box_actor_ids = to_torch(
            num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        self._box_pos = self._box_states[..., :3]
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._box_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]
    
    def _get_noise_scale_vec(self, cfg):
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
        self._build_box_tensors()

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
    