from isaacgym import gymapi, gymtorch
import torch

def _load_box_asset(gym, sim, box_length, box_width, box_height, num_envs, density, device):
    box_asset = []
    asset_density = torch.zeros(num_envs).to(device)

    for env_id in range(num_envs):
        asset_options = gymapi.AssetOptions()
        asset_options.density = density

        asset_density[env_id] = asset_options.density
        box_asset.append(gym.create_box(sim, box_length, box_width, box_height, asset_options))

    return box_asset, asset_density

def _build_box(gym, env_id, env_ptr, box_asset, box_handles):
    col_group = env_id
    col_filter = 0
    segmentation_id = 0

    default_pose = gymapi.Transform()
    default_pose.p.x = 3.0

    box_handle = gym.create_actor(
        env_ptr, box_asset[env_id], default_pose, "box", col_group, col_filter, segmentation_id)
    
    box_handles.append(box_handle)
    return box_handles

