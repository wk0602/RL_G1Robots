from pathlib import Path
import sys
import numpy as np
from isaacgym import gymapi, gymtorch
import torch

def init_sim(use_gpu: bool = False, disable_gravity: bool = False):
    # 获取Isaac Gym接口
    gym = gymapi.acquire_gym()
    # 配置仿真参数
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z                        # Z轴为向上方向
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81 if not disable_gravity else 0.0)
    sim_params.use_gpu_pipeline = use_gpu                        # 是否使用GPU管线
    # 创建仿真器（使用PhysX引擎）
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    # 添加平面地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)
    return gym, sim

def create_env_and_robots(gym, sim, urdf_path: str, fix_base: bool = True, use_gpu: bool = False):
    # 在世界坐标中创建一个环境，扩大环境范围以容纳两个机器人和沙发
    lower = gymapi.Vec3(-3, -2, 0)
    upper = gymapi.Vec3( 3,  2, 2)
    env = gym.create_env(sim, lower, upper, 1)

    # 加载URDF模型（23 DOF的G1机器人）
    asset_opts = gymapi.AssetOptions()
    asset_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)  # 关节位置驱动模式
    asset_opts.fix_base_link = True                               # 是否固定底座
    asset_opts.disable_gravity = False                            # 机器人自身是否受重力影响

    # 检查URDF路径
    urdf_path = Path(urdf_path)
    asset = gym.load_asset(sim, str(urdf_path.parent), str(urdf_path.name), asset_opts)

    # 配置资产加载选项
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False
    asset_options.collapse_fixed_joints = True  # 合并固定关节

    # 导入沙发模型
    sofa_asset_root = "resources/sofa"
    sofa_asset_file = "sofa.urdf"
    sofa_asset = gym.load_asset(sim, str(sofa_asset_root), str(sofa_asset_file), asset_options)

    # 创建两个机器人实体，肩并肩站立
    robot_handles = []
    robot_positions = [
        gymapi.Vec3(-1.0, -1.25, 0.8),  # 第一个机器人位置（左侧）
        gymapi.Vec3(-1.0, 1.25, 0.8)    # 第二个机器人位置（右侧），间距1米
    ]
    theta = np.pi / 2  # 90度弧度
    robot_orientations = [
        gymapi.Quat(0, 0, np.sin(theta/2), np.cos(theta/2)),  # 逆时针旋转90度
        gymapi.Quat(0, 0, np.sin(-theta/2), np.cos(-theta/2)) # 顺时针旋转90度
    ]
    
    for i, (pos, orn) in enumerate(zip(robot_positions, robot_orientations)):
        start_transform = gymapi.Transform()
        start_transform.p = pos    # 位置
        start_transform.r = orn    # 朝向
        robot_handle = gym.create_actor(env, asset, start_transform, f"G1_Robot_{i+1}", 0, 0)
        robot_handles.append(robot_handle)

    # 创建沙发实体
    sofa_transform = gymapi.Transform()
    sofa_transform.p = gymapi.Vec3(-8.2738, 3.96, 0.95)  # 在机器人前方的手臂高度
    sofa_transform.r = gymapi.Quat(0.7071, 0, 0, 0.7071)
    sofa_handle = gym.create_actor(env, sofa_asset, sofa_transform, "Sofa", 0, 0)
    gym.set_rigid_body_color(env, sofa_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.8, 0.8))

    return env, robot_handles, sofa_handle

def run_viewer_loop(gym, sim, env, robot_handles, box_handle):
    # 创建Viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    # 设置摄像头位置以便观察两个机器人和沙发
    cam_pos = gymapi.Vec3(0, -4, 2)
    cam_target = gymapi.Vec3(-1, 0, 1)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # 主循环
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)                # 仿真推进
        gym.fetch_results(sim, True)     # 获取结果
        gym.step_graphics(sim)           # 图形步进
        gym.draw_viewer(viewer, sim, False)  # 渲染画面
        gym.sync_frame_time(sim)         # 同步显示帧率
        
    # 关闭并清理
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


def main():
    URDF_PATH = str(Path(__file__).resolve().parents[1] / "resources/robots/g1_description/g1_23dof.urdf")
    gym, sim = init_sim(use_gpu=False, disable_gravity=False) 
    # 创建环境并加载两个机器人和搬运的物体
    env, robots, box = create_env_and_robots(gym, sim, URDF_PATH, fix_base=True, use_gpu=True)
    # 运行viewer主循环
    run_viewer_loop(gym, sim, env, robots, box)


if __name__ == "__main__":
    main()