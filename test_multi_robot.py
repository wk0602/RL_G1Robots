#!/usr/bin/env python3
"""
测试多机器人环境的创建和运行
每个环境包含两个机器人和一个沙发，验证环境间的隔离和环境内的交互
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from isaacgym import gymapi

# 导入我们的测试环境
from legged_gym.envs.test.test_env import TestG1Robot
from legged_gym.envs.test.test_config import TestG1RobotCfg

def create_test_config():
    """创建测试配置，减少环境数量以便观察"""
    cfg = TestG1RobotCfg()
    
    # 减少环境数量以便测试和观察
    cfg.env.num_envs = 4  # 只创建4个环境进行测试
    
    # 设置机器人资产路径
    cfg.asset.file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_23dof.urdf"
    cfg.asset.name = "G1_Robot"
    cfg.asset.foot_name = "ankle_roll"
    cfg.asset.penalize_contacts_on = ["torso"]
    cfg.asset.terminate_after_contacts_on = ["torso"]
    cfg.asset.self_collisions = 0
    
    # 设置初始状态
    cfg.init_state.pos = [0.0, 0.0, 0.8]  # x,y,z [m]
    cfg.init_state.rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
    cfg.init_state.lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
    cfg.init_state.ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
    
    # 设置地形
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.curriculum = False
    
    # 设置观察者视角
    cfg.viewer.pos = [0, -4, 2]
    cfg.viewer.lookat = [0, 0, 1]
    
    return cfg

def test_multi_robot_environment():
    """测试多机器人环境的创建和基本功能"""
    print("🚀 开始测试多机器人环境...")
    
    # 创建配置
    cfg = create_test_config()
    
    # 设置仿真参数
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
    sim_params.use_gpu_pipeline = False  # 使用CPU进行测试
    
    try:
        # 创建环境
        print("📦 正在创建TestG1Robot环境...")
        env = TestG1Robot(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device="cpu",
            headless=True  # 无头模式进行测试
        )
        
        print("✅ 环境创建成功！")
        
        # 验证环境结构
        print("\n🔍 验证环境结构...")
        assert len(env.envs) == cfg.env.num_envs, f"环境数量不匹配: {len(env.envs)} != {cfg.env.num_envs}"
        assert len(env.actor_handles) == cfg.env.num_envs, f"Actor句柄数量不匹配: {len(env.actor_handles)} != {cfg.env.num_envs}"
        
        # 验证每个环境的actor数量
        for i, env_actors in enumerate(env.actor_handles):
            assert len(env_actors) == 3, f"环境{i}的actor数量不正确: {len(env_actors)} != 3 (应该是2个机器人+1个沙发)"
        
        print("✅ 环境结构验证通过！")
        
        # 测试多机器人信息获取
        print("\n📊 获取多机器人环境信息...")
        info = env.get_multi_robot_info()
        print(f"环境数量: {info['num_envs']}")
        print(f"每环境机器人数: {info['robots_per_env']}")
        print(f"总机器人数: {info['total_robots']}")
        
        # 测试环境重置
        print("\n🔄 测试环境重置...")
        obs, privileged_obs = env.reset()
        print(f"观测维度: {obs.shape}")
        if privileged_obs is not None:
            print(f"特权观测维度: {privileged_obs.shape}")
        
        # 测试动作执行
        print("\n🎮 测试动作执行...")
        actions = torch.zeros(cfg.env.num_envs, cfg.env.num_actions)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        print(f"奖励: {rewards.mean().item():.4f}")
        print(f"完成环境数: {dones.sum().item()}")
        
        # 测试辅助方法
        print("\n🛠️ 测试辅助方法...")
        primary_handle = env._get_primary_robot_handle(0)
        all_robot_handles = env._get_all_robot_handles(0)
        sofa_handle = env._get_sofa_handle(0)
        
        print(f"环境0的主机器人句柄: {primary_handle}")
        print(f"环境0的所有机器人句柄数: {len(all_robot_handles)}")
        print(f"环境0的沙发句柄: {sofa_handle}")
        
        print("\n🎉 所有测试通过！多机器人环境工作正常！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🤖 多机器人环境测试脚本")
    print("=" * 60)
    
    success = test_multi_robot_environment()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 测试成功完成！")
        print("🎯 多机器人环境已准备好进行训练")
    else:
        print("❌ 测试失败！")
        print("🔧 请检查配置和实现")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 