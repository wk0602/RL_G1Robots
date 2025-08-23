from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TestG1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,
           # torso / waist
           'waist_yaw_joint': 0.0,
           'waist_roll_joint': 0.0,
           'waist_pitch_joint': 0.0,
           # left arm
           'left_shoulder_pitch_joint': 0.0,
           'left_shoulder_roll_joint': 0.0,
           'left_shoulder_yaw_joint': 0.0,
           'left_elbow_joint': 0.0,
           'left_wrist_roll_joint': 0.0,
           'left_wrist_pitch_joint': 0.0,
           'left_wrist_yaw_joint': 0.0,
           # left hand
           'left_hand_palm_joint': 0.0,  # fixed in URDF but keep 0 if present
           'left_hand_thumb_0_joint': 0.0,
           'left_hand_thumb_1_joint': 0.0,
           'left_hand_thumb_2_joint': 0.0,
           'left_hand_middle_0_joint': 0.0,
           'left_hand_middle_1_joint': 0.0,
           'left_hand_index_0_joint': 0.0,
           'left_hand_index_1_joint': 0.0,
           # right arm
           'right_shoulder_pitch_joint': 0.0,
           'right_shoulder_roll_joint': 0.0,
           'right_shoulder_yaw_joint': 0.0,
           'right_elbow_joint': 0.0,
           'right_wrist_roll_joint': 0.0,
           'right_wrist_pitch_joint': 0.0,
           'right_wrist_yaw_joint': 0.0,
           # right hand
           'right_hand_palm_joint': 0.0,  # fixed in URDF but keep 0 if present
           'right_hand_thumb_0_joint': 0.0,
           'right_hand_thumb_1_joint': 0.0,
           'right_hand_thumb_2_joint': 0.0,
           'right_hand_middle_0_joint': 0.0,
           'right_hand_middle_1_joint': 0.0,
           'right_hand_index_0_joint': 0.0,
           'right_hand_index_1_joint': 0.0,
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 140
        num_privileged_obs = 143
        num_actions = 43
        episode_length_s = 60

    # 冻结配置：允许在训练下半身时冻结上半身关节
    class freeze:
        enable_upper_body = True
        # 通过名称子串匹配上半身相关关节（不包含腿部/髋/膝/踝）
        name_substrings = [
            'waist_', 'shoulder', 'elbow', 'wrist',
            'hand_', 'thumb', 'index', 'middle'
        ]
        # 对被冻结关节放大 PD 增益，增强抱持姿态能力，避免坍塌
        kp_scale = 3.0
        kd_scale = 2.0

    # 添加commands配置，让机器人沿着难度递增方向前进
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.  # 命令重采样时间间隔[秒]
        heading_command = False  # 不使用朝向命令，直接控制角速度
        class ranges:
            # 关键：在课程学习地形中，X方向是难度递增方向！
            # 机器人需要沿着+X方向前进，从简单地形走向复杂地形
            lin_vel_x = [0.8, 1.5]   # 沿+X方向前进（难度递增方向），速度0.8-1.5 m/s
            lin_vel_y = [0, 0]  # 允许轻微的左右调整，避免过于僵硬
            ang_vel_yaw = [0, 0] # 允许轻微转向调整，保持前进方向
            heading = [-3.14, 3.14]  # 保持默认朝向范围（虽然不会使用

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.8, 1.4]  # 缩小摩擦力范围，提高最小值，让机器人更稳定
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist': 50,
                     'shoulder': 60,
                     'elbow': 40,
                     'wrist': 10,
                     'thumb': 5,
                     'index': 5,
                     'middle': 5,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist': 2,
                     'shoulder': 3,
                     'elbow': 2,
                     'wrist': 1,
                     'thumb': 0.5,
                     'index': 0.5,
                     'middle': 0.5,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_with_hand.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class TestG1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic"
        max_iterations = 50000
        run_name = ''
        experiment_name = 'test_g1'
