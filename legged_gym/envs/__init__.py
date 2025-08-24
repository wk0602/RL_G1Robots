from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.test.test_config import TestG1RoughCfg, TestG1RoughCfgPPO
from legged_gym.envs.test.test_env import TestG1Robot
from legged_gym.envs.legged_g1.legged_g1_config import LeggedG1RoughCfg, LeggedG1RoughCfgPPO
from legged_gym.envs.legged_g1.legged_g1_env import LeggedG1Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "test", TestG1Robot, TestG1RoughCfg(), TestG1RoughCfgPPO())
task_registry.register( "legged_g1", LeggedG1Robot, LeggedG1RoughCfg(), LeggedG1RoughCfgPPO())