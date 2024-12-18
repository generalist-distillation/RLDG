import gymnasium as gym
import jax 

from franka_env.envs.wrappers import (
    GripperCloseEnv,
    Quat2EulerWrapper,
    SpacemouseIntervention,
    ConnectorInsertResetWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from serl_experiments.config import DefaultTrainingConfig
from serl_experiments.connector_insert.wrappers import ConnectorEnv
import numpy as np

class EnvConfig(DefaultEnvConfig):
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "128422272758",
            "dim": (1280, 720),
            "exposure": 30000,
        },
    }
    IMAGE_RESOLUTION = 224
    TARGET_POSE = np.array([0.5983114218829463,-0.1363499926389447,0.112, np.pi, 0, np.pi/2])
    REWARD_THRESHOLD = np.array([0.005, 0.005, 0.003, 0.1, 0.1, 0.1])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.05, 0.05, 0.02, 0.05, 0.05, 0.1])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.1])

    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = 0.1

    MAX_EPISODE_LENGTH = 100
    IMAGE_CROP = {
        "wrist_1": lambda image: image[150:, 200:-200],
    }
    
    ACTION_SCALE = np.array([0.015, 0.1, 1])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 10,
        "rotational_Ki": 0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.003,
        "translational_clip_z": 0.004,
        "translational_clip_neg_x": 0.003,
        "translational_clip_neg_y": 0.003,
        "translational_clip_neg_z": 0.004,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 30,
        "rotational_Ki": 10,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.1,
        "rotational_clip_y": 0.1,
        "rotational_clip_z": 0.1,
        "rotational_clip_neg_x": 0.1,
        "rotational_clip_neg_y": 0.1,
        "rotational_clip_neg_z": 0.1,
    }

class TrainConfig(DefaultTrainingConfig):
    image_keys = ['wrist_1']
    checkpoint_period = 2000
    buffer_period = 2000
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = ConnectorEnv(
                            fake_env=fake_env,
                            save_video=save_video,
                            config=EnvConfig,
                    )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque"])
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        return env