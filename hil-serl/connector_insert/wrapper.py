import time
from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
import gymnasium as gym
import numpy as np
import requests

class ConnectorEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def go_to_rest(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.1)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        if self.currpos[2] < 0.15:
            self.currpos[2] += 0.03
            self.interpolate_move(self.currpos, timeout=0.5)
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)