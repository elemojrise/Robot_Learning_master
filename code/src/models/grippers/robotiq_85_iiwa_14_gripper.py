"""
6-DoF gripper with its open/close variant
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel
import os


class Robotiq85GripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        loc = os.getcwd()
        if loc == '/home/kukauser/dev_ws': #if we are using the MANULAB computer and running policy node from dev_ws
            loc = '/home/kukauser/Robot_Learning_master/code'
        loc = loc + "/src/models/assets/"
        super().__init__(xml_path_completion(loc + "grippers/robotiq_gripper_85_lab.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([-0.026, -0.267, -0.200, -0.026, -0.267, -0.200])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision"
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision"
            ],
            "left_fingerpad": [
                "left_fingerpad_collision"
            ],
            "right_fingerpad": [
                "right_fingerpad_collision"
            ],
        }


class Robotiq85Gripper_iiwa_14(Robotiq85GripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        if action <= 0.5:
            sign = -1
        else: sign = 1

        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * sign, 0.0, 1.0)
        #print("formatting", action, "to", self.current_action)
        return self.current_action

    @property
    def speed(self):
        return 0.02

    @property
    def dof(self):
        return 1
