import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion
import os


class IIWA_14(ManipulatorModel):
    """
    IIWA is a bright and spunky robot created by KUKA

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        loc = os.getcwd()
        if loc == '/home/kukauser/dev_ws': #if we are using the MANULAB computer and running policy node from dev_ws
            loc = '/home/kukauser/Robot_Learning_master/code'
        loc = loc + "/src/models/assets/"
        super().__init__(xml_path_completion(loc + "robots/iiwa_14/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "Robotiq140Gripper"

    @property
    def default_controller_config(self):
        return "default_iiwa"

    @property
    def init_qpos(self):
        return np.array([0.000, 0.85, 0.000, -1.5, 0.000, 0.600, 0.000]) # [0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length/2, 0, 0)
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"



class IIWA_14_modified(ManipulatorModel):
    """
    IIWA is a bright and spunky robot created by KUKA

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        loc = os.getcwd()
        if loc == '/home/kukauser/dev_ws': #if we are using the MANULAB computer and running policy node from dev_ws
            loc = '/home/kukauser/Robot_Learning_master/code'
        loc = loc + "/src/models/assets/"
        super().__init__(xml_path_completion(loc + "robots/iiwa_14_modified/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "Robotiq140Gripper"

    @property
    def default_controller_config(self):
        return "default_iiwa"

    @property
    def init_qpos(self):
        return np.array([0.000, 0.85, 0.000, -1.5, 0.000, 0.600, 0.000]) # [0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length/2, 0, 0)
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"



class IIWA_14_modified_flange(ManipulatorModel):
    """
    IIWA is a bright and spunky robot created by KUKA

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        loc = os.getcwd()
        if loc == '/home/kukauser/dev_ws': #if we are using the MANULAB computer and running policy node from dev_ws
            loc = '/home/kukauser/Robot_Learning_master/code'
        loc = loc + "/src/models/assets/"
        super().__init__(xml_path_completion(loc + "robots/iiwa_14_modified_flange/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "Robotiq140Gripper"

    @property
    def default_controller_config(self):
        return "default_iiwa"

    @property
    def init_qpos(self):
        return np.array([0.00, 0.88, 0.00, -1.30, 0.00, 0.90, 0.00]) # Petter forslag [0.00, 0.88, 0.00, -1.30, 0.00, 0.90, 0.00],  [0.00, 0.88, 0.00, -1.27, 0.00, 0.90, 0.00], [0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length/2, 0, 0)
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"

class IIWA_14_modified_flange_multi(ManipulatorModel):
    """
    IIWA is a bright and spunky robot created by KUKA

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        loc = os.getcwd()
        if loc == '/home/kukauser/dev_ws': #if we are using the MANULAB computer and running policy node from dev_ws
            loc = '/home/kukauser/Robot_Learning_master/code'
        loc = loc + "/src/models/assets/"
        super().__init__(xml_path_completion(loc + "robots/iiwa_14_modified_flange/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "Robotiq140Gripper"

    @property
    def default_controller_config(self):
        return "default_iiwa"

    @property
    def init_qpos(self):
        return np.array([0.00, 0.72, 0.00, -1.28, 0.00, 1.00, 0.00]) # Petter forslag [0.00, 0.88, 0.00, -1.30, 0.00, 0.90, 0.00],  [0.00, 0.88, 0.00, -1.27, 0.00, 0.90, 0.00], [0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000])
 #[0.00, 0.72, 0.00, -1.28, 0.00, 1.00, 0.00]
    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length/2, 0, 0)
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"