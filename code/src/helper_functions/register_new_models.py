from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.robots import ROBOT_CLASS_MAPPING

from robosuite.robots.single_arm import SingleArm

#Funksjon for å registrer custom made gripper
def register_gripper(target_class):
    GRIPPER_MAPPING[target_class.__name__] = target_class

#Funksjon for å registrer det nye robotnavnet
def register_robot_class_mapping(robot_name):
    if robot_name == "IIWA_14" or robot_name == "IIWA_14_modified":
        ROBOT_CLASS_MAPPING[robot_name] = SingleArm
    else:
        raise ValueError("{robot_name}, is not a defined robot")