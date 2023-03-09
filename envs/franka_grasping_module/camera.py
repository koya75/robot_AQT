from pathlib import Path

from isaacgym import gymapi
from isaacgym import gymtorch
import math
import numpy as np

from isaacgym.torch_utils import *


class Camera:
    def create(self, gym, sim, width, height, image_type, hand, demo):
        self.gym = gym
        self.sim = sim
        self.hand = hand
        self.cam_props = gymapi.CameraProperties()
        self.cam_props.width = width
        self.cam_props.height = height
        self.cam_props.enable_tensors = True
        self.cam_props.horizontal_fov = 100

        if demo:
            self.bard_cam_props = gymapi.CameraProperties()
            self.cam_props.width = width
            self.cam_props.height = height
            self.bard_cam_props.enable_tensors = True
            self.bard_cam_props.horizontal_fov = 30

        self.camera_pos = gymapi.Vec3(0.4, 0.4, 1.2)
        self.camera_target = gymapi.Vec3(0, 0.4, 0.6)
        """self.camera_pos = gymapi.Vec3(0.0, 1.0, 2.0)
        self.camera_target = gymapi.Vec3(0, 0.4, 0.5)"""

        self.local_transform = gymapi.Transform()
        self.local_transform.p = gymapi.Vec3(0.03,0.0,0.06)#0.03,-0.037,0.14
        self.local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(-90.0))#*gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(225.0))

        self.image_type =image_type
        
    def add(self, env, hand_handle):
        # add camera
        cam_handle = self.gym.create_camera_sensor(env, self.cam_props)
        if self.hand:
            self.gym.attach_camera_to_body(cam_handle, env, hand_handle, self.local_transform, gymapi.FOLLOW_TRANSFORM)
        else:
            self.gym.set_camera_location(cam_handle, env, self.camera_pos, self.camera_target)

        # obtain camera tensor
        cam_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, env, cam_handle, self.image_type
        )
        cam_tensor = gymtorch.wrap_tensor(cam_tensor)
        return cam_tensor

    def bard_add(self, env):
        # add camera
        self.bard_cam_handle = self.gym.create_camera_sensor(env, self.bard_cam_props)
        self.gym.set_camera_location(self.bard_cam_handle, env, self.camera_pos, self.camera_target)

        # obtain camera tensor
        cam_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, env, self.bard_cam_handle, self.image_type
        )
        cam_tensor = gymtorch.wrap_tensor(cam_tensor)
        return cam_tensor
