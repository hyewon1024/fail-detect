import sys
import os
import torch
import cv2
import numpy as np
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F
# Detection 모델 라이브러리 임포트
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_utils.setting_config import device
from task_utils.setting_config import env

# Contact-GraspNet 모델 라이브러리 임포트
# from cgnet.utils.config import cfg_from_yaml_file
# from cgnet.tools import builder
# from cgnet.inference_cgnet import inference_cgnet

import carb
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/isaaclab/cameras_enabled", True)


DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(DIR_PATH, 'data/checkpoint/maskrcnn_ckpt/maskrcnn_trained_model_refined.pth') # <-- 사전 학습된 Weight

def depth2pc(depth, K, rgb=None):
    """ 뎁스 이미지를 포인트 클라우드로 변환하는 함수 """
    
    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32)-K[0,2])
    normalized_y = (y.astype(np.float32)-K[1,2])
    
    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]
    
    if rgb is not None:
        rgb = rgb[y, x]
    
    pc = np.vstack([world_x, world_y, world_z]).T
    return (pc, rgb)

def get_world_bbox(depth, K, bb): 
    """ Bounding Box의 좌표를 포인트 클라우드 기준 좌표로 변환하는 함수 """

    image_width = depth.shape[1]
    image_height = depth.shape[0]

    x_min, x_max = bb[0], bb[2]
    y_min, y_max = bb[1], bb[3]
    
    if y_min < 0:
        y_min = 0
    if y_max >= image_height:
        y_max = image_height-1
    if x_min < 0:
        x_min = 0
    if x_max >=image_width:
        x_max = image_width-1

    z_0, z_1 = depth[int(y_min), int(x_min)], depth[int(y_max), int(x_max)]
    
    def to_world(x, y, z):
        """ 뎁스 포인트를 3D 포인트로 변환하는 함수 """
        world_x = (x - K[0, 2]) * z / K[0, 0]
        world_y = (y - K[1, 2]) * z / K[1, 1]
        return world_x, world_y, z
    
    x_min_w, y_min_w, z_min_w = to_world(x_min, y_min, z_0)
    x_max_w, y_max_w, z_max_w = to_world(x_max, y_max, z_1)
    
    return x_min_w, y_min_w, x_max_w, y_max_w

def get_model_instance_segmentation(num_classes):
    """ 학습 때와 동일한 구조로 Mask R-CNN 모델을 생성합니다. """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def get_random_color():
    """ 시각화를 위해 랜덤 RGB 색상을 생성합니다. """
    return [random.randint(50, 255) for _ in range(3)] # 너무 어둡지 않은 색상
# # Detection 모델 로드
# NUM_CLASSES = 79
# detection_model = get_model_instance_segmentation(NUM_CLASSES)
# detection_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# detection_model.eval()
# detection_model.to(device)

# # Contact-GraspNet 모델 config를 불러오기 위한 경로 설정
# grasp_model_config_path = os.path.join(DIR_PATH, 'cgnet/configs/config.yaml')
# grasp_model_config = cfg_from_yaml_file(grasp_model_config_path)

# # Contact-GraspNet 모델 선언 및 checkpoint 입력을 통한 모델 weight 로드
# grasp_model = builder.model_builder(grasp_model_config.model)
# grasp_model_path = os.path.join(DIR_PATH, 'data/checkpoint/contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth')
# builder.load_model(grasp_model, grasp_model_path)
# grasp_model.to(device)
# grasp_model.eval()

# def load_cam():
    
    
    

# load_cam()
#print("sensors:", list(env.unwrapped.scene.sensors.keys()))
robot_camera = env.unwrapped.scene.sensors['camera']
# front_camera = env.unwrapped.scene.sensors['front_camera']
robot_camera_K = robot_camera.data.intrinsic_matrices.squeeze().cpu().numpy()

# num_envs = env.num_envs                  # 또는 self.num_envs
# env_ids = torch.arange(num_envs, device=device, dtype=torch.int32)

# # (1) Euler → Quaternion (XYZ 순서, degree 단위)
# quat_xyzw = rot_utils.euler_angles_to_quats(
#     np.array([88, 0, 90]), degrees=True
# )  # numpy array shape (4,)

# # (2) Tensor 변환 및 반복
# quat = torch.tensor(quat_xyzw, dtype=torch.float32, device=device).repeat(num_envs, 1)
#orientation에는 2가지 문제가 있다. 첫째는 orientation을 잘 맞춰야 한다는 것이다. 아마 뒤로 돌려야될 것 같다. 두번째는 env당 별개로 분리되어 있지 않다. world 좌표계 기준으로 세팅하는 데에 주의가 필요하다. 


# # (3) 위치 벡터
# pos = torch.tensor([[3.7245, 0.218, 1.053]], dtype=torch.float32, device=device).repeat(num_envs, 1)

# # (4) 적용
# front_camera.set_world_poses(positions=pos, orientations=quat, env_ids=env_ids)
