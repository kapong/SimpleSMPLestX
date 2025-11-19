import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from utils.transforms import rot6d_to_axis_angle

class Model(nn.Module):
    def __init__(self, encoder, decoder, input_body_shape=(256, 192), focal=(5000,5000), camera_3d_size=2.5):
        super(Model, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.input_body_shape = input_body_shape
        self.focal = focal
        self.camera_3d_size = camera_3d_size


    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(self.focal[0] * self.focal[1] * 
                            self.camera_3d_size * self.camera_3d_size / (
                self.input_body_shape[0] * self.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def forward(self, img):
        body_img = F.interpolate(img, self.input_body_shape)

        # 1. Encoder
        img_feat, task_tokens = self.encoder(body_img)  # task_token:[bs, N, c]

        # 2. Decoder
        pred_mano_params = self.decoder(task_tokens, img_feat)

        # get transl
        body_trans = self.get_camera_trans(pred_mano_params['body_cam'])

        # convert predicted rot6d to aa
        root_pose_aa = rot6d_to_axis_angle(pred_mano_params['body_root_pose'])
        body_pose_aa = rot6d_to_axis_angle(pred_mano_params['body_pose'].reshape(-1, 6)).reshape(pred_mano_params['body_pose'].shape[0], -1)
        lhand_pose_aa = rot6d_to_axis_angle(pred_mano_params['lhand_pose'].reshape(-1, 6)).reshape(pred_mano_params['lhand_pose'].shape[0], -1)
        rhand_pose_aa = rot6d_to_axis_angle(pred_mano_params['rhand_pose'].reshape(-1, 6)).reshape(pred_mano_params['rhand_pose'].shape[0], -1)
        face_jaw_pose_aa = rot6d_to_axis_angle(pred_mano_params['face_jaw_pose'])
        
        out = {}
        out['smplx_root_pose'] = root_pose_aa
        out['smplx_body_pose'] = body_pose_aa
        out['smplx_lhand_pose'] = lhand_pose_aa
        out['smplx_rhand_pose'] = rhand_pose_aa
        out['smplx_jaw_pose'] = face_jaw_pose_aa
        out['smplx_shape'] = pred_mano_params['body_betas']
        out['smplx_expr'] = pred_mano_params['face_expression']
        out['cam_trans'] = body_trans
        
        return out