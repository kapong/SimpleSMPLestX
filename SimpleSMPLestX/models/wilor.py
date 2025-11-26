import torch

from utils.transforms import rotmat_to_aa, rot6d_to_rotmat

class WiLoR(torch.nn.Module):
    def __init__(self, backbone_path: str, refinenet_path: str, mano_path:str, device ='cuda'):
        super().__init__()

        self.backbone = torch.jit.load(backbone_path).to(device)
        self.refine_net = torch.jit.load(refinenet_path).to(device)
        self.mano = torch.jit.load(mano_path).to(device)
        self.FOCAL_LENGTH = 5000
        self.to(device)

    def forward(self, batch):
        # Use RGB image as input
        batch_size = batch.shape[0]
        temp_global_orient, temp_hand_pose, temp_betas, pred_cam, pose_rot6d, vit_out = self.backbone(batch[:,:,:,32:-32]) # B, 1280, 16, 12
    
        # Compute camera translation
        device = temp_hand_pose.device
        dtype = temp_hand_pose.dtype
        focal_length = self.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        
        ## Temp MANO 
        temp_global_orient = temp_global_orient.reshape(batch_size, -1, 3, 3) #rotmat
        temp_hand_pose = temp_hand_pose.reshape(batch_size, -1, 3, 3)
        temp_betas = temp_betas.reshape(batch_size, -1)
        
        temp_vertices, _ = self.mano(temp_betas, temp_global_orient, temp_hand_pose)

        delta_pose, delta_beta, delta_cam = self.refine_net(vit_out, temp_vertices, pred_cam, focal_length)
        pose_rot6d_plus = pose_rot6d + delta_pose
        pose_aa = rot6d_to_rotmat(pose_rot6d_plus.reshape(batch_size, -1, 6))

        return {
            'global_orient': rotmat_to_aa(pose_aa[:, :1]),
            'hand_pose': rotmat_to_aa(pose_aa[:, 1:]),
            'betas': temp_betas + delta_beta,
            'pred_cam': pred_cam + delta_cam,
        }