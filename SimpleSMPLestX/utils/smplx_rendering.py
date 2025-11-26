"""
SMPLX Video Rendering Module

This module provides a class-based interface for rendering SMPLX model parameters
into video sequences using pyrender and OpenCV.
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import numpy as np
import torch
import trimesh
import pyrender
import smplx
from tqdm.auto import tqdm
from typing import Dict, Tuple, Optional

class SMPLXRenderer:
    """
    A class for rendering SMPLX model parameters into images and video sequences.
    
    Uses a singleton pattern for the SMPLX layer to avoid repeated model loading.
    """
    
    # Class-level singleton instance
    _smplx_layer = None
    _model_config = None
    
    # Default camera configuration
    DEFAULT_CAMERA_POSE = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.3],
        [0.0, 0.0, 1.0, 1.3],
        [0.0, 0.0, 0.0, 1.0]
    ])
    DEFAULT_YFOV = np.radians(45)
    
    def __init__(
        self,
        model_path: str,
        gender: str = 'NEUTRAL',
        device: str = 'cuda'
    ):
        """
        Initialize SMPLXRenderer.
        
        Args:
            model_path: Path to SMPLX model files
            gender: Gender of the model ('MALE', 'FEMALE', or 'NEUTRAL')
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.gender = gender
        self.device = device
        
        # Load or get singleton SMPLX layer
        self._load_smplx_layer()
    
    def _load_smplx_layer(self) -> None:
        """
        Load SMPLX layer as singleton.
        
        Only creates a new layer if one doesn't exist or if configuration changed.
        """
        current_config = (self.model_path, self.gender, self.device)
        
        if SMPLXRenderer._smplx_layer is None or SMPLXRenderer._model_config != current_config:
            layer_args = {
                'create_global_orient': False,
                'create_body_pose': False,
                'create_left_hand_pose': False,
                'create_right_hand_pose': False,
                'create_jaw_pose': False,
                'create_leye_pose': False,
                'create_reye_pose': False,
                'create_betas': False,
                'create_expression': False,
                'create_transl': False,
                'flat_hand_mean': True,
            }
            
            SMPLXRenderer._smplx_layer = smplx.create(
                self.model_path,
                'smplx',
                gender=self.gender,
                use_pca=False,
                use_face_contour=True,
                ext='npz',
                **layer_args
            ).to(self.device)
            
            SMPLXRenderer._model_config = current_config
    
    @property
    def smplx_layer(self):
        """Get the singleton SMPLX layer."""
        return SMPLXRenderer._smplx_layer
    
    def create_frame(
        self,
        params: Dict[str, torch.Tensor],
        frame_idx: int = 0,
        resolution: Tuple[int, int] = (1024, 1024),
        camera_pose: Optional[np.ndarray] = None,
        yfov: Optional[float] = None
    ) -> np.ndarray:
        """
        Create a single rendered frame from SMPLX parameters.
        
        Args:
            params: Dictionary containing SMPLX parameters:
                - smplx_shape: Shape parameters (N, 10)
                - smplx_body_pose: Body pose parameters (N, 63)
                - smplx_rhand_pose: Right hand pose (N, 45)
                - smplx_lhand_pose: Left hand pose (N, 45)
                - smplx_jaw_pose: Jaw pose (N, 3)
                - smplx_expr: Expression parameters (N, 10)
            frame_idx: Index of the frame to render
            resolution: Image resolution (width, height)
            camera_pose: Camera pose matrix (4x4)
            yfov: Vertical field of view in radians
        
        Returns:
            Rendered frame as BGR numpy array
        """
        # Use default camera settings if not provided
        camera_pose = camera_pose if camera_pose is not None else self.DEFAULT_CAMERA_POSE
        yfov = yfov if yfov is not None else self.DEFAULT_YFOV
        
        # Setup rendering scene
        scene = self._setup_scene(camera_pose, yfov)
        renderer = pyrender.OffscreenRenderer(
            viewport_width=resolution[0],
            viewport_height=resolution[1]
        )
        
        try:
            # Generate mesh for the frame
            mesh = self._generate_mesh(params, frame_idx)
            
            # Render the frame
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            scene.add(mesh_pyrender)
            
            color, _ = renderer.render(scene)
            frame_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            
            return frame_bgr
        
        finally:
            # Cleanup resources
            renderer.delete()
    
    def _generate_mesh(
        self,
        params: Dict[str, torch.Tensor],
        frame_idx: int
    ) -> trimesh.Trimesh:
        """
        Generate a trimesh object from SMPLX parameters for a specific frame.
        
        Args:
            params: Dictionary containing SMPLX parameters (optional keys will default to zeros)
            frame_idx: Index of the frame to generate
        
        Returns:
            Trimesh object with centered vertices
        """
        # Prepare zero pose tensor
        zero_pose = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        
        # Helper function to get parameter or default zeros
        def get_param(key: str, default_shape: Tuple[int, ...]) -> torch.Tensor:
            if key in params and params[key] is not None:
                return params[key][frame_idx:frame_idx+1, :]
            return torch.zeros((1, default_shape[-1]), dtype=torch.float32, device=self.device)
        
        # Generate SMPLX output for current frame
        smplx_output = self.smplx_layer(
            betas=get_param('smplx_shape', (1, 10)),
            body_pose=get_param('smplx_body_pose', (1, 63)),
            right_hand_pose=get_param('smplx_rhand_pose', (1, 45)),
            left_hand_pose=get_param('smplx_lhand_pose', (1, 45)),
            jaw_pose=get_param('smplx_jaw_pose', (1, 3)),
            expression=get_param('smplx_expr', (1, 10)),
            global_orient=zero_pose,
            transl=zero_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            return_full_pose=True
        )
        
        # Process mesh
        vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
        mesh = trimesh.Trimesh(vertices, self.smplx_layer.faces)
        mesh.vertices -= mesh.center_mass
        
        return mesh
    
    def create_video(
        self,
        params: Dict[str, torch.Tensor],
        output_video_path: str,
        fps: int = 25,
        resolution: Tuple[int, int] = (1024, 1024),
        camera_pose: Optional[np.ndarray] = None,
        yfov: Optional[float] = None
    ) -> Tuple[bool, np.ndarray]:
        """
        Create a video from SMPLX parameters.
        
        Args:
            params: Dictionary containing SMPLX parameters:
                - smplx_shape: Shape parameters (N, 10)
                - smplx_body_pose: Body pose parameters (N, 63)
                - smplx_rhand_pose: Right hand pose (N, 45)
                - smplx_lhand_pose: Left hand pose (N, 45)
                - smplx_jaw_pose: Jaw pose (N, 3)
                - smplx_expr: Expression parameters (N, 10)
            output_video_path: Path to save the output video
            fps: Frames per second for the output video
            resolution: Video resolution (width, height)
            camera_pose: Camera pose matrix (4x4)
            yfov: Vertical field of view in radians
        
        Returns:
            Tuple of (success, last_frame)
        """
        # Use default camera settings if not provided
        camera_pose = camera_pose if camera_pose is not None else self.DEFAULT_CAMERA_POSE
        yfov = yfov if yfov is not None else self.DEFAULT_YFOV
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, resolution)
        
        # Setup rendering scene
        scene = self._setup_scene(camera_pose, yfov)
        renderer = pyrender.OffscreenRenderer(
            viewport_width=resolution[0],
            viewport_height=resolution[1]
        )
        
        # Render frames
        mesh_node = None
        last_frame = None
        num_frames = params['smplx_body_pose'].shape[0]
        
        try:
            for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
                # Generate mesh for the frame
                mesh = self._generate_mesh(params, frame_idx)
                
                # Update scene with new mesh
                mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=True)
                if mesh_node is not None:
                    scene.remove_node(mesh_node)
                mesh_node = scene.add(mesh_pyrender)
                
                # Render and write frame
                color, _ = renderer.render(scene)
                frame_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
                last_frame = frame_bgr
        
        finally:
            # Cleanup resources
            video_writer.release()
            renderer.delete()
        
        return True, last_frame
    
    def _setup_scene(
        self,
        camera_pose: np.ndarray,
        yfov: float
    ) -> pyrender.Scene:
        """
        Setup pyrender scene with camera and lighting.
        
        Args:
            camera_pose: Camera pose matrix (4x4)
            yfov: Vertical field of view in radians
        
        Returns:
            Configured pyrender Scene
        """
        scene = pyrender.Scene(
            ambient_light=[0.3, 0.3, 0.3],
            bg_color=[0.0, 0.1, 0.3, 1.0]
        )
        
        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=yfov)
        scene.add(camera, pose=camera_pose)
        
        # Add directional light
        light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=2.0
        )
        scene.add(light, pose=camera_pose)
        
        return scene
    
    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton SMPLX layer (useful for testing or reloading)."""
        cls._smplx_layer = None
        cls._model_config = None