from ultralytics import YOLO
from typing import List, Union, Optional, Dict, Any
import numpy as np

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F

def ResizePad(target_size):
    def _resize_pad_transform(image):
        # target_size: (height, width)
        h, w = image.shape[-2:]
        target_h, target_w = target_size
        
        scale = min(target_w / w, target_h / h)
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = T.functional.resize(image, (new_h, new_w), antialias=True)
        
        pad_l = (target_w - new_w) // 2
        pad_t = (target_h - new_h) // 2
        pad_r = target_w - new_w - pad_l
        pad_b = target_h - new_h - pad_t
        
        # padding: (left, top, right, bottom)
        return F.pad(image, (pad_l, pad_t, pad_r, pad_b), fill=0)
    return _resize_pad_transform

default_body_transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    ResizePad((512, 384))
])

class PersonDetectAndCrop:
    def __init__(self, person_model_path: str, hand_model_path: str, device: str = 'cuda'):
        self.device = device
        self.person_detector = YOLO(person_model_path)
        self.hand_detector = YOLO(hand_model_path)
    
    def _extract_keypoints(self, keypoints_obj, idx: int) -> Optional[np.ndarray]:
        """Helper to extract keypoints and confidence."""
        if keypoints_obj is None:
            return None
            
        # Check if keypoints exist for this detection
        if len(keypoints_obj) <= idx:
            return None

        kpts = keypoints_obj[idx].xy.cpu().numpy()[0]  # [num_kpts, 2]
        
        if keypoints_obj[idx].conf is not None:
            conf = keypoints_obj[idx].conf.cpu().numpy()[0]  # [num_kpts]
            return np.concatenate([kpts, conf[:, None]], axis=1)  # [num_kpts, 3]
        
        return kpts

    def predict(
        self, 
        img_path: Union[str, np.ndarray],
        person_conf: float = 0.25, 
        hand_conf: float = 0.3,
        padding: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Detect persons and their hands in an image.

        Args:
            img_path: Path to image or numpy array (BGR).
            person_conf: Confidence threshold for person detection.
            hand_conf: Confidence threshold for hand detection.
            padding: Padding around person crop.

        Returns:
            List of dicts for each person:
            {
                'person_crop': np.ndarray,
                'person_keypoints': np.ndarray or None,  # shape: [num_kpts, 3] (x, y, conf) relative to person_crop
                'right_hand_crop': np.ndarray or None,
                'left_hand_crop': np.ndarray or None,
                'right_hand_keypoints': np.ndarray or None, # relative to right_hand_crop
                'left_hand_keypoints': np.ndarray or None   # relative to left_hand_crop
            }
        """
        # 1. Detect persons
        person_results = self.person_detector.predict(
            source=img_path,
            classes=[0],
            conf=person_conf,
            iou=0.45,
            max_det=1,  # only detect one person per image
            verbose=False,
            device=self.device
        )
        
        if not person_results:
            return []
            
        person_pred = person_results[0]
        
        if len(person_pred.boxes) == 0:
            return []
        
        # Use original image from YOLO results
        img = person_pred.orig_img
        h, w = img.shape[:2]
        
        # 2. Collect person crops and keypoints
        person_data = []
        for idx, person_box in enumerate(person_pred.boxes):
            x1, y1, x2, y2 = map(int, person_box.xyxy[0])
            
            # Get person keypoints if available
            person_keypoints = self._extract_keypoints(person_pred.keypoints, idx)
            
            # Add padding
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            person_crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Transform person keypoints to crop coordinates
            person_keypoints_crop = None
            if person_keypoints is not None:
                person_keypoints_crop = person_keypoints.copy()
                person_keypoints_crop[:, 0] -= x1_pad  # Adjust x
                person_keypoints_crop[:, 1] -= y1_pad  # Adjust y
            
            person_data.append({
                'crop': person_crop,
                'keypoints': person_keypoints_crop,
                'offset': (x1_pad, y1_pad)
            })
        
        # 3. Batch detect hands in all person crops
        person_crops = [p['crop'] for p in person_data]
        
        if not person_crops:
            return []

        hand_results = self.hand_detector.predict(
            source=person_crops,
            classes=[0, 1],  # 0: left, 1: right
            conf=hand_conf,
            verbose=False,
            device=self.device
        )
        
        # 4. Extract hand crops and keypoints for each person
        results = []
        for person_info, hand_out in zip(person_data, hand_results):
            result = {
                'person_keypoints': person_info['keypoints'],
                'right_hand_crop': None,
                'left_hand_crop': None,
                'right_hand_keypoints': None,
                'left_hand_keypoints': None
            }
            
            if hand_out.boxes:
                for hand_idx, hand_box in enumerate(hand_out.boxes):
                    hx1, hy1, hx2, hy2 = map(int, hand_box.xyxy[0])
                    hand_class = int(hand_box.cls)
                    
                    # Ensure coordinates are within bounds
                    ph, pw = person_info['crop'].shape[:2]
                    hx1, hy1 = max(0, hx1), max(0, hy1)
                    hx2, hy2 = min(pw, hx2), min(ph, hy2)
                    
                    if hx1 >= hx2 or hy1 >= hy2:
                        continue

                    # Crop hand from person image
                    hand_crop = person_info['crop'][hy1:hy2, hx1:hx2]
                    
                    # Get keypoints if available
                    keypoints = self._extract_keypoints(hand_out.keypoints, hand_idx)
                    
                    # Transform hand keypoints to hand crop coordinates
                    if keypoints is not None:
                        keypoints[:, 0] -= hx1
                        keypoints[:, 1] -= hy1
                    
                    if hand_class == 0:  # Left hand
                        result['left_hand_crop'] = hand_crop
                        result['left_hand_keypoints'] = keypoints
                    elif hand_class == 1:  # Right hand
                        result['right_hand_crop'] = hand_crop
                        result['right_hand_keypoints'] = keypoints
            
            result['person_crop'] = default_body_transform(person_info['crop']).unsqueeze(0)

            results.append(result)
        
        return results