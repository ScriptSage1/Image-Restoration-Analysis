# In restorations/deep_learning/esrgan_model.py

import cv2
import numpy as np
import torch

# Import the RRDBNet architecture from our new local file
from .rrdbnet_arch import RRDBNet 

class SuperResolutionModel:
    """
    A wrapper class for the RRDB_ESRGAN model
    using our own rrdbnet_arch.py file.
    """
    def __init__(self, model_path, scale=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Define the model architecture (standard for RRDB_ESRGAN)
        self.model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=scale
        )
        
        # Load the pre-trained weights
        loadnet = torch.load(model_path, map_location=self.device)
        
        if 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        elif 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
            
        self.model.eval()
        self.model = self.model.to(self.device)
        self.scale = scale
        
        print(f"Loaded RRDB_ESRGAN x{scale} model from {model_path}")

    def upscale(self, image):
        """
        Upscales an image using the loaded DL model.
        """
        # 1. Pre-process the image
        img_lr = image.astype(np.float32) / 255.0
        img_lr = torch.from_numpy(np.transpose(img_lr, (2, 0, 1))).float()
        img_lr = img_lr.unsqueeze(0).to(self.device)

        # 2. Run inference
        with torch.no_grad():
            output = self.model(img_lr)

        # 3. Post-process the image
        output = output.squeeze(0).clamp(0, 1).cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        return output