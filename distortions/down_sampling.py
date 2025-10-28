# In distortions/down_sampling.py

import cv2 as cv

def apply_DownSampling(image, scale_factor=2):
    """
    Creates a low-resolution image by downsampling.
    """
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    new_h = h // scale_factor
    new_w = w // scale_factor
    
    # Create the low-resolution image
    image_down = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
    
    # Return ONLY the small H/2, W/2 image
    return image_down