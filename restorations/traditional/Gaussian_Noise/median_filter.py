import cv2
import numpy as np

def apply_median_filter(image_rgb, kernel_size=5):
        
    #convert RGB to BGR for openCV processing
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    restored_bgr = cv2.medianBlur(image_bgr, kernel_size)
    
    #convert back to RGB
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    
    return restored_rgb
