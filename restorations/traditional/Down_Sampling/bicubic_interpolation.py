import cv2

# ... (your existing functions like apply_median_filter, apply_nlm)

def apply_bicubic_interpolation(image, scale_factor=4):
    height, width = image.shape[:2]
    new_height, new_width = height * scale_factor, width * scale_factor
    
    # cv2.INTER_CUBIC is the standard Bicubic interpolation
    upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return upscaled_image